import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import IoU, transform_coord, xywh2xyxy


class MaxPool1s(nn.Module):
    """Max pooling layer with stride 1"""

    def __init__(self, kernel_size):
        super(MaxPool1s, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    """Empty layer for shortcut connection"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """Detection layer

    Args
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int) # classes
      reso: (int) original image resolution
      ignore_thresh: (float)
    """

    def __init__(self, anchors, num_classes, reso, ignore_thresh):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.reso = reso
        self.ignore_thresh = ignore_thresh

    def forward(self, x, y_true=None):
        """
        Transform feature map into 2-D tensor. Transformation includes
        1. Re-organize tensor to make each row correspond to a bbox
        2. Transform center coordinates
          bx = sigmoid(tx) + cx
          by = sigmoid(ty) + cy
        3. Transform width and height
          bw = pw * exp(tw)
          bh = ph * exp(th)
        4. Activation

        Args
        - x: (Tensor) feature map with size [bs, (5+nC)*nA, gs, gs]
            bs = batch size
            gs = grid size
            nC = number of classes
            nA = number of anchors
        - y_true: (Tensor)

        Returns
        - detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
        """
        bs, _, gs, _ = x.size()
        stride = self.reso // gs
        nC = self.num_classes
        nA = len(self.anchors)
        anchors = torch.Tensor([(w / stride, h / stride)
                                for w, h in self.anchors])

        # [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, 5 + nC, gs, gs).permute(0, 1, 3, 4, 2)

        '''prediction'''
        pred_tx = torch.sigmoid(x[..., 0])       # sigmoid(tx)
        pred_ty = torch.sigmoid(x[..., 1])       # sigmoid(ty)
        pred_tw = x[..., 2]                      # tw
        pred_th = x[..., 3]                      # th
        pred_conf = torch.sigmoid(x[..., 4])     # objectness
        pred_cls = torch.sigmoid(x[..., 5:])     # class

        '''detection result'''
        detections = torch.Tensor(bs, nA, gs, gs, 5 + nC)
        grid_x = torch.arange(gs).repeat(gs, 1).view([1, 1, gs, gs])
        grid_y = torch.arange(gs).repeat(gs, 1).t().view([1, 1, gs, gs])
        anchor_w = anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = anchors[:, 1:2].view((1, nA, 1, 1))

        detections[..., 0] = pred_tx + grid_x.float().cuda()
        detections[..., 1] = pred_ty + grid_y.float().cuda()
        detections[..., 2] = torch.exp(pred_tw) * anchor_w.cuda()
        detections[..., 3] = torch.exp(pred_th) * anchor_h.cuda()
        detections[..., :4] *= stride
        detections[..., 4] = pred_conf
        detections[..., 5:] = pred_cls
        pred_bbox = detections[..., 0:4]

        if self.training == False:
            return detections.view(bs, -1, 5 + nC)
        else:
            mask_shape = bs, nA, gs, gs
            coord_mask = torch.zeros(mask_shape).cuda()
            conf_mask = torch.zeros(mask_shape).cuda()
            cls_mask = torch.zeros(mask_shape).cuda()


class NMSLayer(nn.Module):
    """
    NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection
    """

    def __init__(self, conf_thresh=0.5, nms_thresh=0.5, cls_thresh=0.5):
        """
        Args:
        - conf_thresh: (float) fore-ground confidence threshold
        - nms_thresh: (float) nms threshold
        """
        super(NMSLayer, self).__init__()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # self.cls_thresh = cls_thresh

    def forward(self, x):
        """
        Args
          x: (Tensor) detection feature map, with size [bs, num_bboxes, 5 + nC]

        Returns
          detections: (Tensor) detection result with size [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
        """
        bs, num_bboxes, num_attrs = x.size()
        detections = torch.Tensor().cuda()

        for idx in range(bs):
            pred = x[idx]

            try:
                non_zero_pred = pred[pred[:, 4] > self.conf_thresh]
                non_zero_pred[:, :4] = xywh2xyxy(non_zero_pred[:, :4])
                max_score, max_idx = torch.max(non_zero_pred[:, 5:], 1)
                max_idx = max_idx.float().unsqueeze(1)
                max_score = max_score.float().unsqueeze(1)
                non_zero_pred = torch.cat((non_zero_pred[:, :5], max_score, max_idx), 1)
                classes = torch.unique(non_zero_pred[:, -1])
            except Exception:  # no object detected
                continue

            for cls in classes:
                cls_pred = non_zero_pred[non_zero_pred[:, -1] == cls]
                conf_sort_idx = torch.sort(cls_pred[:, 5], descending=True)[1]
                cls_pred = cls_pred[conf_sort_idx]
                max_preds = []
                while cls_pred.size(0) > 0:
                    max_preds.append(cls_pred[0].unsqueeze(0))
                    ious = IoU(max_preds[-1], cls_pred)
                    cls_pred = cls_pred[ious < self.nms_thresh]

                if len(max_preds) > 0:
                    max_preds = torch.cat(max_preds).data
                    batch_idx = max_preds.new(max_preds.size(0), 1).fill_(idx)
                    seq = (batch_idx, max_preds)
                    detections = torch.cat(seq, 1) if detections.size(0) == 0 else torch.cat((detections, torch.cat(seq, 1)))

        return detections
