import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import IoU, transform_coord


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
            5 => [4 offsets (xc, yc, w, h), objectness]
        - y_true: (Tensor)

        Returns
        - detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
        """
        bs, _, gs, _ = x.size()
        stride = self.reso // gs  # no pooling used, stride is the only downsample
        num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
        nA = len(self.anchors)
        scaled_anchors = torch.Tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])

        # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, num_attrs, gs, gs).permute(0, 1, 3, 4, 2).contiguous()

        pred_tx = torch.sigmoid(x[..., 0])       # center relative to (i,j)
        pred_ty = torch.sigmoid(x[..., 1])       # center relative to (i,j)
        pred_tw = x[..., 2]                      # tw
        pred_th = x[..., 3]                      # th
        pred_conf = torch.sigmoid(x[..., 4])     # objectness
        # if self.training == True:
        #     pred_cls = x[..., 5:]  # softmax in cross entropy
        # else:
        #     pred_cls = F.softmax(x[..., 5:], dim=-1)  # class
        pred_cls = torch.sigmoid(x[..., 5:]).cuda()  # sigmoid to avoid inter-class competetion

        grid_x = torch.arange(gs).repeat(gs, 1).view([1, 1, gs, gs]).float().cuda()
        grid_y = torch.arange(gs).repeat(gs, 1).t().view([1, 1, gs, gs]).float().cuda()
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1)).cuda()
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1)).cuda()

        detections = torch.Tensor(bs, nA, gs, gs, num_attrs)
        detections[..., 0] = pred_tx + grid_x
        detections[..., 1] = pred_ty + grid_y
        detections[..., 2] = torch.exp(pred_tw) * anchor_w
        detections[..., 3] = torch.exp(pred_th) * anchor_h
        detections[..., :4] *= stride  # scale relative to feature map
        detections[..., 4] = pred_conf
        detections[..., 5:] = pred_cls
        pred_bbox = detections[..., 0:4]

        if self.training == False:
            return detections
        else:
            gt_tx = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_ty = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_tw = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            gt_th = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            pos_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            ignore_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            cls_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False).cuda()
            for batch_idx in range(bs):
                for box_idx, y_true_one in enumerate(y_true[batch_idx]):
                    y_true_one = y_true_one.cuda()
                    gt_bbox = y_true_one[:4] * gs  # scale bbox relative to feature map
                    gt_cls_label = y_true_one[4]

                    gt_xc, gt_yc, gt_w, gt_h = gt_bbox[0:4]
                    gt_i = torch.clamp(gt_xc.long(), min=0, max=gs - 1)
                    gt_j = torch.clamp(gt_yc.long(), min=0, max=gs - 1)

                    gt_box_shape = torch.Tensor([0, 0, gt_w, gt_h]).unsqueeze(0)
                    anchor_bboxes = torch.zeros(3, 4)
                    anchor_bboxes[:, 2:] = scaled_anchors
                    anchor_ious = IoU(gt_box_shape, anchor_bboxes, format='center')
                    best_anchor = np.argmax(anchor_ious).item()
                    anchor_w, anchor_h = scaled_anchors[best_anchor]

                    gt_tw[batch_idx, best_anchor, gt_j, gt_i] = torch.log(gt_w / anchor_w.cuda() + 1e-16)
                    gt_th[batch_idx, best_anchor, gt_j, gt_i] = torch.log(gt_h / anchor_h.cuda() + 1e-16)
                    gt_tx[batch_idx, best_anchor, gt_j, gt_i] = gt_xc - gt_i.float()
                    gt_ty[batch_idx, best_anchor, gt_j, gt_i] = gt_yc - gt_j.float()

                    pos_mask[batch_idx, best_anchor, gt_j, gt_i] = 1
                    cls_mask[batch_idx, best_anchor, gt_j, gt_i] = int(gt_cls_label)

                    from IPython import embed
                    embed()

            MSELoss = nn.MSELoss()
            BCELoss = nn.BCELoss()
            CELoss = nn.CrossEntropyLoss()

            loss = dict()
            loss['x'] = BCELoss(pred_tx[pos_mask == 1], gt_tx[pos_mask == 1])
            loss['y'] = BCELoss(pred_ty[pos_mask == 1], gt_ty[pos_mask == 1])
            loss['w'] = MSELoss(pred_tw[pos_mask == 1], gt_tw[pos_mask == 1])
            loss['h'] = MSELoss(pred_th[pos_mask == 1], gt_th[pos_mask == 1])
            # loss['cls'] = MSELoss(pred_cls[pos_mask == 1], cls_mask[pos_mask == 1])
            loss['cls'] = CELoss(pred_cls[pos_mask == 1].view(-1, self.num_classes), cls_mask[pos_mask == 1].view(-1).long())
            # loss['cls'] = BCELoss(pred_cls[pos_mask == 1], cls_mask[pos_mask == 1])
            loss['conf'] = BCELoss(pred_conf[pos_mask == 1], pos_mask[pos_mask == 1])
            return loss


class NMSLayer(nn.Module):
    """
    NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection

    Args    
      conf_thresh: (float) fore-ground confidence threshold
      nms_thresh: (float) nms threshold
    """

    def __init__(self, conf_thresh=0.9, nms_thresh=0.3, cls_thresh=0.5):
        super(NMSLayer, self).__init__()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.cls_thresh = cls_thresh

    def forward(self, x):
        """
        Args
          x: (Tensor) detection feature map, with size [bs, num_bboxes, [x,y,w,h,p_obj]+num_classes]

        Returns
          detections: (Tensor) detection result with size [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
        """
        bs, num_bboxes, num_attrs = x.size()
        detections = torch.Tensor().cuda()

        for idx in range(bs):
            pred = x[idx]

            try:
                non_zero_pred = pred[pred[:, 4] > self.conf_thresh]
                non_zero_pred[:, :4] = transform_coord(non_zero_pred[:, :4], src='center', dst='corner')
                max_score, max_idx = torch.max(non_zero_pred[:, 5:], 1)
                max_idx = max_idx.float().unsqueeze(1)
                max_score = max_score.float().unsqueeze(1)
                non_zero_pred = torch.cat((non_zero_pred[:, :5], max_score, max_idx), 1)
                # from IPython import embed
                # embed()
                # non_zero_pred = non_zero_pred[non_zero_pred[:, 5] > self.cls_thresh]
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
                    # ious1, ious2 = IoU(max_preds[-1], cls_pred, type='part')
                    # cls_pred = cls_pred[(ious < self.nms_thresh) * (ious2 == 2)]

                if len(max_preds) > 0:
                    max_preds = torch.cat(max_preds).data
                    batch_idx = max_preds.new(max_preds.size(0), 1).fill_(idx)
                    seq = (batch_idx, max_preds)
                    detections = torch.cat(seq, 1) if detections.size(0) == 0 else torch.cat((detections, torch.cat(seq, 1)))

        return detections
