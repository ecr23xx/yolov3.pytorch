import os
import torch
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyemojify import emojify
from PIL import Image, ImageFont, ImageDraw
opj = os.path.join

import config


def parse_cfg(cfgfile):
    """Parse a configuration file

    Args
      cfgfile: (str) path to config file

    Returns
      blocks: (list) list of blocks, with each block describes a block in the NN to be built
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # skip empty lines
    lines = [x for x in lines if x[0] != '#']  # skip comment
    lines = [x.rstrip().lstrip() for x in lines]
    file.close()

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def transform_coord(bbox, src='center', dst='corner'):
    """Transform bbox coordinates
      |---------|           (x1,y1) *---------|
      |         |                   |         |
      |  (x,y)  h                   |         |
      |         |                   |         |
      |____w____|                   |_________* (x2,y2)
         center                         corner

    Args
      bbox: (Tensor) bbox with size [..., 4]

    Returns
      bbox_transformed: (Tensor) bbox with size [..., 4]
    """
    flag = False
    if len(bbox.size()) == 1:
        bbox = bbox.unsqueeze(0)
        flag = True

    bbox_transformed = bbox.new(bbox.size())
    if src == 'center' and dst == 'corner':
        bbox_transformed[..., 0] = (bbox[..., 0] - bbox[..., 2]/2)
        bbox_transformed[..., 1] = (bbox[..., 1] - bbox[..., 3]/2)
        bbox_transformed[..., 2] = (bbox[..., 0] + bbox[..., 2]/2)
        bbox_transformed[..., 3] = (bbox[..., 1] + bbox[..., 3]/2)
    elif src == 'corner' and dst == 'center':
        bbox_transformed[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
        bbox_transformed[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
        bbox_transformed[..., 2] = bbox[..., 2] - bbox[..., 0]
        bbox_transformed[..., 3] = bbox[..., 3] + bbox[..., 1]
    else:
        raise Exception(emojify("format not supported! :shit:"))

    if flag == True:
        bbox_transformed = bbox_transformed.squeeze(0)

    return bbox_transformed


def IoU(box1, box2, format='corner', type='full'):
    """Compute IoU between box1 and box2

    Args
    - box: (torch.cuda.Tensor) bboxes with size [# bboxes, 4]
    - format: (str) bbox format
        'corner' => [x1, y1, x2, y2]
        'center' => [xc, yc, w, h]
    - type: (str) IoU type
        'full' => regular IoU
        'part' => intersect / one bbox area
    """
    if box1.is_cuda == True:
        box1 = box1.cpu()
    if box2.is_cuda == True:
        box2 = box2.cpu()

    if format == 'center':
        box1 = transform_coord(box1)
        box2 = transform_coord(box2)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    if type == 'full':
        return inter_area / (b1_area + b2_area - inter_area)
    elif type == 'part':
        return inter_area / b1_area, inter_area / b2_area


def draw_detection(img_path, detection, reso, dataset, type):
    """Draw detection result

    Args
    - img_path: (str) path to image
    - detection: (np.array) detection result
        1. (type == 'pred') with size [#bbox, [batch_idx, top-left x, top-left y, bottom-right x, bottom-right y, objectness, cls_conf, class idx]]
        2. (type == 'gt') with size [#box, [top-left x, top-left y, bottom-right x, bottom-right y]]
    - reso: (int) image resolution
    - dataset: (str) dataset name
    - type: (str) prediction or ground truth

    Returns
    - img: (Pillow.Image) detection result
    """
    class_names = config.datasets[dataset]['class_names']

    img = Image.open(img_path)
    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso
    h_ratio, w_ratio
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("../assets/Roboto-Bold.ttf", 15)

    for i in range(detection.shape[0]):
        if type == 'pred':
            bbox = detection[i, 1:5]
            category = int(detection[i, -1])
            label = class_names[category]
            conf = '%.2f' % detection[i, -2]
            caption = str(label) + ' ' + str(conf)
        elif type == 'gt':
            bbox = transform_coord(detection[i, 0:4], src='center', dst='corner')
            category = int(detection[i, -1])
            label = class_names[category]
            caption = str(label)
            w_ratio = w
            h_ratio = h
        else:
            raise Exception(emojify("detection type not recognized! :shit:"))

        if category not in config.colors.keys():
            config.colors[category] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        draw.rectangle(((x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio)), outline=config.colors[category])
        draw.rectangle((bbox[0] * w_ratio, bbox[1] * h_ratio - 15, bbox[2] * w_ratio, bbox[1] * h_ratio), fill=config.colors[category])
        draw.text((bbox[0] * w_ratio + 2, bbox[1] * h_ratio - 15), caption, fill='white', font=font)

    return img


def write_detection(img_path, detection, reso, dataset):
    """Write detection result to .txt file

    Args
    - img_path: (str) path to image
    - detections: (Tensor) with size [#bbox, [batch_idx, top-left x, top-left y, bottom-right x, bottom-right y, objectness, cls_conf, class idx]]
    - reso: (int) image resolution
    - dataset: (str) dataset name
    """
    label_name = img_path.split('/')[-1].replace('.jpg', '.txt')
    label_dir = config.datasets[dataset]['result_dir']
    os.makedirs(label_dir, exist_ok=True)
    label_path = opj(label_dir, label_name)

    labels = []
    for i in range(detection.shape[0]):
        bbox = detection[i, 1:5]
        xc, yc, h, w = transform_coord(bbox / reso, src='corner', dst='center')
        conf = detection[i, -2]
        cls = int(detection[i, -1])
        labels.append("%d %.5f %.5f %.5f %.5f %.5f\n" % (cls, conf, xc, yc, w, h))

    if os.path.exists(label_path):
        os.remove(label_path)
    with open(label_path, 'a') as f:
        for label in labels:
            f.write(label)


def get_current_time():
    """Get current datetime

    Returns
    - time: (str) time in format "dd-hh-mm"
    """
    time = str(datetime.datetime.now())
    time = time.split('-')[-1].split('.')[0]
    time = time.replace(' ', ':')
    day, hour, minute, _ = time.split(':')
    if day[-1] == '1':
        day += 'st'
    elif day[-1] == '2':
        day += 'nd'
    elif day[-1] == '3':
        day += 'rd'
    else:
        day += 'th'
    time = day + '.' + hour + '.' + minute
    return str(time)


def load_checkpoint(checkpoint_dir, epoch, iteration):
    """Load checkpoint from path

    Args
    - checkpoint_dir: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint
    - iteration: (int) iteration of checkpoint in one epoch

    Returns
    - start_epoch: (int)
    - start_iteration: (int)
    - state_dict: (dict) state of model
    """
    path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    if not os.path.isfile(path):
        raise Exception(emojify("Checkpoint in epoch %d doesn't exist :shit:" % epoch))

    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    start_iteration = checkpoint['iteration']

    assert epoch == start_epoch, emojify("`epoch` != checkpoint's `start_epoch` :shit:")
    assert iteration == start_iteration, emojify("`iteration` != checkpoint's `start_iteration` :shit:")
    return start_epoch, start_iteration, state_dict


def save_checkpoint(checkpoint_dir, epoch, iteration, save_dict):
    """Save checkpoint to path

    Args
    - path: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint file
    - iteration: (int) iteration of checkpoint in one epoch
    - save_dict: (dict) saving parameters dict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    assert epoch == save_dict['epoch'], emojify("`epoch` != save_dict's `start_epoch` :poop:")
    assert iteration == save_dict['iteration'], emojify("`iteration` != save_dict's `start_iteration` :poop:")
    if os.path.isfile(path):
        print(emojify("Overwrite checkpoint in epoch %d, iteration %d :exclamation:" % (epoch, iteration)))
    try:
        torch.save(save_dict, path)
    except Exception:
        raise Exception(emojify("Fail to save checkpoint :sob:"))

    print(emojify("Checkpoint %s saved :heavy_check_mark:" % (str(epoch) + '.' + str(iteration) + '.ckpt')))


def log(writer, name, info, step):
    """Wrapper for tensorboard writer

    Args
    - writer: (SummaryWriter)
    - name: (string) category name
    - info: (dict or float) value
    - step: (int) global steps
    """
    if isinstance(info, dict):
        for key, value in info.items():
            tag = name + '/' + key
            writer.add_scalar(tag, value, step)
    elif isinstance(info, float):
        writer.add_scalar(name, info, step)
    else:
        raise TypeError("Logging info type", type(info), "is not supported for", name)


def voc_mAP(pred_dir, gt_dir, dataset, debug=False):
    """Compute mAP between predictions and ground truths
    
    Args
    - pred_dir: (str) full path to predictions label file, each file with format
        [class_idx, confidence, xc, yc, w, h]
    - gt_dir: (str) full path to predictions label file, each file with format
        [class_idx, xc, yc, w, h]
    - dataset: (str) dataset name
    - debug: (bool)
    
    Returns
    - mAP: (float)

    Variables
    - preds_cls: (list of np.array) predictions for all images of each class
        each with size [TP/FP, confidence, xc, yc, w, h]
    - pred_cls: (np.array) predictions for one image of specified class
        with size [class_idx, confidence, xc, yc, w, h]
    - gt_cls: (np.array) ground truths for one image of specified class
        with size [class_idx, xc, yc, w, h]
    - fn_cls: (list of int) FN for each class
    """
    
    num_classes = config.datasets[dataset]['num_classes']
    class_names = config.datasets[dataset]['class_names']
    num_imgs = len(os.listdir(pred_dir))
    
    preds_cls = [np.array([])] * num_classes
    num_pred_bboxes, num_gt_bboxes = [0] * num_classes, [0] * num_classes
    APs = []

    for name in tqdm(os.listdir(pred_dir), ncols=80, ascii=True):
        pred_path = opj(pred_dir, name)
        gt_path = opj(gt_dir, name)
        pred = np.loadtxt(pred_path)
        gt = np.loadtxt(gt_path)
        if len(pred.shape) == 1 and pred.shape[0] != 0:
            pred = pred.reshape(1, -1)
        if len(gt.shape) == 1 and gt.shape[0] != 0:
            gt = gt.reshape(1, -1)
        for cls in range(num_classes):
            pred_cls = pred[pred[:, 0] == cls] if pred.shape[0] != 0 else np.array([])
            gt_cls = gt[gt[:, 0] == cls] if gt.shape[0] != 0 else np.array([])
            num_pred_bboxes[cls] += pred_cls.shape[0]
            num_gt_bboxes[cls] += gt_cls.shape[0]
            if pred_cls.shape[0] != 0:
                pred_cls = pred_cls[(-pred_cls[:,1]).argsort()]
                for idx in range(pred_cls.shape[0]):
                    if gt_cls.shape[0] == 0:
                        pred_cls[idx, 0] = -2  # -2 for FP
                        continue
                    pred_det = pred_cls[idx]
                    ious = IoU(np.expand_dims(pred_det, 0)[:, 2:], gt_cls[:, 1:], format='center').cpu().numpy()
                    best_iou = np.max(ious)
                    best_idx = np.argmax(ious)
                    if best_iou > 0.5:
                        pred_cls[idx, 0] = -1  # -1 for TP
                        gt_cls = np.concatenate((gt_cls[0:best_idx], gt_cls[best_idx+1:]), 0)
                preds_cls[cls] = np.concatenate((preds_cls[cls], pred_cls), 0) if preds_cls[cls].shape[0] != 0 else pred_cls

    for class_idx, pred_cls in enumerate(preds_cls):
        pred_cls = pred_cls[(-pred_cls[:,1]).argsort()]
        tp = np.cumsum(pred_cls[:,0] == -1)
        fp = np.cumsum(pred_cls[:,0] == -2)
        recall = tp / num_gt_bboxes[class_idx]
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        mrecall = np.concatenate(([0.], recall, [1.]))
        mprecision = np.concatenate(([0.], precision, [0.]))
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])
        i = np.where(mrecall[1:] != mrecall[:-1])[0]
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])
        APs.append(ap)
        print(ap, class_idx)
    
    mAP = np.mean(APs)
    return mAP