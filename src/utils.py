import config
import os
import torch
import random
import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
opj = os.path.join


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
        raise Exception("Format not supported!")

    if flag == True:
        bbox_transformed = bbox_transformed.squeeze(0)

    return bbox_transformed


def xywh2xyxy(bbox):
    bbox_ = bbox.clone()
    if len(bbox_.size()) == 1:
        bbox_ = bbox_.unsqueeze(0)
    xc, yc = bbox_[..., 0], bbox_[..., 1]
    half_w, half_h = bbox_[..., 2] / 2, bbox_[..., 3] / 2
    bbox_[..., 0] = xc - half_w
    bbox_[..., 1] = yc - half_h
    bbox_[..., 2] = xc + 2 * half_w
    bbox_[..., 3] = yc + 2 * half_h
    return bbox_


def IoU(box1, box2):
    """Compute IoU between box1 and box2

    Args
    - box: (torch.cuda.Tensor) bboxes with size [# bboxes, 4]
    """
    if box1.is_cuda == True:
        box1 = box1.cpu()
    if box2.is_cuda == True:
        box2 = box2.cpu()

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * \
        torch.max(inter_rect_y2 - inter_rect_y1 + 1,
                  torch.zeros(inter_rect_x2.shape))
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def draw_detection(img_path, detection, reso, names, save_path):
    """Draw detection result

    Args
    - img_path: (str) Path to image
    - detection: (np.array) Detection result with size [#bbox, 8]
        8 = [batch_idx, x1, y1, x2, y2, objectness, cls_conf, class idx]
    - reso: (int) Image resolution
    - names: (list) Class names
    - save_path: (str) Path to save detection result
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso
    h_ratio, w_ratio
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("../assets/Roboto-Bold.ttf", 15)

    try:
        for i in range(detection.shape[0]):
            bbox = detection[i, 1:5]
            category = int(detection[i, -1])
            label = names[category]
            conf = '%.2f' % detection[i, -2]
            caption = str(label) + ' ' + str(conf)

            if category not in config.colors.keys():
                config.colors[category] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            draw.rectangle(((x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio)),
                           outline=config.colors[category])
            draw.rectangle((bbox[0] * w_ratio, bbox[1] * h_ratio - 15,
                            bbox[2] * w_ratio, bbox[1] * h_ratio),
                           fill=config.colors[category])
            draw.text((bbox[0] * w_ratio + 2, bbox[1] * h_ratio - 15),
                      caption, fill='white', font=font)
    except Exception:
        from IPython import embed
        embed()

    img.save(save_path)
    img.close()


def get_current_time():
    """Get current datetime

    Returns
    - time: (str) time in format "month-dd"
    """
    time = str(datetime.datetime.now())
    dhms = time.split('-')[-1].split('.')[0]
    day, hour, minute, _ = dhms.replace(' ', ':').split(':')
    month = calendar.month_name[int(time.split('-')[1])][:3]
    time = month + '.' + day
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
        raise Exception("Checkpoint in epoch %d doesn't exist" % epoch)

    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    start_iteration = checkpoint['iteration']

    assert epoch == start_epoch, "epoch != checkpoint's start_epoch"
    assert iteration == start_iteration, "iteration != checkpoint's start_iteration"
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
    assert epoch == save_dict['epoch'], "[ERROR] epoch != save_dict's start_epoch"
    assert iteration == save_dict['iteration'], "[ERROR] iteration != save_dict's start_iteration"
    if os.path.isfile(path):
        print("[WARNING] Overwrite checkpoint in epoch %d, iteration %d" %
              (epoch, iteration))
    try:
        torch.save(save_dict, path)
    except Exception:
        raise Exception("[ERROR] Fail to save checkpoint")

    print("[LOG] Checkpoint %d.%d.ckpt saved" % (epoch, iteration))


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
        raise TypeError("Logging info type", type(
            info), "is not supported for", name)
