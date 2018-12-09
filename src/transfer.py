import os
import config
import argparse
from model import YOLOv3
from utils import save_checkpoint
opj = os.path.join


def parse_arg():
    parser = argparse.ArgumentParser("Transfer .weights to PyTorch checkpoint")
    parser.add_argument('--dataset', type=str, help="Dataset name",
                        choices=['voc', 'coco', 'linemod'])
    parser.add_argument('--weights', default='darknet53.conv.74.weights',
                        type=str, help=".weights file name (stored in checkpoint/darknet)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    weight_path = opj(config.CKPT_ROOT, 'darknet', args.weights)
    print("[LOG] Loading weights from", weight_path)
    model = YOLOv3(config.network[args.dataset]['cfg'], 416).cuda()

    if len(args.weights.split('.')) > 2:  # with cutoff
        cutoff = int(args.weights.split('.')[-2])
        model.load_weights(weight_path, cutoff=cutoff)
        save_checkpoint(opj(config.CKPT_ROOT, args.dataset), 0, 0, {
            'epoch': 0,
            'iteration': 0,
            'state_dict': model.state_dict(),
        })
    else:  # pretrained
        model.load_weights(weight_path)
        save_checkpoint(opj(config.CKPT_ROOT, args.dataset), -1, -1, {
            'epoch': -1,
            'iteration': -1,
            'state_dict': model.state_dict(),
        })
