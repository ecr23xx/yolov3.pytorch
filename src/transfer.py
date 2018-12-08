import os
import config
import argparse
from model import YOLOv3
from utils import save_checkpoint
opj = os.path.join


def parse_arg():
    parser = argparse.ArgumentParser("Transfer .weights file to PyTorch readable checkpoint")
    parser.add_argument('--dataset', default='voc', choices=['voc', 'coco'], type=str, help="Dataset name")
    parser.add_argument('--weights', default='darknet53.conv.74.weights', type=str, help=".weights file name (stored in checkpoint/darknet)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    print(args)
    model = YOLOv3(config.network[args.dataset]['cfg'], 416).cuda()

    # weights with cutoff
    if len(args.weights.split('.')) > 2:
        cutoff = int(args.weights.split('.')[-2])

    model.load_weights(opj(config.CKPT_ROOT, 'darknet', args.weights), cutoff=cutoff)
    save_checkpoint(opj(config.CKPT_ROOT, args.dataset), 0, 0, {
        'epoch': 0,
        'iteration': 0,
        'state_dict': model.state_dict(),
    })
