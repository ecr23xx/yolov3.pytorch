import os
import argparse
import warnings
from tqdm import tqdm
from PIL import Image
from termcolor import colored
from pyemojify import emojify
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_demo_dataset
from utils import draw_detection, load_checkpoint


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--reso', default=480, type=int, help="Input image resolution of the network")
    parser.add_argument('--dataset', default='coco', choices=['coco'], type=str, help="Trained dataset name")
    parser.add_argument('--checkpoint', default='-1.-1', type=str, help="Checkpoint name in format: `epoch.iteration`")
    return parser.parse_args()


args = parse_arg()
cfg = config.network['coco']['cfg']              # model cfg file path


if __name__ == '__main__':
    print(colored("\n==>", 'blue'), emojify("Parsing arguments ... :hammer:\n"))
    assert args.reso % 32 == 0, "Resolution must be interger times of 32"
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))

    print(colored("\n==>", 'blue'), emojify("Prepare data ... :coffee:\n"))
    img_datasets, dataloader = prepare_demo_dataset(config.demo['images_dir'], args.reso)
    print("Number of demo images:", len(img_datasets))

    print(colored("\n==>", 'blue'), emojify("Loading network ... :hourglass:\n"))
    yolo = YOLOv3(cfg, args.reso).cuda()
    start_epoch, start_iteration = args.checkpoint.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        opj(config.CKPT_ROOT, args.dataset),
        int(start_epoch),
        int(start_iteration)
    )
    yolo.load_state_dict(state_dict)
    print("Model starts training from epoch %d iteration %d" % (start_epoch, start_iteration))

    print(colored("\n==>", 'blue'), emojify("Evaluation ...\n"))
    yolo.eval()
    for batch_idx, (img_path, inputs) in enumerate(tqdm(dataloader, ncols=80, ascii=True)):
        inputs = inputs.cuda()
        detections = yolo(inputs)

        # take idx 0
        detections = detections[detections[:, 0] == 0]
        path = img_path[0]

        img_name = path.split('/')[-1]
        img = draw_detection(path, detections.data, args.reso, type='pred')
        img.save(opj(config.demo['result_dir'], img_name))
    print(emojify("Done! :+1:\n"))
