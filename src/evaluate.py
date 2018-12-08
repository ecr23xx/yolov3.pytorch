import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyemojify import emojify
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_val_dataset, prepare_train_dataset
from utils import draw_detection, load_checkpoint, write_detection


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
    parser.add_argument('--batch', default=32, type=int, help="Batch size")
    parser.add_argument('--dataset', default='voc', choices=['tejani', 'coco'], type=str, help="Dataset name")
    parser.add_argument('--checkpoint', default='-1.-1', type=str, help="Checkpoint name in format: `epoch.iteration`")
    parser.add_argument('--gpu', default='0', help="GPU ids")
    return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def val(valloader, yolo):
    """Validation wrapper

    Args
      valloader: (Dataloader) validation data loader 
      yolo: (nn.Module) YOLOv3 model
    """
    preds = []
    gts = []
    tbar = tqdm(valloader, ncols=80, ascii=True)
    tbar.set_description("evaluation")
    for batch_idx, (names, inputs, targets) in enumerate(tbar):
        inputs = inputs.cuda()
        detections = yolo(inputs)
        
        for idx, name in enumerate(names):
            img_path = opj(config.datasets[args.dataset]['val_imgs'], name)
            img_name = img_path.split('/')[-1]

            try:
                detection = detections[detections[:, 0] == 0]
            except Exception:
                detection = torch.Tensor([])

            write_detection(img_path, detection, yolo.reso, args.dataset)

            if idx == 0:                
                img = draw_detection(img_path, detection, yolo.reso, dataset=args.dataset, type='pred')
                img.save(opj(config.evaluate['result_dir'], img_name))


if __name__ == '__main__':
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))

    # 2. Prepare data
    print(emojify("\n==> Preparing data and network ... :coffee:"))
    img_datasets, dataloader = prepare_val_dataset(args.dataset, args.reso, args.batch)

    # 3. Loading network
    yolo = YOLOv3(cfg, args.reso).cuda()
    start_epoch, start_iteration = args.checkpoint.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        opj(config.CKPT_ROOT, args.dataset),
        int(start_epoch),
        int(start_iteration)
    )
    yolo.load_state_dict(state_dict)
    print("Number of evaluate images:", len(img_datasets))

    os.system('rm ' + opj(config.evaluate['result_dir'], '*.jpg'))
    yolo.eval()
    with torch.no_grad():
        val(dataloader, yolo)
    print("Done!")
