from utils import get_current_time, draw_detection, save_checkpoint, load_checkpoint, log
from dataset import prepare_train_dataset, prepare_val_dataset
from model import YOLOv3
import config
import os
import time
import torch
import argparse
import warnings
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torchvision import transforms, utils
import torch.optim.lr_scheduler as lr_scheduler
opj = os.path.join
warnings.filterwarnings("ignore")


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--reso', type=int, default=416,
                        help="Input image resolution")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--bs', type=int, default=64, help="Batch size")
    parser.add_argument('--dataset', type=str, help="Dataset name",
                        choices=['voc', 'coco', 'linemod'])
    parser.add_argument('--ckpt', type=str, default='-1.-1',
                        help="Checkpoint name in format: `epoch.iteration`")
    parser.add_argument('--gpu', type=str, default='0', help="GPU id")
    parser.add_argument('--seq', type=str, help="LINEMOD sequence number")
    return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
log_dir = opj(config.LOG_ROOT, get_current_time())
writer = SummaryWriter(log_dir=log_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(epoch, trainloader, yolo, optimizer):
    """Training wrapper

    Args
    - epoch: (int) training epoch
    - trainloader: (Dataloader) train data loader
    - yolo: (nn.Module) YOLOv3 model
    - optimizer: (optim) optimizer
    """
    yolo.train()
    # tbar = tqdm(trainloader, ncols=80, ascii=True)
    # tbar.set_description('training')
    for batch_idx, (paths, inputs, targets) in enumerate(trainloader):
        global_step = batch_idx + epoch * len(trainloader)

        optimizer.zero_grad()
        inputs = inputs.cuda()
        loss = yolo(inputs, targets)
        log(writer, 'training loss', loss, global_step)
        loss['total'].backward()
        optimizer.step()
        # tbar.set_postfix(loss=loss['total'])


if __name__ == '__main__':
    # Loading network
    # TODO: resume tensorboard
    print("[LOG] Loading network and data")
    yolo = YOLOv3(cfg, args.reso)
    start_epoch, start_iteration = args.ckpt.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        opj(config.CKPT_ROOT, args.dataset),
        int(start_epoch),
        int(start_iteration)
    )
    yolo.load_state_dict(state_dict)
    yolo = yolo.cuda()

    # Preparing data
    train_img_datasets, train_dataloader = prepare_train_dataset(
        args.dataset, args.reso, args.bs, seq=args.seq)
    val_img_datasets, val_dataloder = prepare_val_dataset(
        args.dataset, args.reso, args.bs, seq=args.seq)
    print("[LOG] Model starts training from epoch %d iteration %d" %
          (start_epoch, start_iteration))
    print("[LOG] Number of training images:", len(train_img_datasets))
    print("[LOG] Number of validation images:", len(val_img_datasets))

    # Training
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, yolo.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(start_epoch, start_epoch+100):
        print("\n[LOG] Epoch", epoch)
        scheduler.step()
        train(epoch, train_dataloader, yolo, optimizer)
        # save_checkpoint(opj(config.CKPT_ROOT, args.dataset), epoch + 1, 0, {
        #     'epoch': epoch + 1,
        #     'iteration': 0,
        #     'state_dict': yolo.module.state_dict()
        # })
