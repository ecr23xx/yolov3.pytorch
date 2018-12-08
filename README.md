# yolov3.pytorch

This repository is used for object detection. The algorithm is based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch v0.4. **Thanks to  [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) and [ultralytics/yolov3](https://github.com/ultralytics/yolov3)**, based on their work, I re-implemented YOLO v3 in PyTorch for better readability and re-useablity.

## News

Full version of update logs could be seen in issue [#2](https://github.com/ECer23/yolov3.pytorch/issues/2)
 
* (2018/10/10) Support training on VOC dataset.

## Environments

* Python 3.6
* PyTorch 0.4.1
* CUDA (CPU is not supported)

## Train

### How to train on COCO

1. Download [COCO detection](http://cocodataset.org/#download) dataset and annotions and provide full path to your downloaded dataset in `config.py` like below
    ```python
    'coco': {
        'train_imgs': '/home/data/coco/2017/train2017',
        'train_anno': '/home/data/coco/2017/annotations/instances_train2017.json'
    }
    ````
2. Download official pre-trained Darknet53 weights on ImageNet [here](https://pjreddie.com/media/files/darknet53.conv.74), and move it to `checkpoint/darknet/darknet53.conv.74.weights`
3. Transform the weights to PyTorch readable file `0.ckpt` by running
    ```bash
    $ python transfer.py --dataset=coco --weights=darknet53.conv.74.weights
    ```
4. Run
    ```bash
    $ python train.py
    ```

### How to train on custom dataset

1. Implement your own dataset loading function in `dataset.py`. You should keep the interfaces similar to that in `dataset.py`.
2. Add your dataset in `prepare_dataset` function in `dataset.py`
3. Details can be viewed in `dataset.py`. This part requires some coding, and need to be imporved later.

### Training visualization

Logging directory will be displayed when you run training scripts. You can visualize the training process by running
 
```shell
$ tensorboard --logdir path-to-your-logs
```

![tensorboard](https://camo.githubusercontent.com/f477420c421d99d26812bdedc45a58623e8fc09f/687474703a2f2f7778312e73696e61696d672e636e2f6c617267652f303036306c6d37546c79316677687a7038646b706e6a333168633075306a77792e6a7067)

## Evaluation

### How to evaluate on COCO

1. Download [COCO detection](http://cocodataset.org/#download) dataset and annotions and provide full path to your downloaded dataset in `config.py` like below
    ```python
    'coco': {
        'val_imgs': '/home/data/coco/2017/val2017',
        'val_anno': '/home/data/coco/2017/annotations/instances_val2017.json',
    }
    ````
2. Download official pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights) and move it to `checkpoint/darknet/yolov3-coco.weights`
3. Transform the weights to PyTorch readable file `checkpoint/coco/-1.-1.ckpt` by running
    ```bash
    $ python transfer.py --dataset=coco --weights=yolov3-coco.weights
    ```
4. Evaluate on validation sets you specify in `config.py` and compute the mAP by running. Some validation detection examples will be save to `assets/results`
    ```bash
    $ python evaluate.py
    ```

### How to detect COCO objects

1. Download official pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights) and move it to `checkpoint/darknet/yolov3-coco.weights`
2. Transform the weights to PyTorch readable file `checkpoint/coco/-1.-1.ckpt` by running
    ```bash
    $ python transfer.py --dataset=coco --weights=yolov3-coco.weights
    ```
3. Specify the images folder in `config.py`
    ```python
    demo = {
      'images_dir': opj(ROOT, 'assets/imgs'),
      'result_dir': opj(ROOT, 'assets/dets')
    }
    ```
4. Detect your own images by running
    ```bash
    $ python demo.py
    ```

### Evaluation results

mAP computation seems not very accruate

| Test datasets | Training datasets | Resolution | Notes | mAP | FPS |
|---|---|---|---|---|---|
| COCO 2017 |  | 416 | official pretrained YOLO v3 weights | 63.4 | |
| COCO 2017 |  | 608 | paper results | 57.9 | |

### Evaluation demo

![](https://github.com/ECer23/yolov3.pytorch/raw/master/assets/dets/person.jpg)

## TODO

- [ ] Evaluation
  - [x] ~~Draw right bounding box~~
  - [ ] mAP re-implementated
    - [x] ~~VOC mAP implemented~~
    - [ ] COCO mAP implemented
- [ ] Training
  - [x] ~~Loss function implementation~~
  - [x] ~~Visualize training process~~
  - [x] ~~Use pre trained Darknet model to train on custom datasets~~
  - [x] ~~Validation~~
  - [ ] Train COCO from scratch
  - [ ] Train custom datasets from scratch
  - [x] ~~Learning rate scheduler~~
  - [x] ~~Data augumentation~~
- [ ] General
  - [ ] Generalize annotation format to VOC for every dataset
  - [x] ~~Multi-GPU support~~
  - [x] ~~Memory use imporvements~~


## Reference

* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3
* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) PyTorch implmentation of YOLO v3, with only evaluation part
* [ultralytics/yolov3](https://github.com/ultralytics/yolov3) PyTorch implmentation of YOLO v3, with both training and evaluation parts
* [utkuozbulak/pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples) Example of PyTorch custom dataset
