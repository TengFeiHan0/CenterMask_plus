# CenterMask_plus(under construction)

## Note:  The original implementation is at [here](https://github.com/youngwanLEE/CenterMask). More details could be found in their project. this project is still contructing, models are still running on my server, once got some results, I will update this project.

## Highlights
- **LOSS** . various loss functions(DistanceIOU, CompleteIOU) will be supplied
- **DATASET** . interfaces to more driving datasets(including KITTI, Cityscapes) 
- **ATSS**. a new proposed idea(ATSS) based on FCOS will be included
- **Tensorboard**. to display losses 

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Set up datasets

1. Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/).
2. Download the annotation files from the official [website](https://www.cityscapes-dataset.com/downloads/).
3. Organize the dataset as the following structure:
    ```
    ├── /path/to/cityscapes
    │   ├── annotations
    │   ├── leftImg8bit
    │   ├── gtFine
    ```
4. Create a soft link(optional):
    ```
    ROOT=/path/to/cityscapes
    cd $ROOT/datasets
    ln -s /path/to/cityscapes cityscapes
    ```

#### Kitti

1. Download the Kitti dataset from the official [website](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).
2. Download the annotation file `instances_train.json` and `instances_val.json` from [Kins](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).
3. Organize the dataset as the following structure:
	```
    ├── /path/to/kitti
    │   ├── testing
    │   │   ├── image_2
    │   │   ├── instance_val.json
    │   ├── training
    │   │   ├── image_2
    │   │   ├── instance_train.json
    ```
4. Create a soft link:
    ```
    ROOT=/path/to/kitti
    cd $ROOT/datasets
    ln -s /path/to/kitti kitti
    ```
#### Tips
when trying to use these new datasets, three things should be done before launching your project.
1. add your dataloader and eval scripts
2. modifiy  **paths_catalog.py**  to add directories of your new datasets
3. change the **INPUT** and **NUM_CLASSES** in **defaults.py**

## Training
Follow [the instructions](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training) of  [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) guides.

If you want multi-gpu (e.g.,8) training,

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/centermask/centermask_R_50_FPN_1x.yaml" 
```

## Evaluation

Follow [the instruction](https://github.com/facebookresearch/maskrcnn-benchmark#evaluation) of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

First of all, you have to download the weight file you want to inference.

##### multi-gpu evaluation & test batch size 16,
```bash
wget https://www.dropbox.com/s/2enqxenccz4xy6l/centermask-lite-R-50-ms-bs32-1x.pth
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/centermask/centermask_R_50_FPN_lite_res600_ms_bs32_1x.yaml"   TEST.IMS_PER_BATCH 16 MODEL.WEIGHT centermask-lite-R-50-ms-bs32-1x.pth
```
FCOS on COCO2017val

### coco val2017 results
|Detector | Backbone |  epoch |   Mask AP (AP/APs/APm/APl) | Box AP (AP/APs/APm/APl) |  Time (ms) | Weight |
|----------|----------|:--------------:|:-------------------:|:------------------------:| :---:|:---:|
| Mask R-CNN    | R-50-FPN   |    24 |   35.9/17.1/38.9/52.0 |     39.7/24.0/43.0/50.8              | 77      | [link](https://www.dropbox.com/s/r3ocl8ls45wsbgo/MRCN-R-50-FPN-ms-2x.pth?dl=1)|
| **FCOS**    | R-50-FPN   |    24 |   |     33.4/18.4/36.4/41.8             |      |
| **CenterMask**    | **V2-39-FPN**   |    24 | 37.7/17.9/40.8/54.3   |         42.6/25.3/46.3/55.2          | **70**      | [link](https://www.dropbox.com/s/ugcpzcx5b4btvjc/centermask-V2-39-FPN-ms-2x.pth?dl=1)|
| Mask R-CNN    | R-50-FPN   |    36 |   36.5/17.9/39.2/52.5|     40.5/24.7/43.7/52.2              | 77      | [link](https://www.dropbox.com/s/09ny9ofj5t1r883/MRCN-R-50-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    | R-50-FPN   |    36 | 37.0/17.6/39.7/53.8  |       41.7/24.8/45.1/54.5            | 72      | [link](https://www.dropbox.com/s/438pbeuqlj1spf0/centermask-R-50-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    | **V2-39-FPN**   |    36 |  38.5/19.0/41.5/54.7 |      43.5/27.1/46.9/55.9           | **70**      | [link](https://www.dropbox.com/s/5mmq2ok0yopupnz/centermask-V2-39-FPN-ms-3x.pth?dl=1)|
||
| Mask R-CNN    | R-101-FPN   |    24 |  37.8/18.5/40.7/54.9  |         42.2/25.8/45.8/54.0          | 94      | [link](https://www.dropbox.com/s/ptjc4qorps5gbwe/MRCN-R-101-FPN-ms-2x.pth?dl=1)|
| **CenterMask**    | R-101-FPN   |    24 | 38.0/18.2/41.3/55.2  |       43.1/25.7/47.0/55.6            | 91      | [link](https://dl.dropbox.com/s/9w17k9iiihob8vx/centermask-R-101-ms-2x.pth?dl=1)|
| **CenterMask**    | **V2-57-FPN**   |    24 | 38.5/18.6/41.9/56.2  |      43.8/26.7/47.4/57.1             | **76**      | [link](https://www.dropbox.com/s/949k1ednumtd2rk/centermask-V2-57-FPN-ms-2x.pth?dl=1)|
| Mask R-CNN    | R-101-FPN   |    36 | 38.0/18.4/40.8/55.2  |      42.4/25.4/45.5/55.2             |   94    | [link](https://www.dropbox.com/s/hev2k4vfh362d3s/MRCN-R-101-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    | R-101-FPN   |    36 | 38.6/19.2/42.0/56.1 |   43.7/27.2/47.6/56.7    | 91      | [link](https://www.dropbox.com/s/1uxpfh8z0sp8tr2/centermask-R-101-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    | **V2-57-FPN**   |    36 |   39.4/19.6/42.9/55.9  |      44.6/27.7/48.3/57.3 | **76**      | [link](https://www.dropbox.com/s/5m5tc4h30tqp2it/centermask-V2-57-FPN-ms-3x.pth?dl=1)|
||
| Mask R-CNN    | X-101-32x8d-FPN   |    24 |   38.9/19.6/41.6/55.7    |   43.7/27.6/46.9/55.9       |   165    | [link](https://www.dropbox.com/s/o6uu0nft0a8iu5s/MRCN-X-101-FPN-ms-2x.pth?dl=1)|
| **CenterMask**    | X-101-32x8d-FPN  |    24 | 39.1/19.6/42.5/56.1  |        44.3/26.9/48.5/57.0      | 157      | [link](https://www.dropbox.com/s/ovhzjz43nph14mo/centermask-X-101-FPN-ms-2x.pth?dl=1)|
| **CenterMask**    | **V2-99-FPN**   |    24 | 39.6/19.6/43.1/56.9  |      44.8/27.6/49.0/57.7        |  **106**     | [link](https://www.dropbox.com/s/lemwoq6qwoqnbzm/centermask-V2-99-FPN-ms-2x.pth?dl=1)|
| Mask R-CNN    | X-101-32x8d-FPN   |    36 | 38.6/19.7/41.1/55.2  |    43.6/27.3/46.7/55.6          |   165    | [link](https://www.dropbox.com/s/yl3zmeaxghvni43/MRCN-X-101-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    |X-101-32x8d-FPN   |    36 | 39.1/18.5/42.3/56.4  |     44.4/26.7/47.7/57.1         | 157      |[link](https://www.dropbox.com/s/yrczyb1u49hv05a/centermask-X-101-FPN-ms-3x.pth?dl=1)|
| **CenterMask**    | **V2-99-FPN**   |    36 | 40.2/20.6/43.5/57.3  |      45.6/29.2/49.3/58.8       | **106**      | [link](https://www.dropbox.com/s/99i7ydsz2ngrvu1/centermask-V2-99-FPN-ms-3x.pth?dl=1)|

*Note that the all models are trained using **train-time augmentation (multi-scale)**.*\
*The inference time of all models is measured on **Titan Xp** GPU.*\
*24/36 epoch are same as x2/x3 training schedule in [detectron](https://github.com/facebookresearch/Detectron), respectively.*
##### For single-gpu evaluation & test batch size 1,
```bash
wget https://www.dropbox.com/s/2enqxenccz4xy6l/centermask-lite-R-50-ms-bs32-1x.pth
CUDA_VISIBLE_DEVICES=0
python tools/test_net.py --config-file "configs/centermask/centermask_R_50_FPN_lite_res600_ms_bs32_1x.yaml" TEST.IMS_PER_BATCH 1 MODEL.WEIGHT centermask-lite-R-50-ms-bs32-1x.pth
```
#### Tensorbaord
in order to visualize our losses, please install tensorboardX by typing ```pip install tensorboardX```
## TODO
 - [] add [ATSS](https://github.com/sfzhang15/ATSS) provided by a CVPR2020
 - [] add [PointRend](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend)
 - [x] add more different loss functions
