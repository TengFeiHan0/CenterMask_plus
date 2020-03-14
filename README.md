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
