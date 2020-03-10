#CenterMask_PLUS


## Highlights
- ***First* anchor-free one-stage instance segmentation.** To the best of our knowledge, **CenterMask** is the first instance segmentation on top of anchor-free object detection (15/11/2019).
- **Toward Real-Time: CenterMask-Lite.**  This works provide not only large-scale CenterMask but also lightweight CenterMask-Lite that can run at real-time speed (> 30 fps).
- **State-of-the-art performance.**  CenterMask outperforms Mask R-CNN, TensorMask, and ShapeMask at much faster speed and CenterMask-Lite models also surpass YOLACT or YOLACT++ by large margins.
- **Well balanced (speed/accuracy) backbone network, VoVNetV2.**  VoVNetV2 shows better performance and faster speed than ResNe(X)t or HRNet.

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Set up datasets
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
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/kitti kitti
    ```

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

For examaple (CenterMask-Lite-R-50),
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

## TODO
 - [x] add ATSS provided by a CVPR2020
 - [x] add PointRend
 - [x] add depth estimation module
