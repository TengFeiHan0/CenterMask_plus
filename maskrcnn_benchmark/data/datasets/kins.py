import torch
import torch.utils.data as data
import numpy as np
import math
from pycocotools.coco import COCO
import os
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

class KITTIDataset(data.Dataset):
    def __init__(self, ann_file, root, transforms=None):
        super(KITTIDataset, self).__init__()
        self.ann_file = ann_file
        self.root = root
        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.anns = np.array(self.coco.getImgIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self._transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        img = Image.open(path).convert('RGB')
        return img, anno 
    
    def __getitem__(self, idx):
        img, anno = self.process_info(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    def __len__(self):
        return len(self.ids)
            
