# dataset.py - ExDark Dataset with robust bbox handling
import os, cv2, math
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import IMG_SIZE, CLASS_NAMES

from .annotations import parse_exdark_annotation, build_annotation_mapping

CLASS2ID = {c:i for i,c in enumerate(CLASS_NAMES)}

def find_image_path(imgname):
    # tries common extensions & folders relative to project
    possible = [
        os.path.join(imgname),
    ]
    # user supplies absolute paths in config; dataset init can convert
    return None  # caller should pass full path via ann_map to dataset

class ExDarkDataset(Dataset):
    def __init__(self, samples_list, ann_map, img_root, img_size=IMG_SIZE, train=True):
        """
        samples_list: list of image names (like '2015_00001.png') - keys to ann_map
        ann_map: mapping key -> annotation file path
        img_root: root folder containing class folders (ExDark/ExDark/<class>/<img>)
        """
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.ann_map = ann_map
        self.samples = []
        for name in samples_list:
            ann = ann_map.get(name)
            # find image file inside class folders (common ExDark layout)
            img_path = None
            for cls in os.listdir(img_root):
                p = os.path.join(img_root, cls, name)
                if os.path.exists(p):
                    img_path = p; break
            if img_path is None:
                # try direct path (maybe ann file gave full path)
                base = name
                for ext in (".jpg",".png",".jpeg",".JPG",".JPEG",".PNG"):
                    alt = os.path.join(img_root, base)
                    if os.path.exists(alt):
                        img_path = alt; break
            if img_path is None or ann is None:
                continue
            self.samples.append((img_path, ann, name))
        print(f"[DATASET] {len(self.samples)} samples loaded.")

        if train:
            self.augs = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.0))
        else:
            self.augs = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size),
                A.Normalize(),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.0))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path, imgname = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not readable: {img_path}")
        h,w = img.shape[:2]

        annots = parse_exdark_annotation(ann_path)
        boxes = []
        labels = []
        for cls, l, t, ww, hh in annots:
            # handle whether annotation is normalized (<=1) or absolute pixel coords
            if 0.0 <= l <= 1.0 and 0.0 <= t <= 1.0 and 0.0 <= ww <= 1.0 and 0.0 <= hh <= 1.0:
                x1 = l * w
                y1 = t * h
                x2 = (l+ww) * w
                y2 = (t+hh) * h
            else:
                x1 = l; y1 = t; x2 = l + ww; y2 = t + hh
            # clamp to image bounds
            x1 = max(0.0, min(x1, w-1))
            y1 = max(0.0, min(y1, h-1))
            x2 = max(0.0, min(x2, w-1))
            y2 = max(0.0, min(y2, h-1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1,y1,x2,y2])
            if cls in CLASS2ID:
                labels.append(CLASS2ID[cls])
            else:
                try:
                    labels.append(int(cls)-1)
                except:
                    labels.append(0)
        if len(boxes) == 0:
            # dummy tiny box to avoid empty bboxes (will be ignored later)
            boxes = [[0.0,0.0,1.0,1.0]]
            labels = [0]

        # Albumentations expects bboxes in pascal_voc (absolute pixel coords)
        augmented = self.augs(image=img, bboxes=boxes, labels=labels)
        img_t = augmented['image']            # tensor C,H,W
        bboxes = augmented['bboxes']          # list of tuples
        labels = augmented['labels']

        boxes_t = torch.tensor(bboxes, dtype=torch.float32) if len(bboxes)>0 else torch.zeros((0,4),dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long) if len(labels)>0 else torch.zeros((0,),dtype=torch.long)

        return img_t, boxes_t, labels_t, imgname
