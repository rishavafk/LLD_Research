# transforms.py - safe augmentations using albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..config import IMG_SIZE

def get_train_augs(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.1),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0))

def get_val_augs(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0))
