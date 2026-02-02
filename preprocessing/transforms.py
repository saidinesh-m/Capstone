import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_train_transforms(image_size=224):
    """
    Returns training transforms:
    - Resize
    - CLAHE (Contrast Enhancement)
    - MedianBlur (De-noising)
    - RandomRotate
    - HorizontalFlip
    - RandomBrightnessContrast
    - Normalize
    - ToTensor
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5), # Enhance contrast
        A.MedianBlur(blur_limit=3, p=0.1), # Denoising
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225],  # ImageNet std
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

def get_valid_transforms(image_size=224):
    """
    Returns validation/inference transforms (Deterministic):
    - Resize
    - Normalize
    - ToTensor
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # Apply consistent CLAHE if used in train
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])
