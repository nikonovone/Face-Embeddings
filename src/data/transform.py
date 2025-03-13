from typing import List, Optional

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from ellzaf_ml.augments import PatchSwap

from src.utils.constants import IMAGE_MEAN, IMAGE_STD


class FaceSwapAugmentation:
    def __init__(self, features_to_swap: Optional[List[str]] = None):
        self.swapper = PatchSwap()
        self.features_to_swap = features_to_swap or [
            "right_eye",
            "left_eye",
            "nose",
            "lips",
        ]

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply face swap augmentation to a list of images"""
        if len(images) < 2:
            return images

        augmented = []
        # Process pairs of images
        for i in range(0, len(images) - 1, 2):
            img1, img2 = images[i], images[i + 1]
            aug1, aug2 = self.swapper.swap_features(
                img1,
                img2,
                features_to_swap=self.features_to_swap,
            )
            augmented.extend([aug1, aug2])

        # Handle last image if odd number
        if len(images) % 2:
            last_img = images[-1]
            aug_last, _ = self.swapper.swap_features(
                last_img,
                images[0],  # Swap with first image
                features_to_swap=self.features_to_swap,
            )
            augmented.append(aug_last)

        return augmented


def get_train_transforms(
    img_width: int,
    img_height: int,
    use_face_swap: bool = False,
) -> albu.Compose:
    if use_face_swap:
        transforms_list = [FaceSwapAugmentation()]
    else:
        transforms_list = []
    transforms_list.extend(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.HorizontalFlip(p=0.5),
            albu.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            albu.GaussianBlur(blur_limit=(3, 7), p=0.1),
            albu.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.1, 0.2),
                hole_width_range=(0.1, 0.2),
                p=0.3,
            ),
            albu.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ],
    )

    return albu.Compose(
        transforms_list,
    )


def get_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ],
    )
