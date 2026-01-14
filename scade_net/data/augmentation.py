import cv2
import numpy as np
from typing import Optional


try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic augmentations.")


def get_augmentation_pipeline(is_training: bool = True, horizontal_flip_prob: float = 0.5, rotation_limit: int = 15, zoom_limit: float = 0.1,
                              brightness_limit: float = 0.2, hue_shift_limit: int = 10) -> Optional['A.Compose']:
    if not is_training:
        return None

    if not HAS_ALBUMENTATIONS:
        return None

    # These augmentations work with 4-channel images
    # but color augmentations will only affect first 3 channels
    transform = A.Compose([
        # Geometric augmentations (apply to all channels)
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.Rotate(
            limit=rotation_limit,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=zoom_limit,
            rotate_limit=rotation_limit,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
    ])

    return transform


class RGBDAugmentation:
    def __init__(self, horizontal_flip_prob: float = 0.5, rotation_limit: int = 15, zoom_limit: float = 0.1, brightness_limit: float = 0.2, hue_shift_limit: int = 10):
        if not HAS_ALBUMENTATIONS:
            self.transform = None
            return

        self.geometric = A.Compose([
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.Rotate(
                limit=rotation_limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=zoom_limit,
                rotate_limit=rotation_limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
        ])

        # Color transforms (apply only to RGB)
        self.color = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=brightness_limit,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=hue_shift_limit,
                sat_shift_limit=0,
                val_shift_limit=0,
                p=0.5
            ),
        ])

    def __call__(self, rgbd: np.ndarray) -> np.ndarray:
        if not HAS_ALBUMENTATIONS:
            return rgbd

        # Split RGB and D
        rgb = rgbd[:, :, :3]
        defocus = rgbd[:, :, 3:4]

        # Convert RGB to uint8 for augmentation
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Apply color augmentations to RGB
        rgb_aug = self.color(image=rgb_uint8)['image']

        # Convert back to float
        rgb_aug = rgb_aug.astype(np.float32) / 255.0

        # Combine RGB and D
        rgbd_color = np.concatenate([rgb_aug, defocus], axis=-1)

        # Apply geometric augmentations to full RGBD
        rgbd_aug = self.geometric(image=rgbd_color)['image']

        return rgbd_aug


if __name__ == '__main__':

    test_rgbd = np.random.rand(256, 256, 4).astype(np.float32)
    print("Augmentation Test:")
    print(f"  Input shape: {test_rgbd.shape}")
    print(f"  Input range: [{test_rgbd.min():.2f}, {test_rgbd.max():.2f}]")

    transform = get_augmentation_pipeline(is_training=True)
    if transform:
        augmented = transform(image=test_rgbd)['image']
        print(f"  Output shape: {augmented.shape}")
        print(f"  Output range: [{augmented.min():.2f}, {augmented.max():.2f}]")

    rgbd_aug = RGBDAugmentation()
    augmented = rgbd_aug(test_rgbd)
    print(f"\n  RGBD Augmentation:")
    print(f"  Output shape: {augmented.shape}")
    print(f"  Output range: [{augmented.min():.2f}, {augmented.max():.2f}]")