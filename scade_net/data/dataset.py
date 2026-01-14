import cv2
import json
import torch
import random
import numpy as np
from pathlib import Path
from .augmentation import RGBDAugmentation
from .defocus_map import DefocusMapGenerator
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DeepfakeDataset(Dataset):
    """
    Deepfake Detection Dataset with RGBD inputs.

    Expects data organized as:
        rgb_dir/real/video_name/frame_001.jpg
        rgb_dir/fake/video_name/frame_001.jpg
        defocus_dir/real/video_name/frame_001.npy (optional)
        defocus_dir/fake/video_name/frame_001.npy (optional)

    If defocus maps are not precomputed, they will be generated on-the-fly.

    Args:
        rgb_dir: Directory with RGB face crops
        defocus_dir: Directory with precomputed defocus maps (optional)
        video_list: List of video names to include (optional)
        labels_dict: Dictionary mapping video name to label (optional)
        transform: Albumentations transform (optional)
        defocus_generator: DefocusMapGenerator for on-the-fly generation
        max_frames_per_video: Maximum frames per video
        is_training: Whether this is training data
    """

    def __init__(self, rgb_dir: str, defocus_dir: Optional[str] = None, video_list: Optional[List[str]] = None, labels_dict: Optional[Dict[str, int]] = None,
                 transform: Optional[RGBDAugmentation] = None, defocus_generator: Optional[DefocusMapGenerator] = None, max_frames_per_video: int = 100, is_training: bool = True):
        self.rgb_dir = Path(rgb_dir)
        self.defocus_dir = Path(defocus_dir) if defocus_dir else None
        self.transform = transform
        self.defocus_generator = defocus_generator
        self.max_frames_per_video = max_frames_per_video
        self.is_training = is_training

        self.samples = self._build_samples(video_list, labels_dict)

    def _build_samples(self, video_list: Optional[List[str]], labels_dict: Optional[Dict[str, int]]) -> List[Tuple[Path, int]]:
        samples = []

        if video_list is None:
            # Scan directory structure
            for class_name, label in [('real', 0), ('fake', 1)]:
                class_dir = self.rgb_dir / class_name
                if not class_dir.exists():
                    continue

                for video_dir in class_dir.iterdir():
                    if not video_dir.is_dir():
                        continue

                    frames = list(video_dir.glob('*.jpg'))
                    frames += list(video_dir.glob('*.png'))
                    frames = sorted(frames)[:self.max_frames_per_video]

                    for frame_path in frames:
                        samples.append((frame_path, label))
        else:
            # Use provided video list
            for video_name in video_list:
                if labels_dict and video_name in labels_dict:
                    label = labels_dict[video_name]
                else:
                    # Try to infer from path
                    label = -1

                # Check real directory
                video_dir = self.rgb_dir / 'real' / video_name
                if video_dir.exists():
                    label = 0
                    frames = sorted(list(video_dir.glob('*.jpg')) +
                                    list(video_dir.glob('*.png')))
                    frames = frames[:self.max_frames_per_video]
                    for frame_path in frames:
                        samples.append((frame_path, label))
                    continue

                # Check fake directory
                video_dir = self.rgb_dir / 'fake' / video_name
                if video_dir.exists():
                    label = 1
                    frames = sorted(list(video_dir.glob('*.jpg')) +
                                    list(video_dir.glob('*.png')))
                    frames = frames[:self.max_frames_per_video]
                    for frame_path in frames:
                        samples.append((frame_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_path, label = self.samples[idx]

        # Load RGB image
        rgb = cv2.imread(str(frame_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256))

        # Load or generate defocus map
        if self.defocus_dir:
            relative_path = frame_path.relative_to(self.rgb_dir)
            defocus_path = self.defocus_dir / relative_path.with_suffix('.npy')

            if defocus_path.exists():
                defocus = np.load(str(defocus_path))
            elif self.defocus_generator:
                defocus = self.defocus_generator(rgb)
            else:
                defocus = np.zeros((256, 256, 1), dtype=np.float32)
        elif self.defocus_generator:
            defocus = self.defocus_generator(rgb)
        else:
            defocus = np.zeros((256, 256, 1), dtype=np.float32)

        # Normalize RGB to [0, 1]
        rgb_normalized = rgb.astype(np.float32) / 255.0

        # Concatenate RGBD
        rgbd = np.concatenate([rgb_normalized, defocus], axis=-1)

        # Apply augmentation
        if self.transform and self.is_training:
            rgbd = self.transform(rgbd)

        # Convert to tensor: [H, W, 4] -> [4, H, W]
        rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return rgbd_tensor, label_tensor


def create_video_level_splits(rgb_dir: str, val_split: float = 0.15, test_split: float = 0.15, random_seed: int = 42) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    random.seed(random_seed)

    rgb_path = Path(rgb_dir)
    labels_dict = {}
    real_videos = []
    fake_videos = []

    # Collect real videos
    real_dir = rgb_path / 'real'
    if real_dir.exists():
        for video_dir in real_dir.iterdir():
            if video_dir.is_dir():
                video_name = video_dir.name
                real_videos.append(video_name)
                labels_dict[video_name] = 0

    # Collect fake videos
    fake_dir = rgb_path / 'fake'
    if fake_dir.exists():
        for video_dir in fake_dir.iterdir():
            if video_dir.is_dir():
                video_name = video_dir.name
                fake_videos.append(video_name)
                labels_dict[video_name] = 1

    # Shuffle
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    def split_list(videos, val_frac, test_frac):
        n = len(videos)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        return (
            videos[n_test + n_val:],  # train
            videos[n_test:n_test + n_val],  # val
            videos[:n_test]  # test
        )

    real_train, real_val, real_test = split_list(real_videos, val_split, test_split)
    fake_train, fake_val, fake_test = split_list(fake_videos, val_split, test_split)

    train_videos = real_train + fake_train
    val_videos = real_val + fake_val
    test_videos = real_test + fake_test

    random.shuffle(train_videos)
    random.shuffle(val_videos)
    random.shuffle(test_videos)

    return train_videos, val_videos, test_videos, labels_dict


def create_dataloaders(rgb_dir: str, defocus_dir: Optional[str] = None, splits_path: Optional[str] = None, batch_size: int = 75, num_workers: int = 4,
                       val_split: float = 0.15, test_split: float = 0.15, random_seed: int = 42, use_defocus_generator: bool = True,
                       use_weighted_sampler: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load or create splits
    if splits_path and Path(splits_path).exists():
        with open(splits_path, 'r') as f:
            splits = json.load(f)

        train_videos = splits['train']['real'] + splits['train']['fake']
        val_videos = splits['val']['real'] + splits['val']['fake']
        test_videos = splits['test']['real'] + splits['test']['fake']

        labels_dict = {}
        for name in splits['train']['real'] + splits['val']['real'] + splits['test']['real']:
            labels_dict[name] = 0
        for name in splits['train']['fake'] + splits['val']['fake'] + splits['test']['fake']:
            labels_dict[name] = 1
    else:
        train_videos, val_videos, test_videos, labels_dict = create_video_level_splits(
            rgb_dir, val_split, test_split, random_seed
        )

    # Create defocus generator if needed
    defocus_generator = None
    if use_defocus_generator and defocus_dir is None:
        defocus_generator = DefocusMapGenerator()

    # Create augmentation
    train_transform = RGBDAugmentation()

    # Create datasets
    train_dataset = DeepfakeDataset(
        rgb_dir=rgb_dir,
        defocus_dir=defocus_dir,
        video_list=train_videos,
        labels_dict=labels_dict,
        transform=train_transform,
        defocus_generator=defocus_generator,
        is_training=True
    )

    val_dataset = DeepfakeDataset(
        rgb_dir=rgb_dir,
        defocus_dir=defocus_dir,
        video_list=val_videos,
        labels_dict=labels_dict,
        transform=None,
        defocus_generator=defocus_generator,
        is_training=False
    )

    test_dataset = DeepfakeDataset(
        rgb_dir=rgb_dir,
        defocus_dir=defocus_dir,
        video_list=test_videos,
        labels_dict=labels_dict,
        transform=None,
        defocus_generator=defocus_generator,
        is_training=False
    )

    # Calculate class weights from training set for balanced learning
    train_labels = [label for _, label in train_dataset.samples]
    num_real = sum(1 for l in train_labels if l == 0)
    num_fake = sum(1 for l in train_labels if l == 1)
    total = len(train_labels)

    # Create weighted sampler if requested
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        # Calculate weights for each class
        # Give higher weight to minority class (real)
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts

        # Assign weight to each sample based on its class
        sample_weights = np.array([class_weights[label] for label in train_labels])

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
        print("  Using WeightedRandomSampler for class balancing")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # pos_weight for BCE: weight for positive class (fake=1)
    # num_negatives / num_positives
    pos_weight = num_real / num_fake if num_fake > 0 else 1.0

    print(f"Dataloaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Class distribution (train): Real={num_real} ({100*num_real/total:.1f}%), Fake={num_fake} ({100*num_fake/total:.1f}%)")
    print(f"  BCE pos_weight: {pos_weight:.3f}")

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == '__main__':
    # Test dataset
    print("Dataset module loaded successfully.")
    print("Use create_dataloaders() to create train/val/test dataloaders.")