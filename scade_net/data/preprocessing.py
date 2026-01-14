import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from .face_detector import FaceDetector
from .defocus_map import DefocusMapGenerator


class VideoPreprocessor:
    """
    Extracts frames from videos, detects and aligns faces,
    generates defocus maps, and creates train/val/test splits.
    """
    def __init__(self, face_detector: Optional[FaceDetector] = None, defocus_generator: Optional[DefocusMapGenerator] = None,
                 max_frames_per_video: int = 100, frame_sampling_rate: int = 10, min_frames_per_video: int = 10, compute_defocus: bool = True):
        self.face_detector = face_detector or FaceDetector()
        self.defocus_generator = defocus_generator if compute_defocus else None
        self.max_frames_per_video = max_frames_per_video
        self.frame_sampling_rate = frame_sampling_rate
        self.min_frames_per_video = min_frames_per_video
        self.compute_defocus = compute_defocus

        if compute_defocus and defocus_generator is None:
            self.defocus_generator = DefocusMapGenerator()

    def process_video(self, video_path: Path, face_output_dir: Path, defocus_output_dir: Optional[Path] = None, label: int = 0) -> Dict:
        video_name = video_path.stem

        # Create output directories
        video_face_dir = face_output_dir / video_name
        video_face_dir.mkdir(parents=True, exist_ok=True)

        if defocus_output_dir:
            video_defocus_dir = defocus_output_dir / video_name
            video_defocus_dir.mkdir(parents=True, exist_ok=True)
        else:
            video_defocus_dir = None

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                'video': video_name,
                'status': 'error',
                'error': 'Cannot open video'
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame indices to sample
        frame_indices = list(range(
            0, total_frames, self.frame_sampling_rate
        ))[:self.max_frames_per_video]

        if len(frame_indices) < self.min_frames_per_video:
            cap.release()
            return {
                'video': video_name,
                'status': 'skipped',
                'reason': f'Too few frames ({len(frame_indices)})'
            }

        saved_frames = 0
        failed_detections = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect and crop face
            face_crop = self.face_detector.detect_and_crop(frame)

            if face_crop is None:
                failed_detections += 1
                continue

            # Save face crop (RGB -> BGR for OpenCV)
            frame_filename = f"frame_{idx:06d}.jpg"
            face_path = video_face_dir / frame_filename
            cv2.imwrite(str(face_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

            # Generate and save defocus map
            if self.defocus_generator and video_defocus_dir:
                defocus_map = self.defocus_generator(face_crop)
                defocus_path = video_defocus_dir / frame_filename.replace('.jpg', '.npy')
                np.save(str(defocus_path), defocus_map)

            saved_frames += 1
        cap.release()

        return {'video': video_name,
                'status': 'success',
                'label': label,
                'total_frames': total_frames,
                'fps': fps,
                'sampled_frames': len(frame_indices),
                'saved_frames': saved_frames,
                'failed_detections': failed_detections,
                'detection_rate': saved_frames / len(frame_indices) if frame_indices else 0
                }

    def process_dataset(self, dataset_root: str, output_root: str,
                        val_split: float = 0.15, test_split: float = 0.15, random_seed: int = 42) -> Tuple[List[Dict], Dict]:
        dataset_root = Path(dataset_root)
        output_root = Path(output_root)

        # Setup output directories
        face_crops_dir = output_root / 'face_crops'
        defocus_maps_dir = output_root / 'defocus_maps' if self.compute_defocus else None

        (face_crops_dir / 'real').mkdir(parents=True, exist_ok=True)
        (face_crops_dir / 'fake').mkdir(parents=True, exist_ok=True)

        if defocus_maps_dir:
            (defocus_maps_dir / 'real').mkdir(parents=True, exist_ok=True)
            (defocus_maps_dir / 'fake').mkdir(parents=True, exist_ok=True)

        real_videos = self._find_videos(dataset_root / 'real')
        fake_videos = self._find_videos(dataset_root / 'fake')

        print(f"Found {len(real_videos)} real videos")
        print(f"Found {len(fake_videos)} fake videos")

        metadata = []

        # Process real videos
        print(f"\nProcessing real videos...")
        for video_path in tqdm(real_videos, desc="Real"):
            result = self.process_video(
                video_path=video_path,
                face_output_dir=face_crops_dir / 'real',
                defocus_output_dir=defocus_maps_dir / 'real' if defocus_maps_dir else None,
                label=0
            )
            metadata.append(result)

        # Process fake videos
        print(f"\nProcessing fake videos...")
        for video_path in tqdm(fake_videos, desc="Fake"):
            result = self.process_video(
                video_path=video_path,
                face_output_dir=face_crops_dir / 'fake',
                defocus_output_dir=defocus_maps_dir / 'fake' if defocus_maps_dir else None,
                label=1
            )
            metadata.append(result)

        # Create splits
        splits = self._create_splits(
            metadata, val_split, test_split, random_seed
        )

        # Save metadata
        metadata_path = output_root / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to {metadata_path}")

        # Save splits
        splits_path = output_root / 'splits.json'
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Splits saved to {splits_path}")

        # Print statistics
        self._print_statistics(metadata, splits)
        return metadata, splits

    def _find_videos(self, directory: Path) -> List[Path]:
        """Find all video files in directory."""
        if not directory.exists():
            return []

        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            videos.extend(directory.glob(ext))
            videos.extend(directory.glob(ext.upper()))
        return sorted(videos)

    def _create_splits(self, metadata: List[Dict], val_split: float,
                       test_split: float, random_seed: int) -> Dict:
        random.seed(random_seed)

        successful = [m for m in metadata if m['status'] == 'success']
        real_videos = [m['video'] for m in successful if m['label'] == 0]
        fake_videos = [m['video'] for m in successful if m['label'] == 1]

        random.shuffle(real_videos)
        random.shuffle(fake_videos)

        def split_list(videos, val_frac, test_frac):
            n = len(videos)
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            return (videos[:n_test],  # test
                    videos[n_test:n_test + n_val],  # val
                    videos[n_test + n_val:]  # train
                    )

        real_test, real_val, real_train = split_list(real_videos, val_split, test_split)
        fake_test, fake_val, fake_train = split_list(fake_videos, val_split, test_split)

        return {'train': {'real': real_train, 'fake': fake_train},
                'val': {'real': real_val, 'fake': fake_val},
                'test': {'real': real_test, 'fake': fake_test}
                }

    def _print_statistics(self, metadata: List[Dict], splits: Dict):
        print("\n" + "=" * 60)
        print("Preprocessing Statistics")
        print("=" * 60)

        # Count by status
        status_counts = defaultdict(int)
        for m in metadata:
            status_counts[m['status']] += 1

        print("\nProcessing status:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        # Successful videos
        successful = [m for m in metadata if m['status'] == 'success']
        real_count = sum(1 for m in successful if m['label'] == 0)
        fake_count = sum(1 for m in successful if m['label'] == 1)

        print(f"\nSuccessful videos:")
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")

        # Frame statistics
        total_saved = sum(m.get('saved_frames', 0) for m in successful)
        total_failed = sum(m.get('failed_detections', 0) for m in successful)
        avg_rate = np.mean([m.get('detection_rate', 0) for m in successful])

        print(f"\nFrame statistics:")
        print(f"  Total saved frames: {total_saved}")
        print(f"  Failed detections: {total_failed}")
        print(f"  Average detection rate: {avg_rate:.1%}")

        # Split statistics
        print("\nData splits:")
        for split_name, split_data in splits.items():
            total = len(split_data['real']) + len(split_data['fake'])
            print(f"  {split_name}: {total} videos "
                  f"({len(split_data['real'])} real, {len(split_data['fake'])} fake)")


def preprocess_dataset(dataset_root: str, output_root: str, max_frames_per_video: int = 100, frame_sampling_rate: int = 10,
                       compute_defocus: bool = True, val_split: float = 0.15, test_split: float = 0.15, random_seed: int = 42) -> Tuple[List[Dict], Dict]:
    preprocessor = VideoPreprocessor(
        max_frames_per_video=max_frames_per_video,
        frame_sampling_rate=frame_sampling_rate,
        compute_defocus=compute_defocus
    )

    return preprocessor.process_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )


if __name__ == '__main__':
    print("VideoPreprocessor module loaded successfully.")
    print("Use preprocess_dataset() to preprocess a video dataset.")
    print("\nExample:")
    print("  preprocess_dataset(")
    print("      dataset_root='./dataset',")
    print("      output_root='./preprocessed',")
    print("      max_frames_per_video=100,")
    print("      frame_sampling_rate=10,")
    print("      compute_defocus=True")
    print("  )")