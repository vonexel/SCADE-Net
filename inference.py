import sys
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
sys.path.insert(0, str(Path(__file__).parent))
from scade_net import SCADENet, Config
from scade_net.data import DefocusMapGenerator, FaceDetector


class SCADENetInference:
    def __init__(self, model_path: str, device: Optional[torch.device] = None, threshold: float = 0.5):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.threshold = threshold

        # Load model
        self.model = SCADENet()
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

        # Initialize preprocessors
        self.face_detector = FaceDetector(device=device)
        self.defocus_generator = DefocusMapGenerator()

        print(f"Model loaded from {model_path}")
        print(f"Device: {device}")

    def preprocess_face(self, face_rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face crop for inference.

        Args:
            face_rgb: [H, W, 3] RGB face image

        Returns:
            RGBD tensor [1, 4, 256, 256]
        """
        # Resize to 256x256
        face_resized = cv2.resize(face_rgb, (256, 256))

        # Generate defocus map
        defocus_map = self.defocus_generator(face_resized)

        # Normalize RGB to [0, 1]
        rgb_normalized = face_resized.astype(np.float32) / 255.0

        # Concatenate RGBD
        rgbd = np.concatenate([rgb_normalized, defocus_map], axis=-1)

        # Convert to tensor [1, 4, 256, 256]
        rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float().unsqueeze(0)

        return rgbd_tensor.to(self.device)

    @torch.no_grad()
    def predict_face(self, face_rgb: np.ndarray) -> float:
        """
        Get prediction for a single face.

        Args:
            face_rgb: [H, W, 3] RGB face image

        Returns:
            Prediction score (0=real, 1=fake)
        """
        rgbd = self.preprocess_face(face_rgb)
        logits = self.model(rgbd)
        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)
        return probabilities.squeeze().cpu().item()

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not load image: {image_path}'}

        # Detect face
        face = self.face_detector.detect_and_crop(image)
        if face is None:
            return {'error': 'No face detected'}

        # Get prediction
        score = self.predict_face(face)
        label = 'fake' if score > self.threshold else 'real'

        return {
            'path': str(image_path),
            'score': score,
            'label': label,
            'confidence': score if label == 'fake' else 1 - score
        }

    def process_video(self, video_path: str, max_frames: int = 10, sampling_rate: int = 30) -> Dict:
        """
        Process a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to sample
            sampling_rate: Sample 1 frame every N frames

        Returns:
            Dictionary with aggregated prediction
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Could not open video: {video_path}'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, sampling_rate))[:max_frames]

        frame_scores = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Detect face
            face = self.face_detector.detect_and_crop(frame)
            if face is None:
                continue

            # Get prediction
            score = self.predict_face(face)
            frame_scores.append(score)

        cap.release()

        if not frame_scores:
            return {'error': 'No faces detected in video'}

        # Aggregate scores
        avg_score = np.mean(frame_scores)
        label = 'fake' if avg_score > self.threshold else 'real'

        return {
            'path': str(video_path),
            'score': float(avg_score),
            'label': label,
            'confidence': float(avg_score if label == 'fake' else 1 - avg_score),
            'num_frames': len(frame_scores),
            'frame_scores': [float(s) for s in frame_scores]
        }

    def process_directory(self, input_dir: str, max_frames: int = 10) -> List[Dict]:
        input_path = Path(input_dir)
        results = []

        # Find all files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

        files = list(input_path.glob('*'))
        files = [f for f in files if f.suffix.lower() in image_extensions | video_extensions]

        for file_path in tqdm(files, desc="Processing"):
            if file_path.suffix.lower() in image_extensions:
                result = self.process_image(str(file_path))
            else:
                result = self.process_video(str(file_path), max_frames=max_frames)

            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Run SCADE-Net inference on images or videos'
    )

    # Model
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    # Input (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single image file'
    )
    group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to single video file'
    )
    group.add_argument(
        '--input',
        type=str,
        help='Path to directory with images/videos'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output JSON file'
    )

    # Options
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=10,
        help='Max frames to sample from videos (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or cuda:N)'
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize inference
    inferencer = SCADENetInference(
        model_path=args.model,
        device=device,
        threshold=args.threshold
    )

    # Run inference
    if args.image:
        print(f"\nProcessing image: {args.image}")
        results = [inferencer.process_image(args.image)]

    elif args.video:
        print(f"\nProcessing video: {args.video}")
        results = [inferencer.process_video(args.video, max_frames=args.max_frames)]

    else:
        print(f"\nProcessing directory: {args.input}")
        results = inferencer.process_directory(args.input, max_frames=args.max_frames)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    for result in results:
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  {Path(result['path']).name}:")
            print(f"    Label: {result['label'].upper()}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Confidence: {result['confidence']:.2%}")
            if 'num_frames' in result:
                print(f"    Frames analyzed: {result['num_frames']}")

    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print()


if __name__ == '__main__':
    main()