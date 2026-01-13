import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scade_net import SCADENet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scade_net.data import DefocusMapGenerator, FaceDetector


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class SCADENetVisualizer:
    def __init__(self, model_path: str, device: Optional[torch.device] = None, threshold: float = 0.5):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.threshold = threshold

        # Load model
        print(f"Loading model from {model_path}...")
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

        print(f"Model loaded successfully on {device}")

    def preprocess_face(self, face_rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
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
        return rgbd_tensor.to(self.device), defocus_map.squeeze()

    def compute_grad_cam(self, input_tensor: torch.Tensor, target_layer_name: str = 'block4_conv') -> Tuple[np.ndarray, float]:
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Register hooks
        target_layer = getattr(self.model, target_layer_name)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        input_tensor.requires_grad = True
        logits = self.model(input_tensor)

        # Get prediction
        prediction = torch.sigmoid(logits).item()

        # Backward pass
        self.model.zero_grad()
        logits.backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Compute Grad-CAM
        grad = gradients[0][0]  # [C, H, W]
        act = activations[0][0]  # [C, H, W]

        weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * act).sum(dim=0)  # [H, W]
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to 256x256
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam, prediction

    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None, show_plot: bool = True) -> Optional[dict]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face
        face_rgb = self.face_detector.detect_and_crop(image_rgb)
        if face_rgb is None:
            print(f"Error: No face detected in {image_path}")
            return None

        rgbd_tensor, defocus_map = self.preprocess_face(face_rgb)

        grad_cam, prediction = self.compute_grad_cam(rgbd_tensor)

        # Determine label
        label = 'FAKE' if prediction > self.threshold else 'REAL'
        confidence = prediction if label == 'FAKE' else 1 - prediction

        face_display = cv2.resize(face_rgb, (256, 256))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(face_display)
        axes[0, 0].set_title('Original Face Crop', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(defocus_map, cmap='jet')
        axes[0, 1].set_title('Defocus Blur Map', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        im2 = axes[0, 2].imshow(grad_cam, cmap='jet')
        axes[0, 2].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

        axes[1, 0].imshow(face_display)
        axes[1, 0].imshow(defocus_map, cmap='jet', alpha=0.4)
        axes[1, 0].set_title('Defocus Overlay', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(face_display)
        axes[1, 1].imshow(grad_cam, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # Prediction summary
        axes[1, 2].axis('off')
        color = 'red' if label == 'FAKE' else 'green'

        summary_text = f"""
PREDICTION: {label}
Score: {prediction:.4f}
Confidence: {confidence:.2%}
Threshold: {self.threshold}

Input: RGBD (4 channels)
  • RGB face crop (256x256)
  • Defocus blur map

Model: SCADE-Net
  • ~28k parameters
  • SiLU activation
  • ECA attention
  • Single-Center Loss
        """

        axes[1, 2].text(
            0.1, 0.5, summary_text,
            fontsize=11,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2)
        )

        # Main title
        filename = Path(image_path).name
        fig.suptitle(
            f'SCADE-Net Prediction Visualization: {filename}',
            fontsize=16,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

        return {
            'path': str(image_path),
            'prediction': label,
            'score': float(prediction),
            'confidence': float(confidence)
        }

    def visualize_video_frames(self, video_path: str, output_dir: str, max_frames: int = 5, sampling_rate: int = 30):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, sampling_rate))[:max_frames]

        video_name = Path(video_path).stem
        results = []

        print(f"\nProcessing {len(frame_indices)} frames from {video_path}...")

        for idx in tqdm(frame_indices, desc="Processing frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_rgb = self.face_detector.detect_and_crop(frame_rgb)
            if face_rgb is None:
                print(f"  Frame {idx}: No face detected")
                continue

            temp_path = output_path / f"temp_frame_{idx}.jpg"
            cv2.imwrite(str(temp_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            save_path = output_path / f"{video_name}_frame_{idx:06d}.png"
            result = self.visualize_prediction(
                str(temp_path),
                save_path=str(save_path),
                show_plot=False
            )

            temp_path.unlink()
            if result:
                results.append(result)
        cap.release()

        # Print summary
        if results:
            avg_score = np.mean([r['score'] for r in results])
            avg_label = 'FAKE' if avg_score > self.threshold else 'REAL'

            print(f"\n{'='*60}")
            print(f"Video Analysis Summary: {video_name}")
            print(f"{'='*60}")
            print(f"Frames analyzed: {len(results)}")
            print(f"Average score: {avg_score:.4f}")
            print(f"Aggregate label: {avg_label}")
            print(f"Results saved to: {output_dir}")
            print(f"{'='*60}\n")

    def visualize_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in input_path.glob('*')
            if f.suffix.lower() in image_extensions
        ]
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        print(f"\nProcessing {len(image_files)} images from {input_dir}...")

        results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            save_path = output_path / f"{image_file.stem}_visualization.png"
            result = self.visualize_prediction(
                str(image_file),
                save_path=str(save_path),
                show_plot=False
            )
            if result:
                results.append(result)

        if results:
            real_count = sum(1 for r in results if r['prediction'] == 'REAL')
            fake_count = len(results) - real_count

            print(f"\n{'='*60}")
            print(f"Batch Analysis Summary")
            print(f"{'='*60}")
            print(f"Total images: {len(results)}")
            print(f"Real: {real_count}")
            print(f"Fake: {fake_count}")
            print(f"Results saved to: {output_dir}")
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SCADE-Net predictions on images or videos'
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
        '--image',
        type=str,
        help='Path to single image file'
    )
    group.add_argument(
        '--video',
        type=str,
        help='Path to video file'
    )
    group.add_argument(
        '--input',
        type=str,
        help='Path to directory with images'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save visualization(s)'
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
        default=5,
        help='Max frames to visualize from video (default: 5)'
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=30,
        help='Sample 1 frame every N frames from video (default: 30)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or cuda:N)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots, only save them'
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize visualizer
    visualizer = SCADENetVisualizer(
        model_path=args.model,
        device=device,
        threshold=args.threshold
    )

    # Run visualization
    if args.image:
        print(f"\nVisualizing image: {args.image}")

        if args.output is None:
            args.output = Path(args.image).stem + "_visualization.png"

        visualizer.visualize_prediction(
            args.image,
            save_path=args.output,
            show_plot=not args.no_display
        )

    elif args.video:
        print(f"\nVisualizing video: {args.video}")

        if args.output is None:
            args.output = f"./video_visualizations_{Path(args.video).stem}/"

        visualizer.visualize_video_frames(
            args.video,
            output_dir=args.output,
            max_frames=args.max_frames,
            sampling_rate=args.sampling_rate
        )

    else:  # directory
        print(f"\nVisualizing directory: {args.input}")

        if args.output is None:
            args.output = "./visualizations/"

        visualizer.visualize_directory(
            args.input,
            output_dir=args.output
        )


if __name__ == '__main__':
    main()