import io
import sys
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from typing import Optional, Tuple
from scade_net import SCADENet
from scade_net.data import DefocusMapGenerator, FaceDetector


sys.path.insert(0, str(Path(__file__).parent))


# Configure Streamlit page
st.set_page_config(
    page_title="SCADE-Net Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


class SCADENetVisualizer:
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

    def preprocess_face(self,face_rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
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

        target_layer = getattr(self.model, target_layer_name)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        input_tensor.requires_grad = True
        logits = self.model(input_tensor)

        prediction = torch.sigmoid(logits).item()

        self.model.zero_grad()
        logits.backward()

        forward_handle.remove()
        backward_handle.remove()

        grad = gradients[0][0]  # [C, H, W]
        act = activations[0][0]  # [C, H, W]

        weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * act).sum(dim=0)  # [H, W]
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to 256x256
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        return cam, prediction

    def process_image(self, image_rgb: np.ndarray) -> Optional[dict]:
        face_rgb = self.face_detector.detect_and_crop(image_rgb)
        if face_rgb is None:
            return None


        rgbd_tensor, defocus_map = self.preprocess_face(face_rgb)


        grad_cam, prediction = self.compute_grad_cam(rgbd_tensor)


        label = 'FAKE' if prediction > self.threshold else 'REAL'
        confidence = prediction if label == 'FAKE' else 1 - prediction


        face_display = cv2.resize(face_rgb, (256, 256))
        return {
            'face': face_display,
            'defocus_map': defocus_map,
            'grad_cam': grad_cam,
            'prediction': label,
            'score': float(prediction),
            'confidence': float(confidence)
        }


@st.cache_resource
def load_model(model_path: str, threshold: float):
    """Load and cache the SCADE-Net model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return SCADENetVisualizer(model_path, device, threshold)


def create_visualization(result: dict) -> plt.Figure:
    """
    Create matplotlib visualization from processing result.

    Args:
        result: Dictionary with processing results

    Returns:
        Matplotlib figure
    """
    face = result['face']
    defocus_map = result['defocus_map']
    grad_cam = result['grad_cam']
    label = result['prediction']
    prediction = result['score']
    confidence = result['confidence']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, Defocus Map, Grad-CAM
    axes[0, 0].imshow(face)
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

    # Row 2: Overlays and Prediction
    axes[1, 0].imshow(face)
    axes[1, 0].imshow(defocus_map, cmap='jet', alpha=0.4)
    axes[1, 0].set_title('Defocus Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(face)
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
    plt.tight_layout()
    return fig


def process_video_frames(visualizer: SCADENetVisualizer, video_bytes: bytes, max_frames: int = 5, sampling_rate: int = 30):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not open video file")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, total_frames, sampling_rate))[:max_frames]

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, idx in enumerate(frame_indices):
        status_text.text(f"Processing frame {i+1}/{len(frame_indices)}...")
        progress_bar.progress((i + 1) / len(frame_indices))

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        result = visualizer.process_image(frame_rgb)
        if result:
            result['frame_idx'] = idx
            results.append(result)

    cap.release()
    Path(tmp_path).unlink()
    status_text.empty()
    progress_bar.empty()
    return results


def main():
    st.title("🔍 SCADE-Net Deepfake Detector")
    st.markdown("""
    **Spatial Channel Attention and Depth-Enhanced Network** for deepfake detection.
    Upload an image or video to analyze whether faces are real or AI-generated.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./outputs/best_model.pth",
        help="Path to the trained SCADE-Net model checkpoint"
    )

    # Threshold
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Score threshold for fake classification"
    )

    # Video options
    st.sidebar.subheader("Video Options")
    max_frames = st.sidebar.number_input(
        "Max Frames",
        min_value=1,
        max_value=50,
        value=5,
        help="Maximum number of frames to analyze"
    )
    sampling_rate = st.sidebar.number_input(
        "Sampling Rate",
        min_value=1,
        max_value=120,
        value=30,
        help="Analyze 1 frame every N frames"
    )

    # Device info
    device = 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'
    st.sidebar.info(f"**Device:** {device}")

    # Check if model exists
    if not Path(model_path).exists():
        st.error(f"Model not found at: {model_path}")
        st.info("Please specify a valid model path in the sidebar.")
        return

    # Load model (cached)
    try:
        with st.spinner("Loading SCADE-Net model..."):
            visualizer = load_model(model_path, threshold)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    st.header("📤 Upload Media")
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov'],
        help="Drag and drop or click to browse"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'image':
            # Process image
            st.header("📊 Analysis Results")

            # Display original image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            # Process
            with st.spinner("Analyzing image..."):
                image_rgb = np.array(image.convert('RGB'))
                result = visualizer.process_image(image_rgb)

            if result is None:
                st.error("❌ No face detected in the image. Please upload an image with a visible face.")
                return

            # Display results
            with col2:
                st.subheader("Prediction")

                # Big prediction display
                label = result['prediction']
                score = result['score']
                confidence = result['confidence']

                if label == 'FAKE':
                    st.error(f"### 🚨 FAKE DETECTED")
                    st.metric("Fake Score", f"{score:.4f}", f"{confidence:.1%} confidence")
                else:
                    st.success(f"### ✅ REAL FACE")
                    st.metric("Real Score", f"{1-score:.4f}", f"{confidence:.1%} confidence")

            # Detailed visualization
            st.subheader("Detailed Analysis")
            fig = create_visualization(result)
            st.pyplot(fig)
            plt.close(fig)

            # Download button
            buf = io.BytesIO()
            fig = create_visualization(result)
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            st.download_button(
                label="📥 Download Visualization",
                data=buf,
                file_name=f"scadenet_analysis_{uploaded_file.name}.png",
                mime="image/png"
            )

        elif file_type == 'video':
            # Process video
            st.header("🎥 Video Analysis")

            st.info(f"Processing up to {max_frames} frames (1 every {sampling_rate} frames)...")

            video_bytes = uploaded_file.read()

            with st.spinner("Analyzing video frames..."):
                results = process_video_frames(
                    visualizer,
                    video_bytes,
                    max_frames=max_frames,
                    sampling_rate=sampling_rate
                )

            if not results:
                st.error("No faces detected in any video frames.")
                return

            # Summary statistics
            st.subheader("📊 Summary")
            col1, col2, col3, col4 = st.columns(4)

            scores = [r['score'] for r in results]
            avg_score = np.mean(scores)
            fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
            real_count = len(results) - fake_count

            with col1:
                st.metric("Frames Analyzed", len(results))
            with col2:
                st.metric("Average Score", f"{avg_score:.4f}")
            with col3:
                st.metric("Fake Frames", fake_count)
            with col4:
                st.metric("Real Frames", real_count)

            # Overall verdict
            if avg_score > threshold:
                st.error(f"### 🚨 VIDEO LIKELY FAKE ({fake_count}/{len(results)} frames)")
            else:
                st.success(f"### ✅ VIDEO LIKELY REAL ({real_count}/{len(results)} frames)")

            # Display individual frames
            st.subheader("Individual Frame Analysis")

            for i, result in enumerate(results):
                with st.expander(
                    f"Frame {result['frame_idx']} - {result['prediction']} "
                    f"(Score: {result['score']:.4f})",
                    expanded=(i == 0)
                ):
                    fig = create_visualization(result)
                    st.pyplot(fig)
                    plt.close(fig)

    else:
        # Show sample instructions
        st.info("""
        👆 **Get started:**
        1. Upload an image or video file using the uploader above
        2. Wait for the analysis to complete
        3. Review the detailed visualization and results
        4. Download the analysis if needed

        **Supported formats:**
        - Images: JPG, PNG, BMP
        - Videos: MP4, AVI, MOV
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>
    Powered by SCADE-Net | ~28k parameters | Real-time deepfake detection<br>
    Uses RGBD input (RGB + Defocus Map) with ECA attention and Single-Center Loss
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()