import cv2
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple


try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("Warning: facenet-pytorch not installed. "
          "Face detection will use OpenCV cascade fallback.")


class FaceDetector:
    """
    MTCNN detects faces and facial landmarks, then aligns faces
    based on eye positions for consistent orientation.

    If facenet-pytorch is not installed, falls back to OpenCV's
    Haar cascade detector with reduced accuracy.
    """
    def __init__(self, device: Optional[torch.device] = None, face_size: int = 256,
                 margin: float = 0.3, min_face_size: int = 60, threshold: float = 0.9):
        self.face_size = face_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.threshold = threshold

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if HAS_MTCNN:
            self.mtcnn = MTCNN(
                image_size=face_size,
                margin=int(face_size * margin),
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, threshold],
                factor=0.709,
                post_process=False,
                select_largest=True,
                keep_all=False,
                device=device
            )
            self._use_mtcnn = True
        else:
            # OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self._use_mtcnn = False

    def detect_and_crop(self, frame: np.ndarray, return_bbox: bool = False) -> Optional[np.ndarray]:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._use_mtcnn:
            return self._detect_mtcnn(frame_rgb, return_bbox)
        else:
            return self._detect_cascade(frame_rgb, return_bbox)

    def _detect_mtcnn(self, frame_rgb: np.ndarray, return_bbox: bool = False) -> Optional[np.ndarray]:
        pil_image = Image.fromarray(frame_rgb)
        try:
            face = self.mtcnn(pil_image)

            if face is None:
                return (None, None) if return_bbox else None

            # MTCNN returns tensor [3, H, W]. Range depends on post_process:
            # - post_process=True  -> [-1, 1]
            # - post_process=False -> [0, 255]
            # Convert to numpy [H, W, 3] uint8 in [0, 255].
            face_np = face.permute(1, 2, 0).cpu().numpy()
            #face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
            if face_np.min() >= -1.0 and face_np.max() <= 1.0:
                face_np = (face_np + 1) / 2 * 255.0
            face_np = np.clip(face_np, 0, 255).astype(np.uint8)

            if return_bbox:
                # Get bounding box
                boxes, _ = self.mtcnn.detect(pil_image)
                if boxes is not None and len(boxes) > 0:
                    bbox = boxes[0].astype(int)
                else:
                    bbox = None
                return face_np, bbox

            return face_np

        except Exception:
            return (None, None) if return_bbox else None

    def _detect_cascade(self, frame_rgb: np.ndarray, return_bbox: bool = False) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        if len(faces) == 0:
            return (None, None) if return_bbox else None

        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add margin
        margin_pixels = int(max(w, h) * self.margin)
        x1 = max(0, x - margin_pixels)
        y1 = max(0, y - margin_pixels)
        x2 = min(frame_rgb.shape[1], x + w + margin_pixels)
        y2 = min(frame_rgb.shape[0], y + h + margin_pixels)

        # Crop and resize
        face_crop = frame_rgb[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (self.face_size, self.face_size))

        if return_bbox:
            return face_crop, (x, y, w, h)

        return face_crop

    # Detect faces in a batch of frames
    def detect_batch(self,frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if self._use_mtcnn:
            pil_images = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                for f in frames
            ]

            try:
                faces = self.mtcnn(pil_images)
                results = []
                for face in faces:
                    if face is None:
                        results.append(None)
                    else:
                        face_np = face.permute(1, 2, 0).cpu().numpy()
                        #face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
                        if face_np.min() >= -1.0 and face_np.max() <= 1.0:
                            face_np = (face_np + 1) / 2 * 255.0
                        face_np = np.clip(face_np, 0, 255).astype(np.uint8)
                        results.append(face_np)

                return results

            except Exception:
                return [None] * len(frames)
        else:
            # Sequential detection with cascade
            return [self.detect_and_crop(f) for f in frames]


if __name__ == '__main__':
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    detector = FaceDetector()
    print(f"FaceDetector initialized:")
    print(f"  Using MTCNN: {detector._use_mtcnn}")
    print(f"  Device: {detector.device}")
    print(f"  Face size: {detector.face_size}")
    print(f"  Margin: {detector.margin}")

    result = detector.detect_and_crop(test_frame)
    print(f"\nTest detection:")
    print(f"  Input shape: {test_frame.shape}")
    print(f"  Result: {result.shape if result is not None else None}")