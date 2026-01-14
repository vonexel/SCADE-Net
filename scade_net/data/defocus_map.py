import cv2
import time
import numpy as np
from typing import Optional


class DefocusMapGenerator:
    def __init__(self, canny_threshold1: int = 50, canny_threshold2: int = 150,
                 edge_blur_kernel_size: int = 15, guided_filter_radius: int = 8, guided_filter_eps: float = 0.01):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.edge_blur_kernel_size = edge_blur_kernel_size
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps

        # opencv-contrib (ximgproc)
        self._has_ximgproc = hasattr(cv2, 'ximgproc')
        if not self._has_ximgproc:
            print("Warning: opencv-contrib-python not found. "
                  "Using Gaussian blur fallback instead of guided filter.")

    def detect_edges(self, rgb_image: np.ndarray) -> np.ndarray:
        # Convertation to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Applying Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.canny_threshold1,
            self.canny_threshold2
        )
        return edges

    def compute_edge_blur_map(self, edges: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        kernel_size = self.edge_blur_kernel_size

        # Compute local variance (inverse indicator of blur)
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # Numerical stability

        # Blur magnitude: high variance = sharp = low blur
        blur_magnitude = 1.0 / (1.0 + variance)

        # Keep only blur values at edge locations
        edge_blur_map = np.zeros_like(blur_magnitude)
        edge_blur_map[edges > 0] = blur_magnitude[edges > 0]

        return edge_blur_map

    def __call__(self, rgb_image: np.ndarray) -> np.ndarray:
        # Edge detection
        edges = self.detect_edges(rgb_image)

        # Edge blur map
        edge_blur = self.compute_edge_blur_map(edges, rgb_image)

        #  Propagate blur values using guided filter
        gray_guide = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        if self._has_ximgproc:
            defocus_map = cv2.ximgproc.guidedFilter(
                guide=gray_guide,
                src=edge_blur.astype(np.float32),
                radius=self.guided_filter_radius,
                eps=self.guided_filter_eps
            )
        else:
            # Gaussian blur
            defocus_map = cv2.GaussianBlur(
                edge_blur.astype(np.float32),
                (self.guided_filter_radius * 2 + 1,
                 self.guided_filter_radius * 2 + 1),
                0
            )

        # Normalization to [0, 1]
        defocus_min = defocus_map.min()
        defocus_max = defocus_map.max()
        if defocus_max - defocus_min > 1e-8:
            defocus_map = (defocus_map - defocus_min) / (defocus_max - defocus_min)
        else:
            defocus_map = np.zeros_like(defocus_map)

        return defocus_map[..., np.newaxis].astype(np.float32)

    def process_batch(self, images: np.ndarray, progress_callback: Optional[callable] = None) -> np.ndarray:
        batch_size = images.shape[0]
        defocus_maps = []

        for i in range(batch_size):
            defocus_map = self(images[i])
            defocus_maps.append(defocus_map)

            if progress_callback:
                progress_callback(i + 1, batch_size)

        return np.stack(defocus_maps, axis=0)


if __name__ == '__main__':
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Initialize generator
    generator = DefocusMapGenerator()

    # Warm-up
    _ = generator(test_image)

    # Benchmark
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        defocus_map = generator(test_image)
    elapsed = time.time() - start

    print("DefocusMapGenerator Benchmark:")
    print(f"  Input shape:  {test_image.shape}")
    print(f"  Output shape: {defocus_map.shape}")
    print(f"  Output range: [{defocus_map.min():.4f}, {defocus_map.max():.4f}]")
    print(f"  Time per image: {elapsed / num_runs * 1000:.1f} ms")
    print(f"  Has ximgproc: {generator._has_ximgproc}")