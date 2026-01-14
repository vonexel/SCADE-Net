import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field



@dataclass
class Config:
    # Data paths
    dataset_root: str = "./dataset"
    preprocessed_root: str = "./preprocessed"
    rgb_dir: str = "./preprocessed/face_crops"
    defocus_dir: str = "./preprocessed/defocus_maps"
    splits_path: str = "./preprocessed/splits.json"
    output_dir: str = "./outputs"

    # Preprocessing
    max_frames_per_video: int = 100
    frame_sampling_rate: int = 10
    min_frames_per_video: int = 10
    face_size: int = 256
    face_margin: float = 0.3
    min_face_size: int = 60
    face_detection_threshold: float = 0.9
    compute_defocus: bool = True
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    edge_blur_kernel_size: int = 15
    guided_filter_radius: int = 8
    guided_filter_eps: float = 0.01

    # Model
    input_size: int = 256
    input_channels: int = 4
    constrained_conv_option: str = "A"
    eca_kernel_size: int = 3
    dropout_rate: float = 0.5

    # Single-Center Loss
    scl_feature_dim: int = 1024
    scl_margin: float = 0.5
    scl_alpha: float = 0.5
    scl_lambda: float = 0.1

    # Training
    batch_size: int = 75
    num_epochs: int = 50
    initial_lr: float = 1e-3
    min_lr: float = 1e-6

    # LR Scheduler options
    lr_scheduler: str = "step"  # "step" or "cosine"
    lr_decay_step: int = 1000   # for step scheduler
    lr_decay_factor: float = 0.1  # for step scheduler
    warmup_epochs: int = 0  # for cosine scheduler

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0

    # Training options
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Loss function options
    use_focal_loss: bool = False  # Use Focal Loss instead of BCE
    focal_alpha: float = 0.25     # Weight for positive class
    focal_gamma: float = 2.0      # Focusing parameter

    # Data loading
    num_workers: int = 4
    use_weighted_sampler: bool = False  # Balance classes with WeightedRandomSampler
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # Augmentation
    aug_horizontal_flip_prob: float = 0.5
    aug_rotation_limit: int = 15
    aug_zoom_limit: float = 0.1
    aug_brightness_limit: float = 0.2
    aug_hue_shift_limit: int = 10

    # Inference
    video_aggregation_frames: int = 10
    threshold: float = 0.5
    inference_batch_size: int = 32

    # Visualization
    tsne_max_samples: int = 1000
    tsne_perplexity: int = 30
    figure_dpi: int = 150

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.constrained_conv_option in ['A', 'B'], \
            "constrained_conv_option must be 'A' or 'B'"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert 0 < self.test_split < 1, "test_split must be between 0 and 1"
        assert self.val_split + self.test_split < 1, \
            "val_split + test_split must be less than 1"
        assert self.lr_scheduler in ['step', 'cosine'], \
            "lr_scheduler must be 'step' or 'cosine'"
        assert self.warmup_epochs >= 0, "warmup_epochs must be non-negative"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def save(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Filter to only known fields
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)

    def update(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key '{key}'")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Config object
    """
    if config_path is None:
        return Config()

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config file not found: {config_path}. Using defaults.")
        return Config()

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Flatten nested config
    flat_config = {}

    def flatten(d, prefix=''):
        for key, value in d.items():
            if isinstance(value, dict):
                flatten(value, prefix + key + '_')
            else:
                # Convert nested keys to flat format
                flat_key = prefix + key
                # Handle special mappings
                key_mapping = {
                    'data_dataset_root': 'dataset_root',
                    'data_preprocessed_root': 'preprocessed_root',
                    'data_rgb_dir': 'rgb_dir',
                    'data_defocus_dir': 'defocus_dir',
                    'data_splits_path': 'splits_path',
                    'data_output_dir': 'output_dir',
                    'preprocessing_': '',
                    'model_': '',
                    'training_': '',
                    'augmentation_': 'aug_',
                    'inference_': '',
                    'visualization_': '',
                    'scl_lambda': 'scl_lambda',
                }

                for old, new in key_mapping.items():
                    if flat_key.startswith(old):
                        flat_key = new + flat_key[len(old):]
                        break

                flat_config[flat_key] = value

    flatten(yaml_config)

    return Config.from_dict(flat_config)


if __name__ == '__main__':
    # Test config
    config = Config()
    print("Default configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Test loading
    print("\nConfig class loaded successfully.")