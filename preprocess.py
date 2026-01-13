import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scade_net.configs import Config, load_config
from scade_net.data.preprocessing import preprocess_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess video dataset for SCADE-Net training'
    )

    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='./scade_net/dataset',
        help='Path to dataset root with real/ and fake/ subdirectories'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./preprocessed',
        help='Output directory for preprocessed data'
    )

    # Optional arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to extract per video (default: 100)'
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=10,
        help='Sample 1 frame every N frames (default: 10)'
    )
    parser.add_argument(
        '--no-defocus',
        action='store_true',
        help='Skip defocus map computation'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation split fraction (default: 0.15)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test split fraction (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Override with command line arguments
    dataset_root = args.dataset
    output_root = args.output

    print("=" * 60)
    print("SCADE-Net Preprocessing")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Max frames:   {args.max_frames}")
    print(f"Sampling rate: {args.sampling_rate}")
    print(f"Compute defocus: {not args.no_defocus}")
    print(f"Val split:    {args.val_split}")
    print(f"Test split:   {args.test_split}")
    print(f"Random seed:  {args.seed}")
    print("=" * 60)

    # Check dataset exists
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_root}")
        sys.exit(1)

    real_dir = dataset_path / 'real'
    fake_dir = dataset_path / 'fake'

    if not real_dir.exists() and not fake_dir.exists():
        print(f"Error: Expected 'real/' and/or 'fake/' subdirectories in {dataset_root}")
        sys.exit(1)

    # Run preprocessing
    metadata, splits = preprocess_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        max_frames_per_video=args.max_frames,
        frame_sampling_rate=args.sampling_rate,
        compute_defocus=not args.no_defocus,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Face crops: {output_root}/face_crops/")
    if not args.no_defocus:
        print(f"  Defocus maps: {output_root}/defocus_maps/")
    print(f"  Metadata: {output_root}/metadata.json")
    print(f"  Splits: {output_root}/splits.json")
    print()
    print("Next steps:")
    print("  1. Run training: python train.py --data ./preprocessed")
    print()


if __name__ == '__main__':
    main()