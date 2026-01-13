import os
import sys
import torch
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scade_net.data import create_dataloaders
from scade_net.training import SCADENetTrainer, set_seed
from scade_net.training.evaluator import evaluate_model
from scade_net import SCADENet, Config, load_config
from scade_net.visualization import visualize_training_history


def main():
    parser = argparse.ArgumentParser(
        description='Train SCADE-Net for deepfake detection'
    )

    # Data arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='./preprocessed',
        help='Path to preprocessed data directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./outputs',
        help='Output directory for checkpoints and logs'
    )

    # Config
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Initial learning rate (overrides config)'
    )
    parser.add_argument(
        '--scl-lambda',
        type=float,
        default=None,
        help='Single-Center Loss weight (overrides config)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Early stopping patience (overrides config)'
    )

    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or cuda:N)'
    )

    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip evaluation after training'
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Override config with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.initial_lr = args.lr
    if args.scl_lambda:
        config.scl_lambda = args.scl_lambda
    if args.patience:
        config.early_stopping_patience = args.patience
    if args.workers:
        config.num_workers = args.workers

    # Setup paths
    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update config paths
    config.rgb_dir = str(data_root / 'face_crops')
    config.defocus_dir = str(data_root / 'defocus_maps')
    config.splits_path = str(data_root / 'splits.json')
    config.output_dir = str(output_dir)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print configuration
    print("=" * 60)
    print("SCADE-Net Training")
    print("=" * 60)
    print(f"Data directory:    {data_root}")
    print(f"Output directory:  {output_dir}")
    print(f"Device:            {device}")
    print(f"Epochs:            {config.num_epochs}")
    print(f"Batch size:        {config.batch_size}")
    print(f"Initial LR:        {config.initial_lr}")
    print(f"SCL lambda:        {config.scl_lambda}")
    print(f"Mixed precision:   {config.use_mixed_precision}")
    print("=" * 60)

    # Check data exists
    if not Path(config.rgb_dir).exists():
        print(f"\nError: Face crops directory not found: {config.rgb_dir}")
        print("Run preprocessing first: python preprocess.py --dataset ./dataset")
        sys.exit(1)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
        rgb_dir=config.rgb_dir,
        defocus_dir=config.defocus_dir,
        splits_path=config.splits_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed,
        use_weighted_sampler=config.use_weighted_sampler
    )

    # Create model
    print("\nCreating model...")
    model = SCADENet(
        input_channels=config.input_channels,
        constrained_conv_option=config.constrained_conv_option,
        eca_kernel_size=config.eca_kernel_size,
        dropout_rate=config.dropout_rate
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Within budget:        {total_params < 30000}")

    # Create trainer with class weight support
    trainer = SCADENetTrainer(model, config, device, pos_weight=pos_weight)

    # Resume if checkpoint provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scl' in checkpoint:
            trainer.scl_criterion.load_state_dict(checkpoint['scl'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        output_dir=output_dir
    )

    # Plot training history
    visualize_training_history(
        history,
        save_path=str(output_dir / 'training_history.png')
    )

    # Evaluate on test set
    if not args.no_eval:
        print("\nEvaluating on test set...")
        trainer.load_best_model()
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=str(output_dir)
        )

    # Save final config
    config.save(str(output_dir / 'config.yaml'))

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Best model:     {output_dir}/best_model.pth")
    print(f"  Final model:    {output_dir}/final_model.pth")
    print(f"  Training plot:  {output_dir}/training_history.png")
    print(f"  Eval results:   {output_dir}/evaluation_results.png")
    print(f"  Config:         {output_dir}/config.yaml")
    print()


if __name__ == '__main__':
    main()