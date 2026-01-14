import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from ..models import SCADENet
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


def visualize_conv_filters(model: SCADENet, layer_name: str = 'block1_conv', save_path: Optional[str] = None):
    # Get the layer
    layer = getattr(model, layer_name)

    # Extract weights
    if hasattr(layer, 'weight'):
        weights = layer.weight.data.cpu().numpy()
    elif hasattr(layer, '_get_constrained_kernel'):
        weights = layer._get_constrained_kernel().detach().cpu().numpy()
    else:
        print(f"Cannot extract weights from {layer_name}")
        return

    n_filters = weights.shape[0]
    n_channels = weights.shape[1]

    fig, axes = plt.subplots(
        nrows=n_filters,
        ncols=n_channels,
        figsize=(n_channels * 2, n_filters * 2)
    )

    if n_filters == 1:
        axes = axes.reshape(1, -1)
    if n_channels == 1:
        axes = axes.reshape(-1, 1)

    channel_names = ['R', 'G', 'B', 'Defocus'] if n_channels == 4 else [f'Ch{i}' for i in range(n_channels)]

    for i in range(n_filters):
        for j in range(n_channels):
            filter_weight = weights[i, j]

            # Normalize for visualization
            vmin, vmax = filter_weight.min(), filter_weight.max()
            if vmax - vmin > 1e-8:
                filter_norm = (filter_weight - vmin) / (vmax - vmin)
            else:
                filter_norm = np.zeros_like(filter_weight)

            axes[i, j].imshow(filter_norm, cmap='viridis', interpolation='nearest')
            axes[i, j].axis('off')

            if i == 0:
                axes[i, j].set_title(channel_names[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f'F{i}', fontsize=10, rotation=0, labelpad=15)

    plt.suptitle(f'Convolutional Filters: {layer_name}', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def visualize_feature_maps(model: SCADENet, input_image: torch.Tensor, label: str = "Sample", layer_names: Optional[List[str]] = None, save_path: Optional[str] = None):
    """
    Visualize feature maps (activations) for a given input.

    Args:
        model: Trained model
        input_image: [1, 4, 256, 256] input tensor
        label: Label for the sample ('Real' or 'Fake')
        layer_names: List of layer names to visualize
        save_path: Path to save figure
    """
    if layer_names is None:
        layer_names = ['block1_bn', 'block2_bn', 'block3_bn', 'block4_bn']

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handles = []
    for name in layer_names:
        layer = getattr(model, name)
        handle = layer.register_forward_hook(hook_fn(name))
        handles.append(handle)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    for handle in handles:
        handle.remove()

    n_layers = len(layer_names)
    n_channels_show = 8

    fig, axes = plt.subplots(n_layers, n_channels_show, figsize=(16, n_layers * 2))

    for i, layer_name in enumerate(layer_names):
        feature_map = activations[layer_name][0]  # [C, H, W]
        n_channels = min(n_channels_show, feature_map.shape[0])

        for j in range(n_channels):
            channel = feature_map[j].cpu().numpy()

            axes[i, j].imshow(channel, cmap='viridis')
            axes[i, j].axis('off')

            if i == 0:
                axes[i, j].set_title(f'Ch {j}', fontsize=8)

        axes[i, 0].set_ylabel(layer_name.replace('_', '\n'), fontsize=8)

    plt.suptitle(f'Feature Maps: {label} Input', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_eca_attention(model: SCADENet, input_image: torch.Tensor, label: str = "Sample", save_path: Optional[str] = None):
    attention_weights = {}

    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            y = module.avg_pool(x)
            y = y.squeeze(-1).transpose(-1, -2)
            y = module.conv(y)
            y = y.transpose(-1, -2).unsqueeze(-1)
            weights = torch.sigmoid(y).squeeze().detach()
            attention_weights[name] = weights
        return hook

    eca_layers = ['block3_eca', 'block4_eca']
    handles = []

    for name in eca_layers:
        layer = getattr(model, name)
        handle = layer.register_forward_hook(hook_fn(name))
        handles.append(handle)

    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    for handle in handles:
        handle.remove()

    fig, axes = plt.subplots(1, len(eca_layers), figsize=(len(eca_layers) * 6, 4))

    if len(eca_layers) == 1:
        axes = [axes]

    for i, layer_name in enumerate(eca_layers):
        weights = attention_weights[layer_name].cpu().numpy()
        channels = np.arange(len(weights))

        axes[i].bar(channels, weights, color='steelblue', alpha=0.7)
        axes[i].set_xlabel('Channel Index')
        axes[i].set_ylabel('Attention Weight')
        axes[i].set_title(f'{layer_name}\n({len(weights)} channels)')
        axes[i].set_ylim([0, 1.1])
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle(f'ECA Channel Attention: {label} Input', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def visualize_grad_cam(model: SCADENet, input_image: torch.Tensor, target_layer_name: str = 'block4_conv', save_path: Optional[str] = None):
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = getattr(model, target_layer_name)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    input_image.requires_grad = True
    output = model(input_image)
    prediction = output.item()

    model.zero_grad()
    output.backward()

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

    # Upsample
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    # Prepare RGB for visualization
    rgb = input_image[0, :3].permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('Input RGB')
    axes[0].axis('off')

    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(rgb)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    label = 'Fake' if prediction > 0.5 else 'Real'
    axes[2].set_title(f'Overlay (Pred: {label}, {prediction:.2f})')
    axes[2].axis('off')

    plt.suptitle('Grad-CAM Visualization', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_tsne(model: SCADENet, dataloader: DataLoader, device: torch.device, max_samples: int = 1000, save_path: Optional[str] = None):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            if len(features_list) * images.size(0) >= max_samples:
                break

            images = images.to(device)
            _, features = model(images, return_features=True)

            features_list.append(features.cpu())
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0).numpy()[:max_samples]
    labels = torch.cat(labels_list, dim=0).numpy()[:max_samples]

    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    real_mask = (labels == 0)
    fake_mask = (labels == 1)

    plt.scatter(features_2d[real_mask, 0], features_2d[real_mask, 1],
                c='green', alpha=0.6, s=20, label='Real', edgecolors='k', linewidths=0.5)
    plt.scatter(features_2d[fake_mask, 0], features_2d[fake_mask, 1],
                c='red', alpha=0.6, s=20, label='Fake', edgecolors='k', linewidths=0.5)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Feature Space')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_scl_distances(model: SCADENet, scl_center: torch.Tensor, dataloader: DataLoader,
                            device: torch.device, save_path: Optional[str] = None):
    model.eval()
    real_distances = []
    fake_distances = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing distances"):
            images = images.to(device)
            _, features = model(images, return_features=True)

            center = scl_center.unsqueeze(0).to(device)
            distances = torch.norm(features - center, dim=1).cpu().numpy()

            real_mask = (labels == 0)
            fake_mask = (labels == 1)

            real_distances.extend(distances[real_mask])
            fake_distances.extend(distances[fake_mask])

    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, max(max(real_distances), max(fake_distances)), 50)

    plt.hist(real_distances, bins=bins, alpha=0.6, color='green',
             label=f'Real (mean={np.mean(real_distances):.2f})', density=True)
    plt.hist(fake_distances, bins=bins, alpha=0.6, color='red',
             label=f'Fake (mean={np.mean(fake_distances):.2f})', density=True)

    plt.xlabel('Distance to Learned Center')
    plt.ylabel('Density')
    plt.title('Single-Center Loss: Distance Distributions')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    print(f"Mean distance (Real): {np.mean(real_distances):.3f}")
    print(f"Mean distance (Fake): {np.mean(fake_distances):.3f}")
    print(f"Separation: {np.mean(fake_distances) - np.mean(real_distances):.3f}")


def visualize_training_history(history: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # AUC
    if 'val_auc' in history:
        axes[0, 1].plot(epochs, history['val_auc'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].grid(alpha=0.3)

    # Accuracy
    if 'val_acc' in history:
        axes[1, 0].plot(epochs, history['val_acc'], 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].grid(alpha=0.3)

    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'c-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_defocus_map(rgb_image: np.ndarray, defocus_map: np.ndarray, save_path: Optional[str] = None):
    if defocus_map.ndim == 3:
        defocus_map = defocus_map.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

    axes[1].imshow(defocus_map, cmap='jet')
    axes[1].set_title('Defocus Blur Map')
    axes[1].axis('off')

    axes[2].imshow(rgb_image)
    axes[2].imshow(defocus_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def comprehensive_visualization(model: SCADENet, scl_center: Optional[torch.Tensor], test_loader: DataLoader,
                                device: torch.device, output_dir: str = 'visualizations/'):
    """
    Run all visualization analyses and save results.

    Args:
        model: Trained model
        scl_center: Learned center from SingleCenterLoss
        test_loader: Test data loader
        device: Torch device
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running comprehensive visualization...")

    # Get sample images
    sample_real = None
    sample_fake = None

    for images, labels in test_loader:
        for i, label in enumerate(labels):
            if label == 0 and sample_real is None:
                sample_real = images[i:i+1]
            if label == 1 and sample_fake is None:
                sample_fake = images[i:i+1]
            if sample_real is not None and sample_fake is not None:
                break
        if sample_real is not None and sample_fake is not None:
            break

    print("\n1. Visualizing convolutional filters...")
    visualize_conv_filters(model, 'block1_conv', str(output_dir / 'filters_block1.png'))

    print("\n2. Visualizing feature maps...")
    if sample_real is not None:
        sample_real = sample_real.to(device)
        visualize_feature_maps(model, sample_real, 'Real', save_path=str(output_dir / 'fmaps_real.png'))
    if sample_fake is not None:
        sample_fake = sample_fake.to(device)
        visualize_feature_maps(model, sample_fake, 'Fake', save_path=str(output_dir / 'fmaps_fake.png'))

    print("\n3. Visualizing ECA attention...")
    if sample_real is not None:
        visualize_eca_attention(model, sample_real, 'Real', str(output_dir / 'eca_real.png'))
    if sample_fake is not None:
        visualize_eca_attention(model, sample_fake, 'Fake', str(output_dir / 'eca_fake.png'))

    print("\n4. Visualizing Grad-CAM...")
    if sample_fake is not None:
        visualize_grad_cam(model, sample_fake, save_path=str(output_dir / 'gradcam_fake.png'))

    print("\n5. Visualizing t-SNE feature space...")
    visualize_tsne(model, test_loader, device, max_samples=500, save_path=str(output_dir / 'tsne.png'))

    if scl_center is not None:
        print("\n6. Visualizing SCL distances...")
        visualize_scl_distances(model, scl_center, test_loader, device, str(output_dir / 'scl_distances.png'))

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    print("Visualization module loaded successfully!")