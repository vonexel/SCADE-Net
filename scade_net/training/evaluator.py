import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from ..models import SCADENet
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


class SCADENetEvaluator:
    def __init__(self, model: SCADENet, device: Optional[torch.device] = None, threshold: float = 0.5):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.model = model.to(device)
        self.threshold = threshold

    @torch.no_grad()
    def evaluate_frame_level(self, dataloader: DataLoader) -> Dict[str, Union[float, np.ndarray]]:
        self.model.eval()

        all_predictions = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="Evaluating frames"):
            images = images.to(self.device)
            logits = self.model(images)
            # Apply sigmoid to convert logits to probabilities
            predictions = torch.sigmoid(logits)

            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_labels.extend(labels.numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_binary = (all_predictions > self.threshold).astype(int)

        # Compute metrics
        auc = roc_auc_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_binary, average='binary', pos_label=1
        )
        conf_matrix = confusion_matrix(all_labels, all_binary)

        return {'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix,
                'predictions': all_predictions,
                'labels': all_labels
                }

    @torch.no_grad()
    def evaluate_video_level(self, dataloader: DataLoader, video_frame_mapping: Dict[str, List[int]]) -> Dict[str, Union[float, np.ndarray]]:
        self.model.eval()

        frame_predictions = []
        frame_labels = []

        for images, labels in tqdm(dataloader, desc="Evaluating frames"):
            images = images.to(self.device)
            logits = self.model(images)
            # Apply sigmoid to convert logits to probabilities
            predictions = torch.sigmoid(logits)

            frame_predictions.extend(predictions.squeeze().cpu().numpy())
            frame_labels.extend(labels.numpy())

        frame_predictions = np.array(frame_predictions)
        frame_labels = np.array(frame_labels)

        video_predictions = {}
        video_labels = {}

        for video_name, frame_indices in video_frame_mapping.items():
            valid_indices = [i for i in frame_indices if i < len(frame_predictions)]
            if valid_indices:
                video_predictions[video_name] = np.mean(frame_predictions[valid_indices])
                video_labels[video_name] = frame_labels[valid_indices[0]]

        # Convert to arrays
        video_names = list(video_predictions.keys())
        predictions_array = np.array([video_predictions[v] for v in video_names])
        labels_array = np.array([video_labels[v] for v in video_names])
        binary_array = (predictions_array > self.threshold).astype(int)

        # Compute metrics
        auc = roc_auc_score(labels_array, predictions_array)
        accuracy = accuracy_score(labels_array, binary_array)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_array, binary_array, average='binary', pos_label=1
        )
        conf_matrix = confusion_matrix(labels_array, binary_array)

        return {'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix,
                'predictions': predictions_array,
                'labels': labels_array,
                'video_names': video_names
                }

    def print_results(self, results: Dict, level: str = "Frame"):
        print(f"\n{'=' * 50}")
        print(f"{level}-Level Evaluation Results")
        print(f"{'=' * 50}")
        print(f"  AUC:       {results['auc']:.4f}")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={results['confusion_matrix'][0, 0]:5d}  "
              f"FP={results['confusion_matrix'][0, 1]:5d}")
        print(f"  FN={results['confusion_matrix'][1, 0]:5d}  "
              f"TP={results['confusion_matrix'][1, 1]:5d}")
        print(f"{'=' * 50}")

    def plot_results(self, results: Dict, title: str = "Frame-Level", save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(results['labels'], results['predictions'])
        axes[0].plot(fpr, tpr, 'b-', linewidth=2,
                     label=f"AUC = {results['auc']:.4f}")
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'{title} ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3)

        # Confusion Matrix
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=axes[1]
        )
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title(f'{title} Confusion Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def plot_prediction_distribution(self, results: Dict, title: str = "Frame-Level", save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 6))

        real_preds = results['predictions'][results['labels'] == 0]
        fake_preds = results['predictions'][results['labels'] == 1]

        bins = np.linspace(0, 1, 50)
        ax.hist(real_preds, bins=bins, alpha=0.6, color='green',
                label='Real', density=True)
        ax.hist(fake_preds, bins=bins, alpha=0.6, color='red',
                label='Fake', density=True)

        ax.axvline(self.threshold, color='black', linestyle='--',
                   label=f'Threshold ({self.threshold})')

        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Prediction Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()


def evaluate_model( model: SCADENet, test_loader: DataLoader, device: Optional[torch.device] = None,
                    output_dir: Optional[str] = None, threshold: float = 0.5) -> Dict:
    evaluator = SCADENetEvaluator(model, device, threshold)
    results = evaluator.evaluate_frame_level(test_loader)

    evaluator.print_results(results, "Frame")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_results(
            results, "Frame-Level",
            str(output_dir / 'evaluation_results.png')
        )
        evaluator.plot_prediction_distribution(
            results, "Frame-Level",
            str(output_dir / 'prediction_distribution.png')
        )
    return results


if __name__ == '__main__':
    print("Evaluator module loaded successfully.")
    print("Use evaluate_model() to evaluate a trained SCADE-Net model.")