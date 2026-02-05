"""
Experiment 2: Linear Probes for Role Classification

Research Question: Can we linearly decode role identity from representations?
Which layers encode role most strongly?

Method:
1. For each layer, train a linear classifier to predict speaker from activations
2. Evaluate on held-out test set
3. Compare accuracy across layers
4. Validate PCA findings from Exp1

Hypothesis:
- Probe accuracy should peak in mid-layers (15-20 for Llama, 4-6 for GPT-2)
- Accuracy curve should align with PCA silhouette scores
- High accuracy indicates linearly separable role representations
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from activation_collector import ActivationData


class RoleProbe(nn.Module):
    """Simple linear probe for role classification."""

    def __init__(self, hidden_dim: int, num_roles: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_roles)

    def forward(self, hidden_states):
        return self.classifier(hidden_states)


class RoleProbeTrainer:
    """Train and evaluate linear probes for role classification."""

    def __init__(self, activation_file: str, test_split: float = 0.2, seed: int = 42):
        """
        Args:
            activation_file: Path to HDF5 activation file
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
        """
        self.activation_data = ActivationData.load_hdf5(activation_file)
        self.test_split = test_split
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*70}")
        print("ROLE PROBE TRAINER")
        print(f"{'='*70}")
        print(f"\nLoaded: {activation_file}")
        print(f"  Transcript: {self.activation_data.transcript_id}")
        print(f"  Condition: {self.activation_data.condition}")
        print(f"  Model: {self.activation_data.model_name}")
        print(f"  Layers: {self.activation_data.layers}")
        print(f"  Tokens: {len(self.activation_data.tokens)}")

        # Get unique speakers
        self.speakers = sorted(set(
            s for s in self.activation_data.speaker_labels
            if s != "UNKNOWN"
        ))
        self.num_roles = len(self.speakers)
        self.speaker_to_id = {s: i for i, s in enumerate(self.speakers)}

        print(f"  Speakers: {self.num_roles}")
        for speaker in self.speakers:
            count = sum(1 for s in self.activation_data.speaker_labels if s == speaker)
            print(f"    - {speaker}: {count} tokens")

        # Prepare data splits
        self._prepare_data_splits()

    def _prepare_data_splits(self):
        """Create train/test splits for each layer."""
        print(f"\nPreparing train/test splits (test={self.test_split:.0%})...")

        # Filter valid indices (non-UNKNOWN speakers)
        valid_indices = [
            i for i, s in enumerate(self.activation_data.speaker_labels)
            if s != "UNKNOWN"
        ]

        # Shuffle indices
        np.random.shuffle(valid_indices)

        # Split
        n_test = int(len(valid_indices) * self.test_split)
        self.test_indices = valid_indices[:n_test]
        self.train_indices = valid_indices[n_test:]

        print(f"  Train samples: {len(self.train_indices)}")
        print(f"  Test samples: {len(self.test_indices)}")

        # Prepare labels
        labels = [
            self.speaker_to_id[self.activation_data.speaker_labels[i]]
            for i in valid_indices
        ]
        self.train_labels = torch.tensor([labels[i] for i in range(len(self.train_indices))], dtype=torch.long)
        self.test_labels = torch.tensor([labels[i] for i in range(len(self.test_indices))], dtype=torch.long)

    def train_probe(
        self,
        layer: int,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64
    ) -> Dict:
        """
        Train a linear probe for a specific layer.

        Args:
            layer: Which layer to train on
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization strength
            batch_size: Batch size for training

        Returns:
            Dict with training results and metrics
        """
        print(f"\n{'='*70}")
        print(f"Training probe for layer {layer}")
        print(f"{'='*70}")

        # Get activations for this layer
        acts = self.activation_data.activations[layer]
        hidden_dim = acts.shape[1]

        # Extract train/test data
        X_train = torch.tensor(acts[self.train_indices], dtype=torch.float32)
        X_test = torch.tensor(acts[self.test_indices], dtype=torch.float32)

        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")

        # Initialize probe
        probe = RoleProbe(hidden_dim, self.num_roles)
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        train_losses = []
        train_accs = []
        test_accs = []

        best_test_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            probe.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Mini-batch training
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train[batch_indices]
                batch_y = self.train_labels[batch_indices]

                optimizer.zero_grad()
                logits = probe(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            train_loss = epoch_loss / (len(X_train) / batch_size)
            train_acc = correct / total

            # Evaluation
            probe.eval()
            with torch.no_grad():
                logits_test = probe(X_test)
                _, predicted_test = torch.max(logits_test, 1)
                test_acc = (predicted_test == self.test_labels).float().mean().item()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

        print(f"\n  Best test accuracy: {best_test_acc:.3f} at epoch {best_epoch+1}")

        # Final evaluation with confusion matrix
        probe.eval()
        with torch.no_grad():
            logits_test = probe(X_test)
            _, predicted_test = torch.max(logits_test, 1)

            # Confusion matrix
            confusion = torch.zeros(self.num_roles, self.num_roles, dtype=torch.long)
            for t, p in zip(self.test_labels, predicted_test):
                confusion[t, p] += 1

        # Random baseline
        random_baseline = 1.0 / self.num_roles
        improvement = best_test_acc - random_baseline

        print(f"  Random baseline: {random_baseline:.3f}")
        print(f"  Improvement: {improvement:.3f} ({improvement/random_baseline:.1%} above random)")

        return {
            'layer': layer,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'confusion_matrix': confusion.numpy(),
            'random_baseline': random_baseline,
            'improvement': improvement,
            'final_train_acc': train_accs[-1],
            'final_test_acc': test_accs[-1]
        }

    def train_all_layers(
        self,
        output_dir: str,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> List[Dict]:
        """Train probes for all layers and generate visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []

        for layer in self.activation_data.layers:
            results = self.train_probe(
                layer=layer,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay
            )

            # Save confusion matrix plot
            self._plot_confusion_matrix(
                results['confusion_matrix'],
                layer,
                str(output_path)
            )

            # Save metrics (remove large arrays for JSON)
            metrics = {
                'layer': int(results['layer']),
                'best_test_acc': float(results['best_test_acc']),
                'best_epoch': int(results['best_epoch']),
                'random_baseline': float(results['random_baseline']),
                'improvement': float(results['improvement']),
                'final_train_acc': float(results['final_train_acc']),
                'final_test_acc': float(results['final_test_acc'])
            }
            all_results.append(metrics)

        # Save summary JSON
        summary_path = output_path / f"{self.activation_data.transcript_id}_{self.activation_data.condition}_probe_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'transcript_id': self.activation_data.transcript_id,
                'condition': self.activation_data.condition,
                'model': self.activation_data.model_name,
                'num_roles': self.num_roles,
                'speakers': self.speakers,
                'train_samples': len(self.train_indices),
                'test_samples': len(self.test_indices),
                'layers': all_results
            }, f, indent=2)

        print(f"\n✓ Saved summary: {summary_path}")

        # Plot metrics across layers
        self._plot_metrics_across_layers(all_results, str(output_path))

        return all_results

    def _plot_confusion_matrix(self, confusion: np.ndarray, layer: int, output_dir: str):
        """Plot confusion matrix for a layer."""
        fig, ax = plt.subplots(figsize=(8, 7))

        # Normalize to percentages
        confusion_norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            confusion_norm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.speakers,
            yticklabels=self.speakers,
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax
        )

        ax.set_xlabel('Predicted Speaker', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Speaker', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Confusion Matrix - Layer {layer}\n'
            f'{self.activation_data.condition.capitalize()} | {self.activation_data.transcript_id}',
            fontsize=13, fontweight='bold', pad=15
        )

        plt.tight_layout()

        output_path = Path(output_dir) / f"{self.activation_data.transcript_id}_{self.activation_data.condition}_layer{layer}_confusion.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def _plot_metrics_across_layers(self, all_results: List[Dict], output_dir: str):
        """Plot probe accuracy and improvement across layers."""
        layers = [r['layer'] for r in all_results]
        test_accs = [r['best_test_acc'] for r in all_results]
        improvements = [r['improvement'] for r in all_results]
        random_baseline = all_results[0]['random_baseline']

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Accuracy plot
        axes[0].plot(layers, test_accs, marker='o', linewidth=2.5, markersize=10,
                    color='#2ecc71', label='Test Accuracy')
        axes[0].axhline(y=random_baseline, color='#e74c3c', linestyle='--',
                       linewidth=2, label=f'Random Baseline ({random_baseline:.3f})')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title(
            f'Linear Probe Accuracy Across Layers\n'
            f'{self.activation_data.condition.capitalize()} | {self.activation_data.transcript_id}',
            fontsize=14, fontweight='bold', pad=15
        )
        axes[0].legend(fontsize=11, loc='lower right')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_ylim(0, 1)

        # Add value labels
        for layer, acc in zip(layers, test_accs):
            axes[0].text(layer, acc + 0.02, f'{acc:.3f}',
                        ha='center', va='bottom', fontsize=9)

        # Improvement over random
        axes[1].bar(layers, improvements, color='#3498db', alpha=0.7, width=0.8)
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Layer', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Improvement over Random', fontsize=12, fontweight='bold')
        axes[1].set_title('Absolute Improvement Above Chance', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add value labels
        for layer, imp in zip(layers, improvements):
            axes[1].text(layer, imp + 0.005, f'{imp:.3f}',
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        output_path = Path(output_dir) / f"{self.activation_data.transcript_id}_{self.activation_data.condition}_probe_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train linear probes for role classification")
    parser.add_argument('--activations', required=True,
                       help='Path to HDF5 activation file')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 regularization strength')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Initialize trainer
    trainer = RoleProbeTrainer(
        activation_file=args.activations,
        test_split=args.test_split,
        seed=args.seed
    )

    # Train probes for all layers
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: LINEAR PROBE TRAINING")
    print(f"{'='*70}")

    results = trainer.train_all_layers(
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output}")

    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"  Layer {r['layer']:2d}: Accuracy={r['best_test_acc']:.3f}, "
              f"Improvement={r['improvement']:+.3f}")


if __name__ == "__main__":
    main()
