"""
Experiment 1: PCA Visualization of Speaker Representations

Research Question: Do models form distinct speaker representations that cluster in activation space?

Method:
1. Extract activations for each speaker's turns across layers
2. Apply PCA to reduce to 2D/3D
3. Color by speaker, compare across conditions

Hypothesis:
- Labeled: Clear speaker clusters
- Unlabeled: Clusters emerge if model infers speakers
- Corrupted: Fragmented/confused clusters
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
except ImportError:
    print("Warning: sklearn/matplotlib not installed. Install with: pip install scikit-learn matplotlib seaborn")

from activation_collector import ActivationData


class SpeakerClusteringAnalyzer:
    """Analyze speaker clustering in activation space."""

    def __init__(self, activation_file: str):
        """Load activations from HDF5 file."""
        self.activation_data = ActivationData.load_hdf5(activation_file)
        self.transcript_id = self.activation_data.transcript_id
        self.condition = self.activation_data.condition
        self.model_name = self.activation_data.model_name

        print(f"\nLoaded: {activation_file}")
        print(f"  Transcript: {self.transcript_id}")
        print(f"  Condition: {self.condition}")
        print(f"  Model: {self.model_name}")
        print(f"  Layers: {self.activation_data.layers}")
        print(f"  Tokens: {len(self.activation_data.tokens)}")

        # Get unique speakers
        self.speakers = sorted(set(
            s for s in self.activation_data.speaker_labels
            if s != "UNKNOWN"
        ))
        print(f"  Speakers: {len(self.speakers)}")
        for speaker in self.speakers:
            count = sum(1 for s in self.activation_data.speaker_labels if s == speaker)
            print(f"    - {speaker}: {count} tokens")

    def analyze_layer(self, layer: int, n_components: int = 3) -> Dict:
        """
        Analyze speaker clustering for a specific layer.

        Args:
            layer: Which layer to analyze
            n_components: Number of PCA components (2 or 3)

        Returns:
            Dict with PCA results and metrics
        """
        print(f"\nAnalyzing layer {layer}...")

        # Get activations for this layer
        acts = self.activation_data.activations[layer]  # (seq_len, hidden_dim)
        labels = self.activation_data.speaker_labels

        # Filter out UNKNOWN tokens
        valid_indices = [i for i, s in enumerate(labels) if s != "UNKNOWN"]
        acts_filtered = acts[valid_indices]
        labels_filtered = [labels[i] for i in valid_indices]

        print(f"  Valid tokens: {len(valid_indices)} / {len(labels)}")
        print(f"  Activation shape: {acts_filtered.shape}")

        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(acts_filtered)

        print(f"  PCA variance explained: {pca.explained_variance_ratio_}")
        print(f"  Total variance: {pca.explained_variance_ratio_.sum():.3f}")

        # Compute clustering metrics
        # Convert speaker labels to numeric
        speaker_to_id = {s: i for i, s in enumerate(self.speakers)}
        numeric_labels = [speaker_to_id[s] for s in labels_filtered]

        try:
            silhouette = silhouette_score(components, numeric_labels)
            davies_bouldin = davies_bouldin_score(components, numeric_labels)
        except:
            silhouette = 0.0
            davies_bouldin = 0.0

        print(f"  Silhouette score: {silhouette:.3f} (higher is better, range [-1, 1])")
        print(f"  Davies-Bouldin score: {davies_bouldin:.3f} (lower is better)")

        return {
            'layer': layer,
            'components': components,
            'labels': labels_filtered,
            'pca': pca,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'total_variance': float(pca.explained_variance_ratio_.sum()),
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'n_samples': len(valid_indices),
            'speakers': self.speakers
        }

    def plot_2d_clusters(self, results: Dict, output_path: str):
        """Create 2D scatter plot of speaker clusters."""
        components = results['components']
        labels = results['labels']
        layer = results['layer']

        # Set up color palette
        palette = sns.color_palette("husl", len(self.speakers))
        speaker_colors = {s: palette[i] for i, s in enumerate(self.speakers)}

        # Create plot
        plt.figure(figsize=(10, 8))
        for speaker in self.speakers:
            mask = [l == speaker for l in labels]
            speaker_points = components[mask]

            plt.scatter(
                speaker_points[:, 0],
                speaker_points[:, 1],
                c=[speaker_colors[speaker]],
                label=speaker,
                alpha=0.6,
                s=50
            )

        plt.xlabel(f'PC1 ({results["explained_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({results["explained_variance"][1]:.1%} variance)')
        plt.title(
            f'Speaker Clustering - Layer {layer}\n'
            f'{self.condition.capitalize()} | {self.transcript_id} | {self.model_name.split("/")[-1]}\n'
            f'Silhouette: {results["silhouette_score"]:.3f}'
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def plot_3d_clusters(self, results: Dict, output_path: str):
        """Create 3D scatter plot of speaker clusters."""
        if results['components'].shape[1] < 3:
            print("  Skipping 3D plot (need 3 components)")
            return

        components = results['components']
        labels = results['labels']
        layer = results['layer']

        # Set up color palette
        palette = sns.color_palette("husl", len(self.speakers))
        speaker_colors = {s: palette[i] for i, s in enumerate(self.speakers)}

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for speaker in self.speakers:
            mask = [l == speaker for l in labels]
            speaker_points = components[mask]

            ax.scatter(
                speaker_points[:, 0],
                speaker_points[:, 1],
                speaker_points[:, 2],
                c=[speaker_colors[speaker]],
                label=speaker,
                alpha=0.6,
                s=50
            )

        ax.set_xlabel(f'PC1 ({results["explained_variance"][0]:.1%})')
        ax.set_ylabel(f'PC2 ({results["explained_variance"][1]:.1%})')
        ax.set_zlabel(f'PC3 ({results["explained_variance"][2]:.1%})')
        ax.set_title(
            f'Speaker Clustering - Layer {layer}\n'
            f'{self.condition.capitalize()} | {self.transcript_id} | {self.model_name.split("/")[-1]}\n'
            f'Silhouette: {results["silhouette_score"]:.3f}'
        )
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def analyze_all_layers(self, output_dir: str, n_components: int = 3):
        """Analyze clustering across all layers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []

        for layer in self.activation_data.layers:
            # Analyze this layer
            results = self.analyze_layer(layer, n_components=n_components)

            # Save plots
            output_prefix = (
                f"{self.transcript_id}_{self.condition}_"
                f"{self.model_name.split('/')[-1]}_layer{layer}"
            )

            # 2D plot
            plot_2d_path = output_path / f"{output_prefix}_2d.png"
            self.plot_2d_clusters(results, str(plot_2d_path))

            # 3D plot
            if n_components >= 3:
                plot_3d_path = output_path / f"{output_prefix}_3d.png"
                self.plot_3d_clusters(results, str(plot_3d_path))

            # Save metrics (without components to reduce size)
            # Convert numpy types to Python types for JSON serialization
            metrics = {
                'layer': int(results['layer']),
                'explained_variance': results['explained_variance'],
                'total_variance': results['total_variance'],
                'silhouette_score': results['silhouette_score'],
                'davies_bouldin_score': results['davies_bouldin_score'],
                'n_samples': int(results['n_samples']),
                'speakers': results['speakers']
            }
            all_results.append(metrics)

        # Save summary JSON
        summary_path = output_path / f"{self.transcript_id}_{self.condition}_clustering_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'transcript_id': self.transcript_id,
                'condition': self.condition,
                'model': self.model_name,
                'layers': all_results
            }, f, indent=2)

        print(f"\n✓ Saved summary: {summary_path}")

        # Plot metrics across layers
        self._plot_metrics_across_layers(all_results, str(output_path))

        return all_results

    def _plot_metrics_across_layers(self, all_results: List[Dict], output_dir: str):
        """Plot clustering metrics across layers."""
        layers = [r['layer'] for r in all_results]
        silhouette = [r['silhouette_score'] for r in all_results]
        davies_bouldin = [r['davies_bouldin_score'] for r in all_results]
        variance = [r['total_variance'] for r in all_results]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Silhouette score
        axes[0].plot(layers, silhouette, marker='o', linewidth=2)
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Speaker Clustering Quality Across Layers')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Davies-Bouldin score
        axes[1].plot(layers, davies_bouldin, marker='o', linewidth=2, color='orange')
        axes[1].set_ylabel('Davies-Bouldin Score')
        axes[1].grid(True, alpha=0.3)

        # Variance explained
        axes[2].plot(layers, variance, marker='o', linewidth=2, color='green')
        axes[2].set_ylabel('Total Variance Explained')
        axes[2].set_xlabel('Layer')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f'{self.transcript_id} | {self.condition} | {self.model_name.split("/")[-1]}',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()

        output_path = Path(output_dir) / f"{self.transcript_id}_{self.condition}_metrics_across_layers.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze speaker clustering in activation space")
    parser.add_argument('--activations', required=True,
                       help='Path to HDF5 activation file')
    parser.add_argument('--layers', type=int, nargs='+',
                       help='Specific layers to analyze (default: all)')
    parser.add_argument('--components', type=int, default=3,
                       choices=[2, 3], help='Number of PCA components')
    parser.add_argument('--output', required=True,
                       help='Output directory for plots and results')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SpeakerClusteringAnalyzer(args.activations)

    # Override layers if specified
    if args.layers:
        analyzer.activation_data.layers = args.layers

    # Run analysis
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: SPEAKER CLUSTERING ANALYSIS")
    print(f"{'='*70}")

    results = analyzer.analyze_all_layers(
        output_dir=args.output,
        n_components=args.components
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
