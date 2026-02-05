"""
Comprehensive Comparison Dashboard

Compares PCA clustering results across different conditions:
- Labeled vs Unlabeled vs Corrupted
- Layer-wise progression
- Side-by-side visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_summary(summary_path):
    """Load summary JSON from experiment."""
    with open(summary_path, 'r') as f:
        return json.load(f)

def create_comparison_dashboard(labeled_summary, unlabeled_summary, corrupted_summary, output_dir):
    """Create comprehensive comparison visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CREATING COMPARISON DASHBOARD")
    print(f"{'='*70}\n")

    # Extract data for each condition
    conditions = {
        'Labeled': labeled_summary,
        'Unlabeled': unlabeled_summary,
        'Corrupted': corrupted_summary
    }

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Silhouette scores across layers for all conditions
    ax1 = fig.add_subplot(gs[0, :])
    colors = {'Labeled': '#2ecc71', 'Unlabeled': '#3498db', 'Corrupted': '#e74c3c'}

    for condition_name, summary in conditions.items():
        layers = [layer['layer'] for layer in summary['layers']]
        silhouette = [layer['silhouette_score'] for layer in summary['layers']]
        ax1.plot(layers, silhouette, marker='o', linewidth=2.5, markersize=8,
                label=condition_name, color=colors[condition_name], alpha=0.8)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Speaker Clustering Quality Across Layers\n(Higher is better)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_ylim(bottom=-0.05)

    # 2. Davies-Bouldin scores across layers
    ax2 = fig.add_subplot(gs[1, :])

    for condition_name, summary in conditions.items():
        layers = [layer['layer'] for layer in summary['layers']]
        davies_bouldin = [layer['davies_bouldin_score'] for layer in summary['layers']]
        ax2.plot(layers, davies_bouldin, marker='s', linewidth=2.5, markersize=8,
                label=condition_name, color=colors[condition_name], alpha=0.8)

    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Davies-Bouldin Score', fontsize=12, fontweight='bold')
    ax2.set_title('Cluster Separation Quality Across Layers\n(Lower is better)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. Bar chart comparing peak layer performance
    ax3 = fig.add_subplot(gs[2, 0])

    peak_scores = []
    condition_names = []
    for condition_name, summary in conditions.items():
        silhouette_scores = [layer['silhouette_score'] for layer in summary['layers']]
        peak_scores.append(max(silhouette_scores))
        condition_names.append(condition_name)

    bars = ax3.bar(condition_names, peak_scores, color=[colors[c] for c in condition_names], alpha=0.7)
    ax3.set_ylabel('Peak Silhouette Score', fontsize=11, fontweight='bold')
    ax3.set_title('Best Clustering Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(peak_scores) * 1.2)

    # Add value labels on bars
    for bar, score in zip(bars, peak_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Peak layer comparison
    ax4 = fig.add_subplot(gs[2, 1])

    peak_layers = []
    for condition_name, summary in conditions.items():
        silhouette_scores = [layer['silhouette_score'] for layer in summary['layers']]
        layers = [layer['layer'] for layer in summary['layers']]
        peak_layer = layers[silhouette_scores.index(max(silhouette_scores))]
        peak_layers.append(peak_layer)

    bars = ax4.bar(condition_names, peak_layers, color=[colors[c] for c in condition_names], alpha=0.7)
    ax4.set_ylabel('Layer Index', fontsize=11, fontweight='bold')
    ax4.set_title('Layer with Peak Performance', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(peak_layers) * 1.3)

    # Add value labels
    for bar, layer in zip(bars, peak_layers):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'Layer {layer}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5. Variance explained comparison
    ax5 = fig.add_subplot(gs[2, 2])

    avg_variance = []
    for condition_name, summary in conditions.items():
        variances = [layer['total_variance'] for layer in summary['layers']]
        avg_variance.append(np.mean(variances))

    bars = ax5.bar(condition_names, avg_variance, color=[colors[c] for c in condition_names], alpha=0.7)
    ax5.set_ylabel('Avg Total Variance', fontsize=11, fontweight='bold')
    ax5.set_title('Average PCA Variance Explained', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)

    # Add value labels
    for bar, variance in zip(bars, avg_variance):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{variance:.2%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle('Speaker Role Binding Analysis: Comprehensive Comparison\n' +
                 f'Model: {labeled_summary["model"]} | Transcript: {labeled_summary["transcript_id"]}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_file = output_path / 'comprehensive_comparison_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Create side-by-side PCA plots for key layer
    create_side_by_side_pca(labeled_summary, unlabeled_summary, corrupted_summary, output_path)

    # Create detailed metrics table
    create_metrics_table(labeled_summary, unlabeled_summary, corrupted_summary, output_path)

    print(f"\n{'='*70}")
    print("DASHBOARD CREATION COMPLETE")
    print(f"{'='*70}\n")

def create_side_by_side_pca(labeled_summary, unlabeled_summary, corrupted_summary, output_path):
    """Create side-by-side PCA scatter plots for peak layer."""

    # Find peak layer (use labeled as reference)
    silhouette_scores = [layer['silhouette_score'] for layer in labeled_summary['layers']]
    layers = [layer['layer'] for layer in labeled_summary['layers']]
    peak_layer = layers[silhouette_scores.index(max(silhouette_scores))]

    print(f"Creating side-by-side PCA plots for peak layer {peak_layer}...")

    # Note: Since we don't have the actual PCA components here, we'll load the pre-generated images
    # and arrange them side by side

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    conditions = ['labeled', 'unlabeled', 'corrupted']
    titles = ['Labeled (Ground Truth)', 'Unlabeled (No Speaker Info)', 'Corrupted (30% Mislabeled)']

    for ax, condition, title in zip(axes, conditions, titles):
        # Load the corresponding 2D PCA image
        img_path = output_path.parent / f'exp1_{condition}' / f'1_{condition}_gpt2_layer{peak_layer}_2d.png'

        if img_path.exists():
            import matplotlib.image as mpimg
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, f'Image not found:\n{img_path.name}',
                   ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold')

    fig.suptitle(f'Speaker Clustering at Layer {peak_layer} (Peak Performance)\n' +
                 'Comparing different labeling conditions',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_file = output_path / f'side_by_side_pca_layer{peak_layer}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_metrics_table(labeled_summary, unlabeled_summary, corrupted_summary, output_path):
    """Create detailed comparison table."""

    print("Creating detailed metrics table...")

    conditions = {
        'Labeled': labeled_summary,
        'Unlabeled': unlabeled_summary,
        'Corrupted': corrupted_summary
    }

    # Create markdown table
    markdown = "# Speaker Role Binding Analysis - Detailed Results\n\n"
    markdown += f"**Model:** {labeled_summary['model']}  \n"
    markdown += f"**Transcript:** {labeled_summary['transcript_id']}  \n\n"

    markdown += "## Layer-wise Metrics Comparison\n\n"

    # Get all layers
    layers = [layer['layer'] for layer in labeled_summary['layers']]

    # Create table
    markdown += "| Layer | Labeled<br>Silhouette | Unlabeled<br>Silhouette | Corrupted<br>Silhouette | Best Condition |\n"
    markdown += "|------:|----------------------:|------------------------:|------------------------:|:---------------|\n"

    for layer in layers:
        labeled_sil = next(l['silhouette_score'] for l in labeled_summary['layers'] if l['layer'] == layer)
        unlabeled_sil = next(l['silhouette_score'] for l in unlabeled_summary['layers'] if l['layer'] == layer)
        corrupted_sil = next(l['silhouette_score'] for l in corrupted_summary['layers'] if l['layer'] == layer)

        scores = {'Labeled': labeled_sil, 'Unlabeled': unlabeled_sil, 'Corrupted': corrupted_sil}
        best = max(scores, key=scores.get)

        markdown += f"| {layer:2d}    | {labeled_sil:.3f}                | {unlabeled_sil:.3f}                   | {corrupted_sil:.3f}                   | **{best}** |\n"

    # Summary statistics
    markdown += "\n## Summary Statistics\n\n"
    markdown += "| Metric | Labeled | Unlabeled | Corrupted |\n"
    markdown += "|:-------|--------:|----------:|----------:|\n"

    for condition_name, summary in conditions.items():
        silhouettes = [l['silhouette_score'] for l in summary['layers']]
        markdown += f"| **{condition_name}** | | | |\n"
        markdown += f"| Peak Silhouette | {max(silhouettes):.3f} | | |\n"
        markdown += f"| Mean Silhouette | {np.mean(silhouettes):.3f} | | |\n"
        markdown += f"| Std Silhouette | {np.std(silhouettes):.3f} | | |\n"

    # Save markdown
    md_file = output_path / 'detailed_metrics_comparison.md'
    with open(md_file, 'w') as f:
        f.write(markdown)
    print(f"✓ Saved: {md_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create comprehensive comparison dashboard")
    parser.add_argument('--labeled', required=True, help='Path to labeled summary JSON')
    parser.add_argument('--unlabeled', required=True, help='Path to unlabeled summary JSON')
    parser.add_argument('--corrupted', required=True, help='Path to corrupted summary JSON')
    parser.add_argument('--output', required=True, help='Output directory for dashboard')

    args = parser.parse_args()

    # Load summaries
    print("Loading experiment summaries...")
    labeled_summary = load_summary(args.labeled)
    unlabeled_summary = load_summary(args.unlabeled)
    corrupted_summary = load_summary(args.corrupted)

    # Create dashboard
    create_comparison_dashboard(
        labeled_summary,
        unlabeled_summary,
        corrupted_summary,
        args.output
    )

if __name__ == "__main__":
    main()
