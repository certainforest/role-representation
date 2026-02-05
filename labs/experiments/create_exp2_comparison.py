"""
Experiment 2 Comparison Dashboard

Compares linear probe results across conditions and creates
comprehensive visualizations showing which layers best encode speaker roles.
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

def create_exp2_comparison_dashboard(labeled_summary, unlabeled_summary, corrupted_summary, output_dir):
    """Create comprehensive comparison visualizations for Experiment 2."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CREATING EXPERIMENT 2 COMPARISON DASHBOARD")
    print(f"{'='*70}\n")

    # Extract data
    conditions = {
        'Labeled': labeled_summary,
        'Unlabeled': unlabeled_summary,
        'Corrupted': corrupted_summary
    }

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {'Labeled': '#2ecc71', 'Unlabeled': '#3498db', 'Corrupted': '#e74c3c'}

    # 1. Probe accuracy across layers
    ax1 = fig.add_subplot(gs[0, :])

    for condition_name, summary in conditions.items():
        layers = [layer['layer'] for layer in summary['layers']]
        accuracies = [layer['best_test_acc'] for layer in summary['layers']]
        ax1.plot(layers, accuracies, marker='o', linewidth=2.5, markersize=8,
                label=condition_name, color=colors[condition_name], alpha=0.8)

    # Add random baseline
    random_baseline = labeled_summary['layers'][0]['random_baseline']
    ax1.axhline(y=random_baseline, color='gray', linestyle='--',
               linewidth=2, label=f'Random Baseline ({random_baseline:.1%})', alpha=0.7)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Probe Classification Accuracy Across Layers\n' +
                  'Can we decode speaker role from activations?',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.4, 1.0)

    # 2. Improvement over random
    ax2 = fig.add_subplot(gs[1, :])

    for condition_name, summary in conditions.items():
        layers = [layer['layer'] for layer in summary['layers']]
        improvements = [layer['improvement'] for layer in summary['layers']]
        ax2.plot(layers, improvements, marker='s', linewidth=2.5, markersize=8,
                label=condition_name, color=colors[condition_name], alpha=0.8)

    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement over Random', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Improvement Above Chance Level\n' +
                  'Higher values = stronger role encoding',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. Peak accuracy comparison (bar chart)
    ax3 = fig.add_subplot(gs[2, 0])

    peak_accs = []
    condition_names = []
    for condition_name, summary in conditions.items():
        accuracies = [layer['best_test_acc'] for layer in summary['layers']]
        peak_accs.append(max(accuracies))
        condition_names.append(condition_name)

    bars = ax3.bar(condition_names, peak_accs,
                   color=[colors[c] for c in condition_names], alpha=0.7)
    ax3.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Peak Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Best Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)

    for bar, acc in zip(bars, peak_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Peak layer comparison
    ax4 = fig.add_subplot(gs[2, 1])

    peak_layers = []
    for condition_name, summary in conditions.items():
        accuracies = [layer['best_test_acc'] for layer in summary['layers']]
        layers = [layer['layer'] for layer in summary['layers']]
        peak_layer = layers[accuracies.index(max(accuracies))]
        peak_layers.append(peak_layer)

    bars = ax4.bar(condition_names, peak_layers,
                   color=[colors[c] for c in condition_names], alpha=0.7)
    ax4.set_ylabel('Layer Index', fontsize=11, fontweight='bold')
    ax4.set_title('Layer with Peak Accuracy', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(peak_layers) * 1.3)

    for bar, layer in zip(bars, peak_layers):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'Layer {layer}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5. Average accuracy
    ax5 = fig.add_subplot(gs[2, 2])

    avg_accs = []
    for condition_name, summary in conditions.items():
        accuracies = [layer['best_test_acc'] for layer in summary['layers']]
        avg_accs.append(np.mean(accuracies))

    bars = ax5.bar(condition_names, avg_accs,
                   color=[colors[c] for c in condition_names], alpha=0.7)
    ax5.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax5.set_title('Mean Across All Layers', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)

    for bar, acc in zip(bars, avg_accs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle('Linear Probe Analysis: Role Classification Performance\n' +
                 f'Model: {labeled_summary["model"]} | Transcript: {labeled_summary["transcript_id"]} | ' +
                 f'{labeled_summary["num_roles"]} speakers, {labeled_summary["test_samples"]} test samples',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_file = output_path / 'exp2_comprehensive_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Create detailed metrics table
    create_metrics_table(labeled_summary, unlabeled_summary, corrupted_summary, output_path)

    print(f"\n{'='*70}")
    print("EXP2 DASHBOARD COMPLETE")
    print(f"{'='*70}\n")

def create_metrics_table(labeled_summary, unlabeled_summary, corrupted_summary, output_path):
    """Create detailed comparison table."""
    print("Creating detailed metrics table...")

    conditions = {
        'Labeled': labeled_summary,
        'Unlabeled': unlabeled_summary,
        'Corrupted': corrupted_summary
    }

    markdown = "# Linear Probe Analysis - Detailed Results\n\n"
    markdown += f"**Model:** {labeled_summary['model']}  \n"
    markdown += f"**Transcript:** {labeled_summary['transcript_id']}  \n"
    markdown += f"**Num Roles:** {labeled_summary['num_roles']}  \n"
    markdown += f"**Train Samples:** {labeled_summary['train_samples']}  \n"
    markdown += f"**Test Samples:** {labeled_summary['test_samples']}  \n\n"

    markdown += "## Layer-wise Accuracy Comparison\n\n"

    layers = [layer['layer'] for layer in labeled_summary['layers']]

    markdown += "| Layer | Labeled<br>Accuracy | Unlabeled<br>Accuracy | Corrupted<br>Accuracy | Best Condition | Improvement vs Random |\n"
    markdown += "|------:|--------------------:|----------------------:|----------------------:|:---------------|----------------------:|\n"

    for layer in layers:
        labeled_acc = next(l['best_test_acc'] for l in labeled_summary['layers'] if l['layer'] == layer)
        unlabeled_acc = next(l['best_test_acc'] for l in unlabeled_summary['layers'] if l['layer'] == layer)
        corrupted_acc = next(l['best_test_acc'] for l in corrupted_summary['layers'] if l['layer'] == layer)

        scores = {'Labeled': labeled_acc, 'Unlabeled': unlabeled_acc, 'Corrupted': corrupted_acc}
        best = max(scores, key=scores.get)
        best_score = scores[best]

        random = labeled_summary['layers'][0]['random_baseline']
        improvement = best_score - random

        markdown += f"| {layer:2d}    | {labeled_acc:.3f} ({labeled_acc:.1%})   | {unlabeled_acc:.3f} ({unlabeled_acc:.1%})    | {corrupted_acc:.3f} ({corrupted_acc:.1%})    | **{best}** | +{improvement:.3f} ({improvement/random:.1%}) |\n"

    markdown += "\n## Summary Statistics\n\n"
    markdown += "| Condition | Peak Accuracy | Mean Accuracy | Peak Layer |\n"
    markdown += "|:----------|:--------------|:--------------|:-----------|\n"

    for condition_name, summary in conditions.items():
        accuracies = [l['best_test_acc'] for l in summary['layers']]
        layers = [l['layer'] for l in summary['layers']]
        peak_acc = max(accuracies)
        mean_acc = np.mean(accuracies)
        peak_layer = layers[accuracies.index(peak_acc)]

        markdown += f"| **{condition_name}** | {peak_acc:.3f} ({peak_acc:.1%}) | {mean_acc:.3f} ({mean_acc:.1%}) | Layer {peak_layer} |\n"

    # Save
    md_file = output_path / 'exp2_detailed_metrics.md'
    with open(md_file, 'w') as f:
        f.write(markdown)
    print(f"✓ Saved: {md_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', required=True)
    parser.add_argument('--unlabeled', required=True)
    parser.add_argument('--corrupted', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    labeled_summary = load_summary(args.labeled)
    unlabeled_summary = load_summary(args.unlabeled)
    corrupted_summary = load_summary(args.corrupted)

    create_exp2_comparison_dashboard(
        labeled_summary,
        unlabeled_summary,
        corrupted_summary,
        args.output
    )

if __name__ == "__main__":
    main()
