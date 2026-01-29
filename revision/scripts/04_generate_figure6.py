#!/usr/bin/env python3
"""
Generate Figure 6: Alternative Splicing Evaluation

Multi-panel figure for TransGenic AS evaluation:
A: Transcript-level precision/recall bar plot
B: Isoform count distribution (Reference vs TransGenic)
C: Splice event recovery by type
D: Example gene schematic (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import click
from typing import Optional

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_gffcompare_results(results_dir: Path) -> dict:
    """Load GFFCompare analysis results."""
    summary_path = results_dir / 'summary_report.json'
    with open(summary_path) as f:
        return json.load(f)


def load_splice_event_results(results_dir: Path, prefix: str = 'splice_events') -> dict:
    """Load splice event analysis results."""
    report_path = results_dir / f'{prefix}_report.json'
    with open(report_path) as f:
        return json.load(f)


def load_isoform_distribution(results_dir: Path) -> pd.DataFrame:
    """Load isoform distribution data."""
    csv_path = results_dir / 'isoform_distribution.csv'
    return pd.read_csv(csv_path)


def plot_panel_a(ax, gffcompare_results: dict):
    """
    Panel A: Transcript-level precision/recall bar plot
    """
    metrics = gffcompare_results['transcript_level_metrics']

    categories = ['Recall', 'Precision', 'F1 Score']
    values = [
        metrics['isoform_recall'],
        metrics['isoform_precision'],
        metrics['isoform_f1']
    ]

    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('A. Transcript-level Accuracy', fontsize=14, fontweight='bold', loc='left')
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add sample size annotation
    n_exact = metrics['exact_matches']
    n_ref = metrics['total_reference']
    n_pred = metrics['total_predicted']
    ax.text(0.98, 0.95, f'Exact matches: {n_exact:,}\nRef: {n_ref:,} | Pred: {n_pred:,}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_panel_b(ax, isoform_dist: pd.DataFrame):
    """
    Panel B: Isoform count distribution (violin/box plot)
    """
    # Prepare data for plotting
    ref_counts = isoform_dist['ref_isoform_count']
    pred_counts = isoform_dist['pred_isoform_count']

    # Cap at 10 for better visualization
    max_display = 10
    ref_capped = ref_counts.clip(upper=max_display)
    pred_capped = pred_counts.clip(upper=max_display)

    data = pd.DataFrame({
        'Isoforms per Gene': list(ref_capped) + list(pred_capped),
        'Source': ['Reference'] * len(ref_capped) + ['TransGenic'] * len(pred_capped)
    })

    # Create violin plot
    parts = ax.violinplot([ref_capped, pred_capped], positions=[1, 2], showmeans=True, showmedians=True)

    colors = ['#3498db', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Reference', 'TransGenic'])
    ax.set_ylabel('Isoforms per Gene', fontsize=12)
    ax.set_title('B. Isoform Count Distribution', fontsize=14, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add statistics
    stats_text = (f'Reference: μ={ref_counts.mean():.2f}\n'
                  f'TransGenic: μ={pred_counts.mean():.2f}')
    ax.text(0.98, 0.95, stats_text,
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_panel_c(ax, splice_results: dict):
    """
    Panel C: Splice event recovery by type (grouped bar chart)
    """
    event_types = ['SE', 'A5SS', 'A3SS', 'IR']
    event_labels = ['Exon\nSkipping', 'Alt 5\'SS', 'Alt 3\'SS', 'Intron\nRetention']

    recalls = []
    precisions = []

    for et in event_types:
        data = splice_results['per_event_type'][et]
        recalls.append(data['recall'])
        precisions.append(data['precision'])

    x = np.arange(len(event_types))
    width = 0.35

    bars1 = ax.bar(x - width/2, recalls, width, label='Recall', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, precisions, width, label='Precision', color='#3498db', edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('C. Splice Event Recovery by Type', fontsize=14, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(event_labels)
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.0%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)


def plot_panel_d(ax, gffcompare_results: dict):
    """
    Panel D: Class code distribution (pie chart or stacked bar)
    """
    class_codes = gffcompare_results.get('class_code_distribution', {})

    if not class_codes:
        ax.text(0.5, 0.5, 'No class code data', ha='center', va='center')
        return

    # Group into categories
    exact = class_codes.get('=', 0)
    partial = sum(class_codes.get(c, 0) for c in ['c', 'k', 'j'])
    overlap = sum(class_codes.get(c, 0) for c in ['e', 'o'])
    other = sum(class_codes.get(c, 0) for c in ['i', 'x', 'p', 'u'])

    categories = ['Exact Match\n(=)', 'Partial Match\n(c,k,j)', 'Overlap\n(e,o)', 'Other\n(i,x,p,u)']
    values = [exact, partial, overlap, other]
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#95a5a6']

    # Filter out zero values
    non_zero = [(c, v, col) for c, v, col in zip(categories, values, colors) if v > 0]
    if non_zero:
        categories, values, colors = zip(*non_zero)

    wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=[0.05] * len(values))

    ax.set_title('D. Transcript Classification', fontsize=14, fontweight='bold', loc='left')

    # Style the text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')


def create_figure6(
    gffcompare_dir: Path,
    splice_event_dir: Path,
    output_path: Path,
    figsize: tuple = (14, 12)
):
    """
    Create the complete Figure 6.
    """
    # Load data
    gffcompare_results = load_gffcompare_results(gffcompare_dir)
    splice_results = load_splice_event_results(splice_event_dir)
    isoform_dist = load_isoform_distribution(gffcompare_dir)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Plot each panel
    plot_panel_a(ax_a, gffcompare_results)
    plot_panel_b(ax_b, isoform_dist)
    plot_panel_c(ax_c, splice_results)
    plot_panel_d(ax_d, gffcompare_results)

    # Add overall title
    fig.suptitle('Figure 6. Alternative Splicing Evaluation of TransGenic',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'),
                   dpi=300, bbox_inches='tight', facecolor='white')

    plt.close(fig)

    print(f"Figure saved to: {output_path}")


@click.command()
@click.option('--gffcompare-dir', '-g', type=click.Path(exists=True), required=True,
              help='Directory with GFFCompare results')
@click.option('--splice-event-dir', '-s', type=click.Path(exists=True), required=True,
              help='Directory with splice event results')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output path for figure (without extension)')
@click.option('--width', default=14, help='Figure width in inches')
@click.option('--height', default=12, help='Figure height in inches')
def main(gffcompare_dir: str, splice_event_dir: str, output: str, width: int, height: int):
    """
    Generate Figure 6: Alternative Splicing Evaluation.

    Creates a multi-panel figure showing:
    A. Transcript-level precision/recall
    B. Isoform count distribution
    C. Splice event recovery by type
    D. Transcript classification breakdown
    """
    gffcompare_path = Path(gffcompare_dir)
    splice_path = Path(splice_event_dir)
    output_path = Path(output)

    create_figure6(
        gffcompare_dir=gffcompare_path,
        splice_event_dir=splice_path,
        output_path=output_path,
        figsize=(width, height)
    )

    print("Figure 6 generated successfully!")


if __name__ == '__main__':
    main()
