#!/usr/bin/env python3
"""
GFFCompare Analysis for TransGenic AS Evaluation

This script performs transcript-level accuracy analysis using GFFCompare.
Calculates isoform recovery rate (recall/precision) metrics.

Metrics calculated:
- Isoform Recall = (# exact matched isoforms) / (total reference isoforms)
- Isoform Precision = (# exact matched isoforms) / (total predicted isoforms)

Reference: GFFCompare class codes
- '=' : Complete, exact match of intron chain
- 'c' : Contained within reference (partial)
- 'k' : Containment of reference
- 'j' : Multi-exon with at least one junction match
- 'e' : Single exon transfrag overlapping reference
- 'o' : Generic exonic overlap
- 'i' : Intronic
- 'x' : Exonic overlap on opposite strand
- 'p' : Possible polymerase run-on
- 'u' : Unknown/intergenic
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re
import click
from rich.console import Console
from rich.table import Table
import json

console = Console()


@dataclass
class GFFCompareResults:
    """Container for GFFCompare analysis results."""
    # Raw counts by class code
    class_code_counts: dict
    # Transcript-level metrics
    total_reference: int
    total_predicted: int
    exact_matches: int  # class code '='
    partial_matches: int  # class codes 'c', 'j', 'k'
    # Calculated metrics
    isoform_recall: float
    isoform_precision: float
    isoform_f1: float
    # Gene-level metrics
    genes_with_exact_match: int
    genes_total: int
    gene_coverage: float


def run_gffcompare(
    reference_gtf: Path,
    predicted_gtf: Path,
    output_prefix: Path,
    genome_fasta: Optional[Path] = None
) -> Path:
    """
    Run GFFCompare to compare predicted transcripts against reference.

    Args:
        reference_gtf: Path to reference annotation (e.g., AtRTD3)
        predicted_gtf: Path to TransGenic predictions
        output_prefix: Output prefix for GFFCompare results
        genome_fasta: Optional genome FASTA for sequence-level comparison

    Returns:
        Path to the tracking file (.tracking)
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gffcompare",
        "-r", str(reference_gtf),
        "-o", str(output_prefix),
        "-T",  # Do not generate .tmap and .refmap files (use tracking instead)
    ]

    if genome_fasta:
        cmd.extend(["-s", str(genome_fasta)])

    cmd.append(str(predicted_gtf))

    console.print(f"[bold blue]Running GFFCompare...[/bold blue]")
    console.print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]GFFCompare error:[/red] {result.stderr}")
        raise RuntimeError(f"GFFCompare failed: {result.stderr}")

    tracking_file = Path(f"{output_prefix}.tracking")
    if not tracking_file.exists():
        # Try alternative output
        tracking_file = Path(f"{output_prefix}.{predicted_gtf.stem}.tmap")

    return tracking_file


def parse_tmap_file(tmap_path: Path) -> pd.DataFrame:
    """
    Parse GFFCompare .tmap file.

    The .tmap file contains per-transcript comparison results:
    - ref_gene_id: Reference gene ID
    - ref_id: Reference transcript ID
    - class_code: Match classification
    - qry_gene_id: Query gene ID
    - qry_id: Query transcript ID
    - num_exons: Number of exons
    - FPKM: Expression level (if available)
    - TPM: Expression level (if available)
    - coverage: Coverage
    - len: Length
    - major_iso_id: Major isoform ID
    - ref_match_len: Reference match length
    """
    columns = [
        'ref_gene_id', 'ref_id', 'class_code', 'qry_gene_id', 'qry_id',
        'num_exons', 'FPKM', 'TPM', 'coverage', 'len', 'major_iso_id', 'ref_match_len'
    ]

    try:
        df = pd.read_csv(tmap_path, sep='\t', comment='#', header=None, names=columns)
    except Exception:
        # Try with fewer columns (older gffcompare versions)
        df = pd.read_csv(tmap_path, sep='\t', comment='#', header=0)

    return df


def parse_stats_file(stats_path: Path) -> dict:
    """
    Parse GFFCompare .stats file for summary statistics.
    """
    stats = {}

    with open(stats_path) as f:
        content = f.read()

    # Extract key metrics using regex
    patterns = {
        'sensitivity': r'Sensitivity\s*\|\s*([\d.]+)',
        'precision': r'Precision\s*\|\s*([\d.]+)',
        'base_sensitivity': r'Base level:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        'exon_sensitivity': r'Exon level:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        'intron_sensitivity': r'Intron level:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        'transcript_sensitivity': r'Transcript level:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            stats[key] = [float(g) for g in match.groups()]

    return stats


def calculate_metrics(tmap_df: pd.DataFrame) -> GFFCompareResults:
    """
    Calculate isoform-level metrics from GFFCompare results.

    Key class codes for exact/partial matching:
    - '=' : Exact match (all introns identical)
    - 'c' : Query contained in reference
    - 'k' : Reference contained in query
    - 'j' : At least one splice junction shared
    """
    class_code_counts = tmap_df['class_code'].value_counts().to_dict()

    # Count transcripts
    total_predicted = len(tmap_df)

    # Exact matches (class code '=')
    exact_matches = class_code_counts.get('=', 0)

    # Partial matches (structural similarity)
    partial_codes = ['c', 'k', 'j']
    partial_matches = sum(class_code_counts.get(c, 0) for c in partial_codes)

    # Reference transcripts (unique ref_ids, excluding '-' for novel)
    ref_transcripts = tmap_df[tmap_df['ref_id'] != '-']['ref_id'].nunique()
    total_reference = ref_transcripts if ref_transcripts > 0 else tmap_df['ref_id'].nunique()

    # Calculate precision/recall
    # Note: For recall, we need to know total reference transcripts
    # This might require parsing the reference GTF separately
    isoform_precision = exact_matches / total_predicted if total_predicted > 0 else 0
    isoform_recall = exact_matches / total_reference if total_reference > 0 else 0

    # F1 score
    if isoform_precision + isoform_recall > 0:
        isoform_f1 = 2 * (isoform_precision * isoform_recall) / (isoform_precision + isoform_recall)
    else:
        isoform_f1 = 0

    # Gene-level analysis
    genes_with_exact = tmap_df[tmap_df['class_code'] == '=']['ref_gene_id'].nunique()
    genes_total = tmap_df['ref_gene_id'].nunique()
    gene_coverage = genes_with_exact / genes_total if genes_total > 0 else 0

    return GFFCompareResults(
        class_code_counts=class_code_counts,
        total_reference=total_reference,
        total_predicted=total_predicted,
        exact_matches=exact_matches,
        partial_matches=partial_matches,
        isoform_recall=isoform_recall,
        isoform_precision=isoform_precision,
        isoform_f1=isoform_f1,
        genes_with_exact_match=genes_with_exact,
        genes_total=genes_total,
        gene_coverage=gene_coverage
    )


def count_reference_transcripts(reference_gtf: Path) -> int:
    """Count total transcripts in reference GTF."""
    count = 0
    with open(reference_gtf) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 3 and fields[2] == 'transcript':
                count += 1
    return count


def analyze_isoform_distribution(
    reference_gtf: Path,
    predicted_gtf: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Analyze isoform count distribution per gene.

    Returns DataFrame with:
    - gene_id
    - ref_isoform_count
    - pred_isoform_count
    - difference
    """
    def count_isoforms_per_gene(gtf_path: Path) -> dict:
        """Count transcripts per gene from GTF."""
        gene_counts = {}
        with open(gtf_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                if fields[2] != 'transcript':
                    continue

                # Parse attributes
                attrs = fields[8]
                gene_id_match = re.search(r'gene_id\s*"([^"]+)"', attrs)
                if gene_id_match:
                    gene_id = gene_id_match.group(1)
                    gene_counts[gene_id] = gene_counts.get(gene_id, 0) + 1

        return gene_counts

    ref_counts = count_isoforms_per_gene(reference_gtf)
    pred_counts = count_isoforms_per_gene(predicted_gtf)

    # Combine into DataFrame
    all_genes = set(ref_counts.keys()) | set(pred_counts.keys())

    data = []
    for gene_id in all_genes:
        ref_count = ref_counts.get(gene_id, 0)
        pred_count = pred_counts.get(gene_id, 0)
        data.append({
            'gene_id': gene_id,
            'ref_isoform_count': ref_count,
            'pred_isoform_count': pred_count,
            'difference': pred_count - ref_count,
            'category': categorize_prediction(ref_count, pred_count)
        })

    df = pd.DataFrame(data)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'isoform_distribution.csv', index=False)

    return df


def categorize_prediction(ref_count: int, pred_count: int) -> str:
    """Categorize prediction accuracy."""
    if ref_count == 0 and pred_count > 0:
        return 'novel_gene'
    elif pred_count == 0 and ref_count > 0:
        return 'missed_gene'
    elif pred_count == ref_count:
        return 'exact'
    elif pred_count > ref_count:
        return 'overprediction'
    else:
        return 'underprediction'


def generate_summary_report(
    results: GFFCompareResults,
    isoform_dist: pd.DataFrame,
    output_path: Path
):
    """Generate a comprehensive summary report."""

    report = {
        'transcript_level_metrics': {
            'isoform_recall': results.isoform_recall,
            'isoform_precision': results.isoform_precision,
            'isoform_f1': results.isoform_f1,
            'exact_matches': results.exact_matches,
            'partial_matches': results.partial_matches,
            'total_reference': results.total_reference,
            'total_predicted': results.total_predicted,
        },
        'gene_level_metrics': {
            'genes_with_exact_match': results.genes_with_exact_match,
            'genes_total': results.genes_total,
            'gene_coverage': results.gene_coverage,
        },
        'class_code_distribution': results.class_code_counts,
        'isoform_count_analysis': {
            'mean_ref_isoforms': isoform_dist['ref_isoform_count'].mean(),
            'mean_pred_isoforms': isoform_dist['pred_isoform_count'].mean(),
            'exact_count_matches': len(isoform_dist[isoform_dist['category'] == 'exact']),
            'overpredictions': len(isoform_dist[isoform_dist['category'] == 'overprediction']),
            'underpredictions': len(isoform_dist[isoform_dist['category'] == 'underprediction']),
            'missed_genes': len(isoform_dist[isoform_dist['category'] == 'missed_gene']),
            'novel_genes': len(isoform_dist[isoform_dist['category'] == 'novel_gene']),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def print_results_table(results: GFFCompareResults, isoform_dist: pd.DataFrame):
    """Print formatted results table."""

    # Transcript-level metrics
    table1 = Table(title="Transcript-level Metrics")
    table1.add_column("Metric", style="cyan")
    table1.add_column("Value", style="green")

    table1.add_row("Isoform Recall", f"{results.isoform_recall:.4f}")
    table1.add_row("Isoform Precision", f"{results.isoform_precision:.4f}")
    table1.add_row("Isoform F1", f"{results.isoform_f1:.4f}")
    table1.add_row("Exact Matches", str(results.exact_matches))
    table1.add_row("Partial Matches", str(results.partial_matches))
    table1.add_row("Total Reference", str(results.total_reference))
    table1.add_row("Total Predicted", str(results.total_predicted))

    console.print(table1)
    console.print()

    # Class code distribution
    table2 = Table(title="Class Code Distribution")
    table2.add_column("Code", style="cyan")
    table2.add_column("Count", style="green")
    table2.add_column("Description", style="yellow")

    code_descriptions = {
        '=': 'Exact intron chain match',
        'c': 'Contained in reference',
        'k': 'Contains reference',
        'j': 'Multi-exon, junction match',
        'e': 'Single exon overlap',
        'o': 'Generic overlap',
        'i': 'Intronic',
        'x': 'Opposite strand',
        'p': 'Polymerase run-on',
        'u': 'Intergenic/unknown',
    }

    for code, count in sorted(results.class_code_counts.items(), key=lambda x: -x[1]):
        desc = code_descriptions.get(code, 'Other')
        table2.add_row(code, str(count), desc)

    console.print(table2)
    console.print()

    # Isoform distribution summary
    table3 = Table(title="Isoform Count Distribution")
    table3.add_column("Category", style="cyan")
    table3.add_column("Count", style="green")

    category_counts = isoform_dist['category'].value_counts()
    for cat, count in category_counts.items():
        table3.add_row(cat, str(count))

    console.print(table3)


@click.command()
@click.option('--reference', '-r', type=click.Path(exists=True), required=True,
              help='Reference GTF file (e.g., AtRTD3)')
@click.option('--predicted', '-p', type=click.Path(exists=True), required=True,
              help='Predicted GTF file (TransGenic output)')
@click.option('--output-dir', '-o', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--genome', '-g', type=click.Path(exists=True), default=None,
              help='Optional genome FASTA for sequence comparison')
@click.option('--prefix', default='transgenic_vs_ref',
              help='Output file prefix')
def main(reference: str, predicted: str, output_dir: str, genome: str, prefix: str):
    """
    Run GFFCompare analysis for TransGenic AS evaluation.

    Calculates transcript-level accuracy metrics:
    - Isoform recall/precision/F1
    - Class code distribution
    - Isoform count distribution per gene
    """
    reference_path = Path(reference)
    predicted_path = Path(predicted)
    output_path = Path(output_dir)
    genome_path = Path(genome) if genome else None

    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]GFFCompare Analysis for TransGenic[/bold]")
    console.print(f"Reference: {reference_path}")
    console.print(f"Predicted: {predicted_path}")
    console.print(f"Output: {output_path}")
    console.print()

    # Step 1: Run GFFCompare
    gffcompare_prefix = output_path / prefix

    try:
        run_gffcompare(reference_path, predicted_path, gffcompare_prefix, genome_path)
    except Exception as e:
        console.print(f"[red]Error running GFFCompare: {e}[/red]")
        raise

    # Step 2: Parse results
    tmap_file = output_path / f"{prefix}.{predicted_path.stem}.tmap"
    if not tmap_file.exists():
        # Try alternative naming
        tmap_files = list(output_path.glob("*.tmap"))
        if tmap_files:
            tmap_file = tmap_files[0]
        else:
            console.print(f"[red]No .tmap file found in {output_path}[/red]")
            raise FileNotFoundError("GFFCompare .tmap output not found")

    console.print(f"[blue]Parsing results from {tmap_file}[/blue]")
    tmap_df = parse_tmap_file(tmap_file)

    # Step 3: Calculate metrics
    # Get actual reference transcript count
    ref_transcript_count = count_reference_transcripts(reference_path)
    console.print(f"[blue]Reference transcripts: {ref_transcript_count}[/blue]")

    results = calculate_metrics(tmap_df)
    # Update with actual reference count
    results.total_reference = ref_transcript_count
    results.isoform_recall = results.exact_matches / ref_transcript_count if ref_transcript_count > 0 else 0
    if results.isoform_precision + results.isoform_recall > 0:
        results.isoform_f1 = 2 * (results.isoform_precision * results.isoform_recall) / (results.isoform_precision + results.isoform_recall)

    # Step 4: Analyze isoform distribution
    console.print("[blue]Analyzing isoform distribution...[/blue]")
    isoform_dist = analyze_isoform_distribution(reference_path, predicted_path, output_path)

    # Step 5: Generate reports
    print_results_table(results, isoform_dist)

    report = generate_summary_report(results, isoform_dist, output_path / 'summary_report.json')

    # Save tmap DataFrame
    tmap_df.to_csv(output_path / 'parsed_tmap.csv', index=False)

    console.print()
    console.print(f"[bold green]Analysis complete![/bold green]")
    console.print(f"Results saved to: {output_path}")
    console.print()
    console.print("[bold]Key findings:[/bold]")
    console.print(f"  Isoform Recall: {results.isoform_recall:.2%}")
    console.print(f"  Isoform Precision: {results.isoform_precision:.2%}")
    console.print(f"  Isoform F1: {results.isoform_f1:.2%}")


if __name__ == '__main__':
    main()
