#!/usr/bin/env python3
"""
Splice Event Detection and Evaluation for TransGenic

Detects and compares alternative splicing events between reference and predictions.

Event types (rMATS-style definitions):
- SE (Skipped Exon / Exon Skipping)
- A5SS (Alternative 5' Splice Site)
- A3SS (Alternative 3' Splice Site)
- IR (Intron Retention)
- MXE (Mutually Exclusive Exons) - optional

Output metrics per event type:
- Event Recall = (# reference events recovered) / (# total reference events)
- Event Precision = (# correctly predicted events) / (# total predicted events)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import re
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import json

console = Console()


@dataclass
class Exon:
    """Represents an exon with genomic coordinates."""
    chrom: str
    start: int  # 0-based
    end: int    # 0-based, exclusive
    strand: str

    def __hash__(self):
        return hash((self.chrom, self.start, self.end, self.strand))

    def __eq__(self, other):
        return (self.chrom == other.chrom and
                self.start == other.start and
                self.end == other.end and
                self.strand == other.strand)

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Intron:
    """Represents an intron (splice junction)."""
    chrom: str
    start: int  # 0-based, intron start
    end: int    # 0-based, intron end (exclusive)
    strand: str

    def __hash__(self):
        return hash((self.chrom, self.start, self.end, self.strand))

    def __eq__(self, other):
        return (self.chrom == other.chrom and
                self.start == other.start and
                self.end == other.end and
                self.strand == other.strand)


@dataclass
class Transcript:
    """Represents a transcript with its exons."""
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: List[Exon] = field(default_factory=list)

    @property
    def introns(self) -> List[Intron]:
        """Derive introns from exon coordinates."""
        if len(self.exons) < 2:
            return []

        sorted_exons = sorted(self.exons, key=lambda e: e.start)
        introns = []
        for i in range(len(sorted_exons) - 1):
            intron = Intron(
                chrom=self.chrom,
                start=sorted_exons[i].end,
                end=sorted_exons[i + 1].start,
                strand=self.strand
            )
            introns.append(intron)
        return introns


@dataclass
class SpliceEvent:
    """Represents an alternative splicing event."""
    event_type: str  # SE, A5SS, A3SS, IR, MXE
    chrom: str
    strand: str
    gene_id: str

    # Event-specific coordinates
    # For SE: skipped exon coordinates
    # For A5SS/A3SS: alternative splice site position
    # For IR: retained intron coordinates
    coordinates: Tuple[int, ...]

    # Context (flanking exons/introns)
    upstream_exon: Optional[Exon] = None
    downstream_exon: Optional[Exon] = None

    def __hash__(self):
        return hash((self.event_type, self.chrom, self.strand, self.coordinates))

    def __eq__(self, other):
        return (self.event_type == other.event_type and
                self.chrom == other.chrom and
                self.strand == other.strand and
                self.coordinates == other.coordinates)

    @property
    def event_id(self) -> str:
        coords_str = '_'.join(map(str, self.coordinates))
        return f"{self.event_type}:{self.chrom}:{self.strand}:{coords_str}"


@dataclass
class EventComparisonResult:
    """Results of comparing events between reference and prediction."""
    event_type: str
    reference_events: Set[SpliceEvent]
    predicted_events: Set[SpliceEvent]
    matched_events: Set[SpliceEvent]

    @property
    def recall(self) -> float:
        if len(self.reference_events) == 0:
            return 0.0
        return len(self.matched_events) / len(self.reference_events)

    @property
    def precision(self) -> float:
        if len(self.predicted_events) == 0:
            return 0.0
        return len(self.matched_events) / len(self.predicted_events)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def parse_gtf(gtf_path: Path) -> Dict[str, List[Transcript]]:
    """
    Parse GTF file and extract transcripts grouped by gene.

    Returns:
        Dict mapping gene_id to list of Transcript objects
    """
    genes = defaultdict(list)
    transcripts = {}  # transcript_id -> Transcript

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attributes = fields
            start = int(start) - 1  # Convert to 0-based
            end = int(end)  # GTF end is already 1-based inclusive, so this is exclusive

            # Parse attributes
            attr_dict = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                match = re.match(r'(\w+)\s*"([^"]*)"', attr)
                if match:
                    attr_dict[match.group(1)] = match.group(2)

            gene_id = attr_dict.get('gene_id', '')
            transcript_id = attr_dict.get('transcript_id', '')

            if not gene_id or not transcript_id:
                continue

            if feature == 'transcript':
                transcript = Transcript(
                    transcript_id=transcript_id,
                    gene_id=gene_id,
                    chrom=chrom,
                    strand=strand
                )
                transcripts[transcript_id] = transcript

            elif feature == 'exon':
                if transcript_id not in transcripts:
                    # Create transcript if not seen yet
                    transcripts[transcript_id] = Transcript(
                        transcript_id=transcript_id,
                        gene_id=gene_id,
                        chrom=chrom,
                        strand=strand
                    )

                exon = Exon(chrom=chrom, start=start, end=end, strand=strand)
                transcripts[transcript_id].exons.append(exon)

    # Group by gene
    for transcript in transcripts.values():
        genes[transcript.gene_id].append(transcript)

    return dict(genes)


def detect_exon_skipping(gene_transcripts: List[Transcript]) -> Set[SpliceEvent]:
    """
    Detect skipped exon (SE) events within a gene.

    SE event: An exon present in one isoform but absent in another,
    where the flanking exons are shared.
    """
    events = set()

    if len(gene_transcripts) < 2:
        return events

    # Collect all exons and introns
    all_exons = set()
    all_introns = set()

    for transcript in gene_transcripts:
        for exon in transcript.exons:
            all_exons.add(exon)
        for intron in transcript.introns:
            all_introns.add(intron)

    # For each exon, check if there's a transcript that skips it
    for exon in all_exons:
        # Find transcripts containing this exon
        containing = [t for t in gene_transcripts
                     if exon in t.exons and len(t.exons) >= 3]

        # Find transcripts NOT containing this exon
        not_containing = [t for t in gene_transcripts
                        if exon not in t.exons and len(t.exons) >= 2]

        if not containing or not not_containing:
            continue

        # Check if any transcript has an intron spanning this exon
        for skip_transcript in not_containing:
            for intron in skip_transcript.introns:
                if intron.start < exon.start and intron.end > exon.end:
                    # This transcript skips the exon
                    event = SpliceEvent(
                        event_type='SE',
                        chrom=exon.chrom,
                        strand=exon.strand,
                        gene_id=gene_transcripts[0].gene_id,
                        coordinates=(exon.start, exon.end)
                    )
                    events.add(event)
                    break

    return events


def detect_alternative_5ss(gene_transcripts: List[Transcript]) -> Set[SpliceEvent]:
    """
    Detect alternative 5' splice site (A5SS) events.

    A5SS: Same 3' splice site but different 5' splice sites (donor sites).
    """
    events = set()

    if len(gene_transcripts) < 2:
        return events

    # Group introns by their 3' end (acceptor site)
    acceptor_groups = defaultdict(list)

    for transcript in gene_transcripts:
        for intron in transcript.introns:
            key = (intron.chrom, intron.end, intron.strand)
            acceptor_groups[key].append(intron)

    # Find groups with multiple different 5' sites
    for key, introns in acceptor_groups.items():
        donor_sites = set(intron.start for intron in introns)

        if len(donor_sites) > 1:
            chrom, acceptor, strand = key
            for donor in donor_sites:
                event = SpliceEvent(
                    event_type='A5SS',
                    chrom=chrom,
                    strand=strand,
                    gene_id=gene_transcripts[0].gene_id,
                    coordinates=(donor, acceptor)
                )
                events.add(event)

    return events


def detect_alternative_3ss(gene_transcripts: List[Transcript]) -> Set[SpliceEvent]:
    """
    Detect alternative 3' splice site (A3SS) events.

    A3SS: Same 5' splice site but different 3' splice sites (acceptor sites).
    """
    events = set()

    if len(gene_transcripts) < 2:
        return events

    # Group introns by their 5' start (donor site)
    donor_groups = defaultdict(list)

    for transcript in gene_transcripts:
        for intron in transcript.introns:
            key = (intron.chrom, intron.start, intron.strand)
            donor_groups[key].append(intron)

    # Find groups with multiple different 3' sites
    for key, introns in donor_groups.items():
        acceptor_sites = set(intron.end for intron in introns)

        if len(acceptor_sites) > 1:
            chrom, donor, strand = key
            for acceptor in acceptor_sites:
                event = SpliceEvent(
                    event_type='A3SS',
                    chrom=chrom,
                    strand=strand,
                    gene_id=gene_transcripts[0].gene_id,
                    coordinates=(donor, acceptor)
                )
                events.add(event)

    return events


def detect_intron_retention(gene_transcripts: List[Transcript]) -> Set[SpliceEvent]:
    """
    Detect intron retention (IR) events.

    IR: An intron in one isoform is retained (appears as exonic) in another.
    """
    events = set()

    if len(gene_transcripts) < 2:
        return events

    # Collect all introns
    all_introns = set()
    for transcript in gene_transcripts:
        for intron in transcript.introns:
            all_introns.add(intron)

    # For each intron, check if any transcript has an exon spanning it
    for intron in all_introns:
        for transcript in gene_transcripts:
            for exon in transcript.exons:
                # Check if exon spans the intron
                if exon.start <= intron.start and exon.end >= intron.end:
                    # Check that this transcript doesn't have this intron
                    if intron not in transcript.introns:
                        event = SpliceEvent(
                            event_type='IR',
                            chrom=intron.chrom,
                            strand=intron.strand,
                            gene_id=gene_transcripts[0].gene_id,
                            coordinates=(intron.start, intron.end)
                        )
                        events.add(event)
                        break

    return events


def detect_all_events(genes: Dict[str, List[Transcript]]) -> Dict[str, Set[SpliceEvent]]:
    """
    Detect all alternative splicing events across all genes.

    Returns:
        Dict mapping event_type to set of events
    """
    all_events = {
        'SE': set(),
        'A5SS': set(),
        'A3SS': set(),
        'IR': set()
    }

    with Progress() as progress:
        task = progress.add_task("[cyan]Detecting splice events...", total=len(genes))

        for gene_id, transcripts in genes.items():
            # Detect each event type
            all_events['SE'].update(detect_exon_skipping(transcripts))
            all_events['A5SS'].update(detect_alternative_5ss(transcripts))
            all_events['A3SS'].update(detect_alternative_3ss(transcripts))
            all_events['IR'].update(detect_intron_retention(transcripts))

            progress.update(task, advance=1)

    return all_events


def compare_events(
    ref_events: Dict[str, Set[SpliceEvent]],
    pred_events: Dict[str, Set[SpliceEvent]]
) -> Dict[str, EventComparisonResult]:
    """
    Compare splice events between reference and prediction.

    Returns:
        Dict mapping event_type to comparison results
    """
    results = {}

    for event_type in ['SE', 'A5SS', 'A3SS', 'IR']:
        ref_set = ref_events.get(event_type, set())
        pred_set = pred_events.get(event_type, set())

        # Find matched events (by coordinates)
        matched = ref_set & pred_set

        results[event_type] = EventComparisonResult(
            event_type=event_type,
            reference_events=ref_set,
            predicted_events=pred_set,
            matched_events=matched
        )

    return results


def generate_event_report(
    results: Dict[str, EventComparisonResult],
    output_path: Path
):
    """Generate JSON report of splice event analysis."""

    report = {
        'summary': {},
        'per_event_type': {}
    }

    total_ref = 0
    total_pred = 0
    total_matched = 0

    for event_type, result in results.items():
        report['per_event_type'][event_type] = {
            'reference_count': len(result.reference_events),
            'predicted_count': len(result.predicted_events),
            'matched_count': len(result.matched_events),
            'recall': result.recall,
            'precision': result.precision,
            'f1': result.f1
        }

        total_ref += len(result.reference_events)
        total_pred += len(result.predicted_events)
        total_matched += len(result.matched_events)

    report['summary'] = {
        'total_reference_events': total_ref,
        'total_predicted_events': total_pred,
        'total_matched_events': total_matched,
        'overall_recall': total_matched / total_ref if total_ref > 0 else 0,
        'overall_precision': total_matched / total_pred if total_pred > 0 else 0
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def print_results_table(results: Dict[str, EventComparisonResult]):
    """Print formatted results table."""

    table = Table(title="Splice Event Recovery Analysis")
    table.add_column("Event Type", style="cyan")
    table.add_column("Reference", style="blue", justify="right")
    table.add_column("Predicted", style="blue", justify="right")
    table.add_column("Matched", style="green", justify="right")
    table.add_column("Recall", style="yellow", justify="right")
    table.add_column("Precision", style="yellow", justify="right")
    table.add_column("F1", style="magenta", justify="right")

    event_names = {
        'SE': 'Exon Skipping',
        'A5SS': 'Alt 5\' SS',
        'A3SS': 'Alt 3\' SS',
        'IR': 'Intron Retention'
    }

    total_ref = 0
    total_pred = 0
    total_matched = 0

    for event_type in ['SE', 'A5SS', 'A3SS', 'IR']:
        result = results[event_type]
        table.add_row(
            event_names[event_type],
            str(len(result.reference_events)),
            str(len(result.predicted_events)),
            str(len(result.matched_events)),
            f"{result.recall:.2%}",
            f"{result.precision:.2%}",
            f"{result.f1:.2%}"
        )

        total_ref += len(result.reference_events)
        total_pred += len(result.predicted_events)
        total_matched += len(result.matched_events)

    # Add total row
    overall_recall = total_matched / total_ref if total_ref > 0 else 0
    overall_precision = total_matched / total_pred if total_pred > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_ref}[/bold]",
        f"[bold]{total_pred}[/bold]",
        f"[bold]{total_matched}[/bold]",
        f"[bold]{overall_recall:.2%}[/bold]",
        f"[bold]{overall_precision:.2%}[/bold]",
        f"[bold]{overall_f1:.2%}[/bold]"
    )

    console.print(table)


def save_events_to_bed(
    events: Dict[str, Set[SpliceEvent]],
    output_dir: Path,
    prefix: str
):
    """Save events to BED format for visualization."""

    for event_type, event_set in events.items():
        bed_path = output_dir / f"{prefix}_{event_type}.bed"

        with open(bed_path, 'w') as f:
            for event in sorted(event_set, key=lambda e: (e.chrom, e.coordinates)):
                start = event.coordinates[0]
                end = event.coordinates[-1] if len(event.coordinates) > 1 else start + 1
                name = f"{event.event_type}_{event.gene_id}"
                score = 1000
                strand = event.strand

                f.write(f"{event.chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")


@click.command()
@click.option('--reference', '-r', type=click.Path(exists=True), required=True,
              help='Reference GTF file (e.g., AtRTD3)')
@click.option('--predicted', '-p', type=click.Path(exists=True), required=True,
              help='Predicted GTF file (TransGenic output)')
@click.option('--output-dir', '-o', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--prefix', default='splice_events',
              help='Output file prefix')
def main(reference: str, predicted: str, output_dir: str, prefix: str):
    """
    Detect and compare alternative splicing events.

    Analyzes splice events (SE, A5SS, A3SS, IR) in reference and predicted
    annotations, calculating recovery metrics.
    """
    reference_path = Path(reference)
    predicted_path = Path(predicted)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Splice Event Detection and Comparison[/bold]")
    console.print(f"Reference: {reference_path}")
    console.print(f"Predicted: {predicted_path}")
    console.print(f"Output: {output_path}")
    console.print()

    # Step 1: Parse GTF files
    console.print("[blue]Parsing reference GTF...[/blue]")
    ref_genes = parse_gtf(reference_path)
    console.print(f"  Found {len(ref_genes)} genes")

    console.print("[blue]Parsing predicted GTF...[/blue]")
    pred_genes = parse_gtf(predicted_path)
    console.print(f"  Found {len(pred_genes)} genes")
    console.print()

    # Step 2: Detect events
    console.print("[blue]Detecting reference splice events...[/blue]")
    ref_events = detect_all_events(ref_genes)

    console.print("[blue]Detecting predicted splice events...[/blue]")
    pred_events = detect_all_events(pred_genes)
    console.print()

    # Step 3: Compare events
    console.print("[blue]Comparing events...[/blue]")
    results = compare_events(ref_events, pred_events)

    # Step 4: Print and save results
    print_results_table(results)
    console.print()

    # Save JSON report
    report = generate_event_report(results, output_path / f'{prefix}_report.json')

    # Save events to BED
    save_events_to_bed(ref_events, output_path, f'{prefix}_reference')
    save_events_to_bed(pred_events, output_path, f'{prefix}_predicted')

    console.print(f"[bold green]Analysis complete![/bold green]")
    console.print(f"Results saved to: {output_path}")
    console.print()
    console.print("[bold]Output files:[/bold]")
    console.print(f"  - {prefix}_report.json: Summary metrics")
    console.print(f"  - {prefix}_reference_*.bed: Reference events")
    console.print(f"  - {prefix}_predicted_*.bed: Predicted events")


if __name__ == '__main__':
    main()
