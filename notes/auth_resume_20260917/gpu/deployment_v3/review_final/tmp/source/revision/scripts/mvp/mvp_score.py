#!/usr/bin/env python3
"""CPU MVP statistics from caller-supplied counts and final eval-pair JSONL.

Legacy JSON: pairs maps '1','2','3' to final deduplicated JSONL paths; loci is
50 records {gene_id, seeds: {'1': {'off': counts, 'on': counts}, ...}}.
counts: f1_tp, f1_predicted, f1_reference (GFFCompare v0.12.6 restricted
transcript counts; monoexonic exact CDS), and optionally precision_tp,
predicted, recall_tp, supported (exact junction counts). All are integers.
Legacy structural counts must be present everywhere or nowhere, never filled as
zero for missing evidence. Explicit development-single uses pairs.mvp_seed123,
loci.seeds['123'], and descriptive single-seed bootstrap;
partially missing structural counts suppress pooled P/R rather than becoming zero. F1 is percentage points; P/R are fractions.
This adapter does not decode models or claim to run GFFCompare. Pair paths
must identify distinct files, but filenames and seed keys do not prove distinct
runs. API and CLI outputs mark pair/run provenance UNVERIFIED and three-seed
identity/policy unverified until anchored producer receipts are integrated.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from transgenic.evidence import mvp_bootstrap as b, mvp_windows as w


def require_coverage_policy(policy):
    # No policy is approved in this implementation. A future author-approved
    # policy must specify technology/run coverage before decision publication.
    raise ValueError('coverage policy pending author approval (HiFi versus ONT-only)')


def predicted_junctions(transcripts, locus):
    """Normalized decoded transcripts: chrom, strand, half-open exon intervals.

    Select by transcript-span overlap with the gene, then retain ALL introns.
    No strand-based overlap filter: wrong-strand predictions stay penalized.
    """
    genes = w.genes_by_chromosome([locus])
    result = set()
    for tx in transcripts:
        exons = sorted(tx['exons'])
        if (not exons or tx['strand'] not in ('+', '-') or
                any(type(a) is not int or type(z) is not int or not 0 <= a < z for a, z in exons) or
                any(a[1] >= z[0] for a, z in zip(exons, exons[1:]))):
            raise ValueError('invalid transcript exons/strand')
        if w.transcript_overlaps_gene(tx['chrom'], exons[0][0], exons[-1][1] - 1, genes):
            result.update((tx['chrom'], a[1], z[0] - 1, tx['strand']) for a, z in zip(exons, exons[1:]))
    return result


def junction_counts(predicted, supported, locus):
    """Predicted set already selected by transcript-overlaps-gene, not clipped."""
    if supported is None:
        return None
    windows = w.windows_by_chromosome([locus])
    supported = {j for j in supported if w.junction_within_windows(*j[:3], windows)}
    genes = w.genes_by_chromosome([locus])
    recall = {j for j in supported if w.junction_within_genes(*j[:3], genes)}
    return dict(predicted=len(predicted), precision_tp=len(predicted & supported),
                supported=len(recall), recall_tp=len(predicted & recall))


def metric(rows, fraction, name, seeds=(1, 2, 3)):
    counts = [r['seeds'][str(s)][fraction] for r in rows for s in seeds]
    total = lambda key: sum(c[key] for c in counts)
    if name == 'F1':
        denominator = total('f1_predicted') + total('f1_reference')
        return 200 * total('f1_tp') / denominator if denominator else float('nan')
    numerator, denominator = ('precision_tp', 'predicted') if name == 'P' else ('recall_tp', 'supported')
    d = total(denominator)
    return total(numerator) / d if d else float('nan')


def score(payload, panel):
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE, validate_seed_profile
    profile = payload.get('profile', LEGACY_PROFILE)
    development = profile == DEVELOPMENT_PROFILE
    seeds = (123,) if development else (1, 2, 3)
    validate_seed_profile(profile, seeds[0])
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict) or metadata.get('profile', profile) != profile:
        raise ValueError('metadata profile mismatch')
    if development:
        for record in (payload, metadata):
            if 'run_id' in record and record['run_id'] != 'mvp_seed123':
                raise ValueError('development scorer run identity mismatch')
            if record.get('run_role', 'mvp_finetune') != 'mvp_finetune':
                raise ValueError('development scorer run role mismatch')
            if 'seed' in record:
                validate_seed_profile(profile, record['seed'])
    pair_keys = {'mvp_seed123'} if development else {'1', '2', '3'}
    if set(payload['pairs']) != pair_keys:
        raise ValueError('exact profile pair run inventory required' if development else 'exact seeds 1, 2, 3 required')
    paths = {s: Path(path).resolve(strict=True) for s, path in payload['pairs'].items()}
    seen = []
    for path in paths.values():
        if any(path == previous or path.samefile(previous) for previous in seen):
            raise ValueError('duplicate pair input file; seed labels do not establish run identity')
        seen.append(path)
    records = {(123 if development else int(s)): [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               for s, path in paths.items()}
    rns, delta = b.build_tiles(records, expected_seeds=seeds, profile=profile)
    primary = b.tile_bootstrap(delta)
    composite = math.fsum(r['composite_loss_off'] - r['composite_loss_on']
                         for rows in records.values() for r in rows if r['has_evidence']) / (len(seeds) * len(rns))
    loci = payload['loci']
    ids = [r['gene_id'] for r in loci]
    if len(ids) != len(set(ids)) or set(ids) != {l.gene_id for l in panel}:
        raise ValueError('exact full panel locus inventory required')
    loci = sorted(loci, key=lambda r: r['gene_id'])
    structural_flags = []
    for row in loci:
        if set(row['seeds']) != {str(seed) for seed in seeds}:
            raise ValueError('each locus must carry exactly the profile seeds')
        frozen = {}
        for pair in row['seeds'].values():
            if set(pair) != {'off', 'on'}:
                raise ValueError('paired fractions off/on required')
            for c in pair.values():
                required = {'f1_tp', 'f1_predicted', 'f1_reference'}
                structural = {'precision_tp', 'predicted', 'recall_tp', 'supported'}
                if set(c) not in (required, required | structural):
                    raise ValueError('count schema mismatch')
                if any(type(v) is not int or v < 0 for v in c.values()):
                    raise ValueError('counts must be nonnegative integers')
                if c['f1_tp'] > min(c['f1_predicted'], c['f1_reference']):
                    raise ValueError('F1 numerator exceeds denominator')
                available = structural <= set(c)
                structural_flags.append(available)
                if available and (c['precision_tp'] > c['predicted'] or c['recall_tp'] > c['supported']):
                    raise ValueError('junction numerator exceeds denominator')
                if available and c['recall_tp'] > c['precision_tp']:
                    raise ValueError('recall true positives must be a subset of precision true positives')
                for field in ['f1_reference'] + (['supported'] if available else []):
                    if c[field] != frozen.setdefault(field, c[field]):
                        raise ValueError(f'{field} must be frozen across all fractions and seeds per locus')
    if not development and any(structural_flags) != all(structural_flags):
        raise ValueError('partially missing structural scoring; do not encode missing as negative')
    result = dict(label='exploratory / MVP', panel_sha256=w.PANEL_SHA256,
                  primary=primary, composite_delta_diagnostic=composite,
                  structural_available=all(structural_flags), readouts={},
                  training_eligible=False,
                  pair_run_provenance_status='UNVERIFIED',
                  three_seed_identity_verified=False, three_seed_policy_verified=False,
                  decision=None, decision_status='BLOCKED_PENDING_COVERAGE_POLICY',
                  precision_rule='point_estimate_ge_0; interpretation pending author review')
    if development:
        result.pop('three_seed_identity_verified')
        result.pop('three_seed_policy_verified')
        result.update(profile=profile, run_id='mvp_seed123', run_role='mvp_finetune', seed=123,
                      decision_status='DESCRIPTIVE_SINGLE_SEED_NO_FINAL_DECISION',
                      structural_status=('AVAILABLE_CALLER_COUNTS_UNVERIFIED' if all(structural_flags)
                                         else 'PARTIAL_UNASSESSABLE_NOT_ZERO' if any(structural_flags)
                                         else 'MISSING_UNASSESSABLE_NOT_ZERO'),
                      structural_pairs_present=sum(structural_flags), structural_pairs_total=len(structural_flags),
                      bootstrap_scope='descriptive paired tiles/loci conditional on one training seed; no across-seed inference',
                      precision_rule='descriptive only; no final GO/STOP')
    core = {l.gene_id for l in panel if l.panel == 'mvp_core'}
    for stratum, rows in [('all50', loci), ('core20', [r for r in loci if r['gene_id'] in core])]:
        result['readouts'][stratum] = {}
        for name in ['F1'] + (['P', 'R'] if all(structural_flags) else []):
            by_seed = {}
            groups = [('123', (123,))] if development else [('pooled', (1, 2, 3)), ('1', (1,)), ('2', (2,)), ('3', (3,))]
            for label, group_seeds in groups:
                try:
                    by_seed[label] = b.locus_bootstrap(rows, lambda sample: metric(sample, 'on', name, group_seeds) - metric(sample, 'off', name, group_seeds))
                except b.BootstrapError as e:
                    by_seed[label] = {'status': 'UNASSESSABLE', 'reason': str(e)}
            result['readouts'][stratum][name] = by_seed
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--panels', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--final-decision', action='store_true', help='currently fail-closed pending author policy')
    from transgenic.training.mvp_profiles import LEGACY_PROFILE, DEVELOPMENT_PROFILE
    p.add_argument('--profile', choices=(LEGACY_PROFILE, DEVELOPMENT_PROFILE))
    a = p.parse_args(argv)
    input_hash = w.sha256_file(a.input)
    payload = json.loads(a.input.read_text())
    profile = payload.get('profile', LEGACY_PROFILE)
    if a.profile is not None and a.profile != profile:
        raise ValueError('CLI/payload profile mismatch')
    if a.final_decision:
        if profile == DEVELOPMENT_PROFILE:
            raise ValueError('development-single has no final decision')
        require_coverage_policy(None)
    hashes = {str(path): w.sha256_file(path) for path in [a.panels, *payload['pairs'].values()]}
    hashes[str(a.input)] = input_hash
    result = score(payload, w.load_panel(a.panels))
    if any(w.sha256_file(path) != expected for path, expected in hashes.items()):
        raise ValueError('scoring input changed during computation')
    result['input_sha256'] = hashes
    result['count_provenance_status'] = 'caller-supplied counts; not independently verified by this adapter'
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
