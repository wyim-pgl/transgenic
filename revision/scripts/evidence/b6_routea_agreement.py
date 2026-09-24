#!/usr/bin/env python3
"""Route A (miniprot CDS phase) vs route B (EST x protein blastx) per-base agreement QC.

A46 rule 6: "the miniprot CDS blocks of the same cross-species proteins give a
genome-derived phase; where A and B disagree on a base both phase channels are set
to 0 and the disagreement is counted in QC."

Route A = CDS features of miniprot mRNAs (`<sp>.protein.gff.gz`) whose alignment is
`accepted` in `accept/<sp>.alignments.tsv`, whose protein is NOT in
`<sp>.mapping_ambiguous.txt` (A43) and whose protein appears in route B (the "same"
proteins; Swiss-Prot proteins were never miniprot-aligned, so they are B-only).
GFF3 CDS phase p (bases to skip before the first whole codon) gives codon position
(3 - p) % 3 at the CDS 5' base; the label is carried along the block in coding
direction. Route B = the `est_protein_hsp` rows of the species' published PROTEIN C0
(`--protein-c0`; est_phase_project.PHASE_CONVENTION).

Comparison unit: (chrom, base, protein_id). A label set is the set of (strand, codon
position) a route assigns there, so the same protein placed on opposite strands at one
base disagrees. agree = both non-empty and equal; disagree = both non-empty and
different; A-only / B-only otherwise. Base level (chrom, base) takes the strongest
protein-level category: disagree > agree > B-only > A-only. Output: per-species base
counts (all proteins), the same comparison per route-B protein_source (`orthodb`,
`swissprot`; always both keys, matching the per-source MULTI_HSP_RULE) with its own
disagreement intervals/bases, and the merged disagreement intervals (0-based half-open,
pooled over sources): ALL six route-B phase channels are zeroed over them. Route A never
becomes a channel.

Single producer (fix round 5, findings 2 and 3). The protein C0 build
(`b6_build_c0.load_hsp`) is the one place that knows every HSP's EST-molecule state, its
protein query's rule 4(b) state and, after both, which HSPs survive the rule 6
MULTI_HSP_RULE; this script reads THAT set from the protein C0 instead of the projection
TSV, so it can never re-derive a different one. Rows are dropped BEFORE `compare` and
`compare_sources` when the C0 flags them `leakage_excluded` (EST molecule excluded_heldout
or never-training, or the protein query excluded; QC `route_b_leakage_filter` keys
rows_dropped_excluded_heldout / rows_dropped_never_training /
rows_dropped_protein_query_excluded) or `train_tiles_excluded` (rows_dropped_train_tiles_excluded,
rows_dropped_protein_query_train_tiles_excluded when the flag came from the protein query).
The zero list is genome-wide and carries no tile scope, so a train-tiles-excluded row cannot
be kept for non-training tiles only.

Input binding (fail closed). `--route-b`, its `<route-b>.molecule_state.tsv` sidecar,
`--est-c0` and `--miniprot-gff` are REQUIRED and each file's sha256 must equal the sha256
the protein C0 recorded for it in `c0_provenance.inputs` (`hsp`, `hsp_molecule_state`,
`est_c0`, `alignment:protein`): an older protein C0 presented with newer files, or newer
files with an older C0, is refused. `--accepted` and `--mapping-ambiguous` are bound by
content: every mRNA of the GFF (per protein, in file order = the C0's record_index) must
have a C0 `evidence_alignment` row whose `mapping_ambiguous` equals membership in the
ambiguous list and whose `accepted` equals (verdict == accepted and not ambiguous), the
accept table's protein must equal the GFF Target, and the C0 must have no alignment the GFF
lacks.

Protein queries (A46 rule 3: "the leakage-closure unit is the protein query sequence with
all its alignments ... A molecule is one unit everywhere it appears"; rule 6: "protein queries
touching excluded intervals are removed molecule-wide"): the per-protein rule 4(b) decision
comes from the protein C0 `molecule_member` (source = 'protein', read_name = the miniprot
query id). An accepted route-A alignment whose protein is `excluded_heldout`, `never` or
train-tiles-excluded is dropped before `compare` (QC `route_a`:
A_leakage_excluded_alignments_removed, A_never_training_alignments_removed,
A_train_tiles_excluded_alignments_removed); an accepted alignment whose protein has no
molecule in the protein C0 fails closed. The output TSV is written in sorted
(reason, chrom, start0) order and is byte-deterministic for one input set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import sys

_EVIDENCE = Path(__file__).resolve().parent
if str(_EVIDENCE) not in sys.path:
    sys.path.insert(0, str(_EVIDENCE))

import b6_build_c0 as B  # noqa: E402

CATEGORIES = ('A_only', 'B_only', 'agree', 'disagree')  # ascending priority
MOLECULE_STATES = ('train_track', 'excluded_heldout', 'never')                        # c0_schema
SOURCES = ('orthodb', 'swissprot')                      # b6_hsp_project.SOURCES
PRIORITY = {c: i for i, c in enumerate(CATEGORIES)}
BOUND_INPUTS = (('--est-c0', 'est_c0'), ('--route-b', 'hsp'), ('<route-b>.molecule_state.tsv', 'hsp_molecule_state'),
                ('--miniprot-gff', 'alignment:protein'))


def _open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)


def anchor(strand, start0, phase):
    """Label at base g is (g - anchor) % 3 on '+', (anchor - g) % 3 on '-'."""
    return (start0 - phase) % 3 if strand == '+' else (start0 + phase) % 3


def label(strand, anc, g):
    return (g - anc) % 3 if strand == '+' else (anc - g) % 3


def gff_cds_phase_at_start0(strand, start0, end0, gff_phase):
    """GFF3 CDS phase -> codon position at start0 (the PHASE_CONVENTION of route B)."""
    if gff_phase not in (0, 1, 2) or not 0 <= start0 < end0:
        raise ValueError('invalid CDS phase/interval')
    five_prime = (3 - gff_phase) % 3
    return five_prime if strand == '+' else (five_prime + end0 - 1 - start0) % 3


def state_path(route_b):
    return Path(str(route_b) + '.molecule_state.tsv')


def bind_inputs(args):
    """Open the published protein C0 and require every other input to be the file it recorded (sha256)."""
    sha, prov = B.open_published_c0(args.protein_c0, args.species, 'protein')
    inputs = json.loads(prov.get('inputs', '{}'))
    bound = {'protein_c0': {'path': str(Path(args.protein_c0).absolute()), 'sha256': sha}}
    paths = {'est_c0': args.est_c0, 'hsp': args.route_b, 'hsp_molecule_state': state_path(args.route_b),
             'alignment:protein': args.miniprot_gff}
    for name, key in BOUND_INPUTS:
        rec = inputs.get(key) or {}
        if not rec.get('sha256'):
            raise ValueError(f'{args.protein_c0}: c0_provenance.inputs records no sha256 for {key} ({name}); '
                             'not a protein C0 of the current builder')
        path = Path(paths[key])
        if not path.is_file():
            raise ValueError(f'missing {name} {path}; refusing to zero without it')
        got = B.digest(path)
        if got != rec['sha256']:
            raise ValueError(f'{name} {path} (sha256 {got}) is not the input the protein C0 {args.protein_c0} was '
                             f'built from (recorded sha256 {rec["sha256"]} for {rec.get("path")}); a stale C0 or a '
                             'rebuilt input cannot be paired')
        bound[key] = {'path': str(path.absolute()), 'sha256': got}
    return bound


POPULATIONS = ('train_track', 'valid_track', 'eval_input')


def load_est_state(est_c0, species):
    """{EST molecule_id: molecule_state} from the species' published EST C0 (eval_input population)."""
    import duckdb

    con = duckdb.connect(str(Path(est_c0).absolute()), read_only=True)
    try:
        rows = con.execute("""SELECT molecule_id, any_value(molecule_state), count(DISTINCT molecule_state)
                              FROM molecule_member WHERE species = ? AND source = 'est' GROUP BY molecule_id""",
                           [species]).fetchall()
    finally:
        con.close()
    bad = [m for m, _, n in rows if n != 1]
    if bad:
        raise ValueError(f'{est_c0}: EST molecules with several states {bad[:3]}')
    return {m: st for m, st, _ in rows}


def load_route_b(protein_c0, species, population='train_track', est_state=None):
    """Route-B rows from the protein C0's est_protein_hsp: (row tuples of the population, QC counts).

    row = (chrom, strand, protein_id, start0, end0, anchor, protein_source).
    Populations (plan §7.4 D4): ``train_track`` drops leakage-excluded and train-tiles-excluded rows (the producer's
    original behaviour); ``valid_track`` admits train-tiles-excluded rows; ``eval_input`` keeps exactly the rows of
    EST molecules whose EST C0 state is ``excluded_heldout`` (the A-EST evaluation input; b6_score sets are
    long-read sets and never have route-B rows)."""
    import duckdb

    if population not in POPULATIONS:
        raise ValueError(f'unknown population {population!r}')
    if population == 'eval_input' and est_state is None:
        raise ValueError('eval_input needs the EST molecule states')
    con = duckdb.connect(str(Path(protein_c0).absolute()), read_only=True)
    try:
        rows = con.execute("""
            SELECT chrom, strand, protein_id, genomic_start0, genomic_end0, phase, protein_source, leakage_excluded,
                   train_tiles_excluded, exclusion_cause, molecule_id
            FROM est_protein_hsp WHERE species = ? ORDER BY hsp_id, interval_index""", [species]).fetchall()
    finally:
        con.close()
    kept, qc = [], Counter()
    for chrom, strand, protein, s, e, p, source, leak, tte, cause, mid in rows:
        if strand not in '+-' or p not in (0, 1, 2) or not 0 <= s < e:
            raise ValueError(f'invalid route-B interval in {protein_c0}: {(chrom, strand, protein, s, e, p)}')
        if source not in SOURCES:
            raise ValueError(f'unknown route-B protein_source {source!r}')
        if population == 'eval_input':
            if mid not in est_state:
                raise ValueError(f'route-B molecule {mid} has no EST C0 molecule')
            if est_state[mid] != 'excluded_heldout':
                qc[f'rows_dropped_not_eval_input_{est_state[mid]}'] += 1
                continue
            kept.append((chrom, strand, protein, s, e, anchor(strand, s, p), source))
            qc['rows_used'] += 1
            continue
        if leak:
            if cause == 'protein_query':
                qc['rows_dropped_protein_query_excluded'] += 1
            elif cause == 'est_molecule_never':
                qc['rows_dropped_never_training'] += 1
            else:
                qc['rows_dropped_excluded_heldout'] += 1
            continue
        if tte and population == 'train_track':
            qc['rows_dropped_protein_query_train_tiles_excluded' if cause == 'protein_query'
               else 'rows_dropped_train_tiles_excluded'] += 1
            continue
        kept.append((chrom, strand, protein, s, e, anchor(strand, s, p), source))
        qc['rows_used'] += 1
    return kept, qc


def load_protein_state(protein_c0, species):
    """{miniprot query id: (molecule_state, train_tiles_excluded)} from the species' published protein C0
    (b6_build_c0.load_protein: one member = one molecule per protein query, read_name = the query id).  Fails closed."""
    import duckdb

    con = duckdb.connect(str(Path(protein_c0).absolute()), read_only=True)
    try:
        rows = con.execute("""
            SELECT read_name, any_value(molecule_state), bool_or(train_tiles_excluded), count(DISTINCT molecule_state)
            FROM molecule_member WHERE species = ? AND source = 'protein' GROUP BY read_name ORDER BY read_name""",
                           [species]).fetchall()
    finally:
        con.close()
    state = {}
    for protein, st, tte, n_states in rows:
        if n_states != 1 or st not in MOLECULE_STATES:
            raise ValueError(f'{protein_c0}: protein query {protein} carries {n_states} molecule states ({st!r})')
        state[protein] = (st, bool(tte))
    if not state:
        raise ValueError(f'{protein_c0}: no protein molecule for {species}')
    return state


def protein_drop_reason(protein_state, protein, population='train_track'):
    st, tte = protein_state[protein]
    if population == 'eval_input':          # protein-query exclusion is a training-side rule (rule 4(b))
        return None
    if st == 'excluded_heldout':
        return 'leakage_excluded'
    if st == 'never':
        return 'never_training'
    if tte and population == 'train_track':
        return 'train_tiles_excluded'
    return None


def gff_mrnas(gff):
    """[(mRNA id, Target protein)] in file order, and {mRNA id: record_index within its protein} (the C0's
    record_index: b6_build_c0.parse_miniprot yields records in file order; load_protein numbers them per protein)."""
    order, index, per_protein = [], {}, Counter()
    with _open(gff) as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) != 9 or c[2] != 'mRNA':
                continue
            attrs = dict(kv.split('=', 1) for kv in c[8].split(';') if '=' in kv)
            mid, target = attrs.get('ID', ''), attrs.get('Target', '').split(' ')[0]
            if not mid or not target:
                raise ValueError(f'{gff}: mRNA without ID/Target: {line.strip()[:120]}')
            if mid in index:
                raise ValueError(f'{gff}: duplicate mRNA ID {mid}')
            index[mid] = per_protein[target]
            per_protein[target] += 1
            order.append((mid, target))
    return order, index


def verify_accept_table(protein_c0, species, gff, accepted_tsv, ambiguous):
    """Bind --accepted / --mapping-ambiguous by content to the protein C0's evidence_alignment (see docstring)."""
    import duckdb

    order, index = gff_mrnas(gff)
    verdict = {}
    with _open(accepted_tsv) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['alignment_id'] in verdict:
                raise ValueError(f'{accepted_tsv}: alignment {r["alignment_id"]} listed twice')
            verdict[r['alignment_id']] = (r['protein'], r['verdict'])
    con = duckdb.connect(str(Path(protein_c0).absolute()), read_only=True)
    try:
        c0 = {(p, i): (acc, amb) for p, i, acc, amb in con.execute("""
            SELECT read_name, record_index, accepted, mapping_ambiguous FROM evidence_alignment
            WHERE species = ? AND source = 'protein' ORDER BY read_name, record_index""", [species]).fetchall()}
    finally:
        con.close()
    seen = set()
    for mid, target in order:
        if mid not in verdict:
            raise ValueError(f'{accepted_tsv}: GFF alignment {mid} has no verdict row')
        prot, v = verdict[mid]
        if prot != target:
            raise ValueError(f'{mid}: GFF Target {target} != accept table protein {prot}')
        key = (target, index[mid])
        if key not in c0:
            raise ValueError(f'{mid}: no evidence_alignment row ({target}, record {index[mid]}) in the protein C0')
        acc, amb = c0[key]
        seen.add(key)
        if amb != (target in ambiguous):
            raise ValueError(f'{mid}: mapping_ambiguous {amb} in the protein C0 but {"in" if target in ambiguous else "not in"}'
                             f' the --mapping-ambiguous list')
        if acc != (v == 'accepted' and target not in ambiguous):
            raise ValueError(f'{mid}: accepted={acc} in the protein C0 but the accept table says {v!r}'
                             f'{" (ambiguous)" if target in ambiguous else ""}; the tables are not from one miniprot run')
    if seen != set(c0):
        extra = sorted(set(c0) - seen)[:3]
        raise ValueError(f'the protein C0 has {len(set(c0) - seen)} alignment(s) absent from --miniprot-gff, e.g. {extra}')
    return len(order)


def load_route_a(gff, accepted_tsv, ambiguous_txt, proteins, protein_state, population='train_track'):
    ambiguous = {line.strip() for line in _open(ambiguous_txt) if line.strip()}
    keep, qc = {}, Counter()
    with _open(accepted_tsv) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['verdict'] != 'accepted':
                continue
            if r['protein'] in ambiguous:
                qc['A_mapping_ambiguous_alignments_removed'] += 1
                continue
            if r['protein'] not in protein_state:
                raise ValueError(f'{r["protein"]}: accepted miniprot alignment {r["alignment_id"]} has no molecule in '
                                 'the protein C0 (--protein-c0 is not the C0 of this miniprot run)')
            reason = protein_drop_reason(protein_state, r['protein'], population)
            if reason is not None:
                qc[f'A_{reason}_alignments_removed'] += 1
                continue
            if r['protein'] in proteins:
                keep[r['alignment_id']] = r['protein']
    rows, mrna_target = [], {}
    with _open(gff) as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) != 9:
                continue
            attrs = dict(kv.split('=', 1) for kv in c[8].split(';') if '=' in kv)
            if c[2] == 'mRNA' and attrs.get('ID') in keep:
                target = attrs.get('Target', '').split(' ')[0]
                if target != keep[attrs['ID']]:
                    raise ValueError(f'{attrs["ID"]}: GFF Target {target} != accept table protein')
                mrna_target[attrs['ID']] = target
            elif c[2] == 'CDS' and attrs.get('Parent') in mrna_target:
                start0, end0, strand = int(c[3]) - 1, int(c[4]), c[6]
                if strand not in '+-' or c[7] not in ('0', '1', '2'):
                    raise ValueError(f'invalid CDS strand/phase: {line.strip()[:120]}')
                p = gff_cds_phase_at_start0(strand, start0, end0, int(c[7]))
                rows.append((c[0], strand, mrna_target[attrs['Parent']], start0, end0, anchor(strand, start0, p)))
    qc['A_alignments_used'] = len(mrna_target)
    qc['A_cds_blocks'] = len(rows)
    return rows, qc


def _sweep(intervals):
    """intervals: (start, end, payload). Yields (start, end, Counter of active payloads)."""
    events = defaultdict(Counter)
    for s, e, payload in intervals:
        events[s][payload] += 1
        events[e][payload] -= 1
    active, last = Counter(), None
    for pos in sorted(events):
        if last is not None and pos > last and +active:
            yield last, pos, +active
        active.update(events[pos])
        last = pos


def compare(route_a, route_b):
    """-> (base counts by category, disagreement intervals by chrom)."""
    per_protein = defaultdict(list)
    for chrom, strand, protein, s, e, anc in route_a:
        per_protein[chrom, protein].append((s, e, ('A', strand, anc)))
    for chrom, strand, protein, s, e, anc, _ in route_b:
        per_protein[chrom, protein].append((s, e, ('B', strand, anc)))
    per_base = defaultdict(list)
    for (chrom, _), intervals in per_protein.items():
        for s, e, active in _sweep(intervals):
            # Anchors are strand-specific, so (strand, anchor) equality == label equality.
            a = {x[1:] for x in active if x[0] == 'A'}
            b = {x[1:] for x in active if x[0] == 'B'}
            cat = ('agree' if a == b else 'disagree') if a and b else ('A_only' if a else 'B_only')
            per_base[chrom].append((s, e, cat))
    counts, zero = Counter({c: 0 for c in CATEGORIES}), defaultdict(list)
    for key, segments in per_base.items():
        for s, e, active in _sweep(segments):
            cat = max(active, key=PRIORITY.get)
            counts[cat] += e - s
            if cat == 'disagree':
                runs = zero[key]
                if runs and runs[-1][1] == s:
                    runs[-1][1] = e
                else:
                    runs.append([s, e])
    return counts, zero


def compare_sources(route_b):
    """Route-B cross-source disagreement (A48, author decision 2026-09-23).

    Per base, the set of (strand, anchor) labels contributed by OrthoDB rows is
    compared with the set contributed by Swiss-Prot rows; anchors are frame
    references, so equal (strand, anchor) means equal codon position at every
    base. Where both sources label a base and the sets differ, all six route-B
    phase channels are zeroed there, exactly as for a route A/B disagreement.
    -> (bases where both sources label the base, disagreement intervals by chrom).
    """
    per_chrom = defaultdict(list)
    for chrom, strand, _protein, s, e, anc, source in route_b:
        per_chrom[chrom].append((s, e, (source, strand, anc)))
    both, zero = 0, defaultdict(list)
    for chrom, intervals in per_chrom.items():
        for s, e, active in _sweep(intervals):
            o = {x[1:] for x in active if x[0] == 'orthodb'}
            w = {x[1:] for x in active if x[0] == 'swissprot'}
            if not (o and w):
                continue
            both += e - s
            if o != w:
                runs = zero[chrom]
                if runs and runs[-1][1] == s:
                    runs[-1][1] = e
                else:
                    runs.append([s, e])
    return both, zero


def run(args):
    for path in (args.out_intervals, args.qc):
        if path.exists():
            raise FileExistsError(path)
    bound = bind_inputs(args)
    ambiguous = {line.strip() for line in _open(args.mapping_ambiguous) if line.strip()}
    n_gff = verify_accept_table(args.protein_c0, args.species, args.miniprot_gff, args.accepted, ambiguous)
    protein_state = load_protein_state(args.protein_c0, args.species)
    population = getattr(args, 'population', 'train_track')
    est_state = load_est_state(args.est_c0, args.species) if population == 'eval_input' else None
    route_b, qc_b = load_route_b(args.protein_c0, args.species, population, est_state)
    proteins = {r[2] for r in route_b}
    route_a, qc_a = load_route_a(args.miniprot_gff, args.accepted, args.mapping_ambiguous, proteins, protein_state,
                                 population)
    counts, zero = compare(route_a, route_b)
    both_sources_bases, cross_zero = compare_sources(route_b)
    by_source, detail = {}, {}
    for source in SOURCES:
        subset = [r for r in route_b if r[6] == source]
        wanted = {r[2] for r in subset}
        a_subset = [r for r in route_a if r[2] in wanted]
        source_counts, source_zero = compare(a_subset, subset)
        by_source[source] = dict(source_counts)
        detail[source] = dict(bases=dict(source_counts), route_b_rows=len(subset),
                              route_b_proteins=len(wanted),
                              route_a_proteins=len({r[2] for r in a_subset}),
                              route_a_cds_blocks=len(a_subset),
                              disagreement_intervals=sum(len(v) for v in source_zero.values()),
                              disagreement_bases=sum(e - s for v in source_zero.values() for s, e in v))
    lines = []
    for chrom, runs in zero.items():
        lines += [(chrom, s, e, 'route_a_b_disagreement') for s, e in runs]
    for chrom, runs in cross_zero.items():
        lines += [(chrom, s, e, 'route_b_cross_source_disagreement') for s, e in runs]
    lines.sort(key=lambda x: (x[3], x[0], x[1], x[2]))
    with open(args.out_intervals, 'x', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['species', 'chrom', 'start0', 'end0', 'action', 'reason'])
        for chrom, s, e, reason in lines:
            w.writerow([args.species, chrom, s, e, 'zero_all_route_b_phase_channels', reason])
    report = dict(species=args.species, population=population,
                  upstream={'protein_c0': bound['protein_c0']['sha256'], 'est_c0': bound['est_c0']['sha256']},
                  bases=dict(counts), bases_by_route_b_source=by_source,
                  by_route_b_source=detail, route_a=dict(qc_a), route_b_leakage_filter=dict(qc_b),
                  route_b_source='protein C0 est_protein_hsp (resolved by ' + B.MULTI_HSP_APPLIED_BY + ')',
                  inputs_bound_to_protein_c0=bound, gff_alignments_bound=n_gff,
                  disagreement_intervals=sum(len(v) for v in zero.values()),
                  disagreement_bases=sum(e - s for v in zero.values() for s, e in v),
                  cross_source=dict(bases_labelled_by_both=both_sources_bases,
                                    disagreement_intervals=sum(len(v) for v in cross_zero.values()),
                                    disagreement_bases=sum(e - s for v in cross_zero.values() for s, e in v),
                                    rule='A48: OrthoDB vs Swiss-Prot disagreement zeroes all route-B phase channels'),
                  unit='bases per (chrom, position); protein-level comparison, see docstring')
    with open(args.qc, 'x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--species', required=True)
    p.add_argument('--route-b', type=Path, required=True,
                   help='est_protein_hsp TSV (b6_hsp_project.py); bound by sha256 to the protein C0, rows are read '
                        'from the C0')
    p.add_argument('--miniprot-gff', type=Path, required=True, help='bound by sha256 to the protein C0')
    p.add_argument('--accepted', type=Path, required=True, help='accept/<sp>.alignments.tsv (bound by content)')
    p.add_argument('--mapping-ambiguous', type=Path, required=True, help='bound by content')
    p.add_argument('--est-c0', type=Path, required=True,
                   help="the species' published EST C0 DB (bound by sha256 to the protein C0's inputs.est_c0)")
    p.add_argument('--protein-c0', type=Path, required=True,
                   help="the species' published protein C0 DB: route-B rows, protein rule 4(b) states, input binding")
    p.add_argument('--population', choices=POPULATIONS, default='train_track',
                   help='plan §7.4 D4: one zero list per population (train_track default = original behaviour)')
    p.add_argument('--out-intervals', type=Path, required=True)
    p.add_argument('--qc', type=Path, required=True)
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z]+', args.species):
        raise ValueError('invalid species')
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
