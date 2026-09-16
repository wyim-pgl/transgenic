#!/usr/bin/env python3
"""RICH17 real-DIAMOND diagnostic extension for unresolved C2 semantics.

Written but NOT submitted: this runs real DIAMOND 2.1.24 on a compute job
(see est_phase_accept_fixture.sbatch); submission requires author
authorization.  It extends est_blastx_fixture.py (unchanged, reviewed at
950cf4c) rather than editing it, so the reviewed fixture's sha256 is
preserved.

Purpose (must-fix 2): empirically pin what raw column 9 ('length') counts
under `--frameshift 15` to inform the UNRESOLVED acceptance definition.
No reading is selected and no production gate is opened by this script.  For every RICH17 hit of
the synthetic frameshift fixture it:
  1. replays BTOP via est_blastx_fixture.replay() (ground truth against input
     codons, CIGAR and all three rich sequence fields);
  2. challenges candidate_decompose_hsp() with the raw13 subset of the row;
  3. requires exact agreement on frameshift count/net and ungapped aa; and
  4. reports, per row, which candidate reading (A col9-as-alignment-columns,
     B ungapped query aa, C per-frame sub-span) reproduces column 9.
Diagnostics are written to btop_pin_report.json; even a unique reading only
means CANDIDATE_REQUIRES_REVIEW. Ambiguous readings are INCONCLUSIVE. Both
C2 and C3 remain blocked until real-engine semantic review and A46 approval.
Outer DONE means diagnostic execution completed, never acceptance approval.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import signal
import sys


def _load(name):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module


fixture = _load('est_blastx_fixture')
accept = _load('est_phase_accept')
launcher = fixture.launcher


def pin_rows(raw17_path, queries, protein):
    """Cross-check decompose_hsp against replay() and tally col-9 readings."""
    rows = []
    with open(raw17_path) as f:
        for line in f:
            row = line.rstrip('\n').split('\t')
            truth = fixture.replay(row, queries[row[0]], protein)
            qlen, qs, qe, frame = int(row[1]), int(row[4]), int(row[5]), int(row[11])
            subspans, ungapped_aa, shifts, subject_cols = accept.candidate_decompose_hsp(
                qlen, qs, qe, frame, row[12])
            fixture.require(shifts == len(truth['frameshifts']),
                            f'{row[0]}: frameshift count disagrees with replay')
            net = sum(s['delta_nt'] for s in truth['frameshifts'])
            our_net = sum((1 if row[12][i] == '\\' else -1)
                          for i in range(len(row[12])) if row[12][i] in '\\/')
            fixture.require(our_net == net, f'{row[0]}: net frameshift disagrees')
            fixture.require(ungapped_aa == truth['translated_residues'],
                            f'{row[0]}: ungapped aa disagrees with translated field')
            fixture.require(len(subspans) == shifts + 1,
                            f'{row[0]}: sub-span count != frameshifts + 1')
            col9 = int(row[8])
            gapped_cols = len(row[14])           # includes gap AND frameshift columns
            readings = {
                'A_alignment_columns_incl_shift': gapped_cols,
                'A_alignment_columns_excl_shift': gapped_cols - shifts,
                'B_ungapped_query_aa': ungapped_aa,
                'C_max_subspan_aa': max(s['subspan_aa'] for s in subspans),
                'C_sum_subspan_aa': sum(s['subspan_aa'] for s in subspans)}
            rows.append(dict(qseqid=row[0], col9=col9, frameshift_ops=shifts,
                             readings=readings,
                             col9_matches=sorted(k for k, v in readings.items() if v == col9)))
    fixture.require(rows, 'no RICH17 rows to pin')
    fixture.require(any(r['frameshift_ops'] for r in rows),
                    'fixture elicited no frameshift; BTOP semantics remain unpinned')
    return dict(rows=rows, **summarize_readings(rows))


def summarize_readings(rows):
    """Agreement of readings is diagnostic, never semantic or author approval."""
    common = set(rows[0]['col9_matches']) if rows else set()
    for r in rows[1:]:
        common &= set(r['col9_matches'])
    return dict(consistent_readings=sorted(common),
                verdict='CANDIDATE_REQUIRES_REVIEW' if len(common) == 1 else 'INCONCLUSIVE',
                production_authorized=False)


def run(diamond, out_dir, tmpdir, threads):
    out_dir = Path(out_dir)
    with launcher.work_directory(out_dir) as out:
        base = out / 'engine_fixture'
        fixture.run_fixture(diamond, base, tmpdir, threads)
        data = fixture.fixture_data()
        reports = {}
        for kind in ('orthodb', 'swissprot'):
            raw = base / kind / f'{kind}.raw17.tsv'
            reports[kind] = pin_rows(raw, data['queries'], data['protein'])
            reports[kind]['raw17_sha256'] = launcher.sha256(raw)
        launcher.write_json(out / 'btop_pin_report.json', dict(
            purpose='diagnose raw13 col 9 semantics; NOT approval',
            status='UNRESOLVED_PENDING_REVIEW', production_authorized=False,
            acceptance_definition_under_test=accept.ACCEPTANCE_DEFINITION,
            extension_sha256=launcher.sha256(__file__),
            engine_provenance_sha256=launcher.sha256(base / 'provenance.json'),
            base_fixture_sha256=launcher.sha256(fixture.__file__),
            accept_module_sha256=launcher.sha256(accept.__file__),
            reports=reports))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diamond', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--tmpdir', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args(argv)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.diamond, args.out_dir, args.tmpdir, args.threads)
    return 0


if __name__ == '__main__':
    sys.exit(main())
