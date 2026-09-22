#!/usr/bin/env python3
"""Bind independently produced junction tables to MVP structural counts.

`mvp_junction_table.py` writes one read-supported junction table per aligned
long-read run; `mvp_score.junction_counts` consumes a `supported` junction set
per panel locus. Nothing joined the two, so the structural counts
(`predicted`, `precision_tp`, `supported`, `recall_tp`) had no producer bound
to actual independent evidence. This module is that join and nothing else.

What it establishes:
  * a junction table is accepted only when the run's own `alignment.json`
    receipt claims it by name and MD5, so a table cannot be swapped for a
    friendlier one after the fact;
  * the receipt's frozen panel identity matches `mvp_windows.PANEL_SHA256`;
  * the run ID is a known long-read run and its source (ONT/HiFi) agrees with
    the run inventory, because a filename never establishes run identity;
  * support is re-derived from the table's read counts through the frozen
    thresholds in `mvp_junction_table`, never from a file's own boolean.

What it deliberately refuses:
  * pooling two or more runs without an explicit approved coverage policy.
    Support thresholds are READ counts, so pooling changes which junctions are
    supported. `bind` pools only when the caller states the contract frozen by
    PROTOCOL_B1 A47.2 (v1.38) and `mvp_score.require_coverage_policy` accepts
    it, and even then only WITHIN one source: A-ONT1 runs pool into the ONT
    column (threshold 3), A-CCS23 runs into the PacBio column labelled HiFi
    (threshold 2). Counts are never summed across sources; a junction is
    supported when EITHER source's own pooled count clears its own threshold
    (the frozen disjunct). The dataset A-HiFi (ERR13994458) is unavailable and
    stays missing rather than contributing zero counts. A-CCS23 supports
    junctions only here, never TES / 3'-end boundaries (none are judged).
    Single-run binding stays available for preparatory work.
  * a run table carrying read counts in another source's column, since that
    would smuggle cross-source counts into a within-source pool.

Counts are READS (UMI status unknown; never molecules). Nothing here decodes a
model, runs GFFCompare, verifies a reference genome, or makes any evaluation,
coverage or training decision; run provenance stays UNVERIFIED exactly as the
upstream driver reports it.
"""
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

_MVP = Path(__file__).resolve().parent
_REPO = _MVP.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import mvp_windows  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _MVP / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_TABLE = _load("mvp_junction_table")
_SCORE = _load("mvp_score")
_DRIVER = _load("mvp_align_panel_longreads")

RUN_SOURCES = {run: "ONT" for run in _DRIVER.ONT_RUNS}
RUN_SOURCES.update({run: "HiFi" for run in _DRIVER.HIFI_RUNS})
SOURCE_COLUMNS = {"ONT": "n_ONT", "HiFi": "n_HiFi"}

COUNTS_ARE = "reads (UMI status unknown; never molecules)"


class JunctionBindingError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise JunctionBindingError(message)


@dataclass(frozen=True)
class RunTable:
    run_id: str
    source: str
    use_uf: bool
    table_path: Path
    table_sha256: str
    rows: Tuple[dict, ...]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(table_path, report_path):
    """Validate one run's junction table against its own alignment receipt."""
    table_path, report_path = Path(table_path), Path(report_path)
    report = json.loads(report_path.read_text())

    run_id = report.get("run")
    _require(run_id not in _DRIVER.UNAVAILABLE_RUNS,
             f"{run_id} ({_DRIVER.UNAVAILABLE_RUNS.get(run_id)}) is unavailable under A47.2; "
             "missing per source, never zero")
    _require(run_id in RUN_SOURCES, f"unknown long-read run {run_id!r}")
    source = report.get("source")
    _require(source == RUN_SOURCES[run_id],
             f"source {source!r} contradicts the run inventory for {run_id}")
    _require(report.get("panel_sha256") == mvp_windows.PANEL_SHA256,
             "alignment receipt panel_sha256 drifted from the frozen panel")
    use_uf = report.get("use_uf")
    _require(isinstance(use_uf, bool), "alignment receipt must state the audited use_uf decision")

    claimed = (report.get("outputs_md5") or {}).get(table_path.name)
    _require(claimed is not None,
             f"alignment receipt does not claim output {table_path.name}")
    actual = hashlib.md5(table_path.read_bytes()).hexdigest()
    _require(claimed == actual,
             f"junction table md5 {actual} does not match the receipt md5 {claimed}")

    rows = _TABLE.read_table(table_path)
    foreign = [c for s, c in SOURCE_COLUMNS.items() if s != source]
    _require(all(row[c] == 0 for row in rows for c in foreign),
             f"run {run_id} ({source}) table carries read counts in another source's column; "
             "counts are never mixed across sources")
    return RunTable(run_id=run_id, source=source, use_uf=use_uf,
                    table_path=table_path, table_sha256=_sha256(table_path),
                    rows=tuple(rows))


def bind(runs, policy=None):
    """Bound junction rows: one validated run, or runs pooled under an approved policy.

    Without a policy this stays single-run, because read-count thresholds make
    pooling a coverage decision rather than arithmetic. With the A47.2 contract
    the runs of each structural source are pooled within that source, and the
    per-source pooled tables are combined per junction by the frozen disjunct
    (ONT >= 3 or HiFi >= 2), never by summing counts across sources. Per-run
    rows are unchanged on each RunTable; `per_source` exposes the pooled tables.
    """
    runs = _checked(runs)
    if policy is None:
        _require(len(runs) == 1,
                 "pooling two or more runs requires an approved coverage policy; "
                 "pass the A47.2 contract to bind(runs, policy=...)")
        return list(runs[0].rows)
    return combine(per_source(runs, policy))


def _checked(runs):
    runs = list(runs)
    _require(runs, "no junction table supplied; missing evidence is not negative evidence")
    identities = [(r.run_id, r.table_sha256) for r in runs]
    _require(len(set(identities)) == len(identities),
             "duplicate run identity; file labels do not establish distinct runs")
    run_ids = [r.run_id for r in runs]
    _require(len(set(run_ids)) == len(run_ids),
             "the same run supplied twice with different tables; pooling it would double-count reads")
    return runs


def per_source(runs, policy):
    """{source: rows pooled WITHIN that source} under the approved contract."""
    runs = _checked(runs)
    contract = _SCORE.require_coverage_policy(policy)
    approved = set(contract["structural_sources"])
    sources = {r.source for r in runs}
    _require(sources <= approved,
             f"run source(s) {sorted(sources - approved)} carry no structural authority "
             f"under coverage policy {contract['policy_id']}")
    return {source: pool([r for r in runs if r.source == source])
            for source in sorted(sources)}


def pool(runs):
    """Sum read counts per junction across runs of ONE source; re-derive `supported`."""
    runs = list(runs)
    sources = {r.source for r in runs}
    _require(len(sources) == 1,
             f"cross-source pooling is not permitted (within_source); got {sorted(sources)}")
    column = SOURCE_COLUMNS[sources.pop()]
    pooled = {}
    for run in runs:
        for row in run.rows:
            key = (row["chrom"], row["intron_start0"], row["intron_end0"], row["tx_strand"])
            acc = pooled.setdefault(key, dict(row, n_ONT=0, n_HiFi=0))
            acc[column] += row[column]
    for row in pooled.values():
        row["supported"] = _supported(row)
    return [pooled[key] for key in sorted(pooled)]


def combine(pooled_by_source):
    """Per-junction union of per-source pooled tables; support is a per-source OR.

    Each output column comes from exactly one source's pool, so n_ONT and n_HiFi
    are never added together; `supported` is re-derived from the frozen
    thresholds, i.e. EITHER source clearing its own threshold.
    """
    combined = {}
    for source, rows in pooled_by_source.items():
        column = SOURCE_COLUMNS[source]
        for row in rows:
            key = (row["chrom"], row["intron_start0"], row["intron_end0"], row["tx_strand"])
            acc = combined.setdefault(key, dict(row, n_ONT=0, n_HiFi=0))
            acc[column] = row[column]
    for row in combined.values():
        row["supported"] = _supported(row)
    return [combined[key] for key in sorted(combined)]


def _supported(row):
    thresholds = _TABLE.SOURCE_THRESHOLDS
    return row["n_HiFi"] >= thresholds["HiFi"] or row["n_ONT"] >= thresholds["ONT"]


def supported_set(rows):
    """Junction keys whose read counts clear the frozen support thresholds."""
    thresholds = _TABLE.SOURCE_THRESHOLDS
    return frozenset(
        (r["chrom"], r["intron_start0"], r["intron_end0"], r["tx_strand"])
        for r in rows
        if r["n_ONT"] >= thresholds["ONT"] or r["n_HiFi"] >= thresholds["HiFi"]
    )


def locus_counts(predicted_by_locus, supported, loci):
    """Structural counts per panel locus, or None everywhere when unsupported.

    `supported=None` means the independent evidence is absent. It is reported
    as None so the scorer suppresses pooled P/R; it is never filled with zero.
    """
    counts = {}
    for locus in loci:
        predicted = predicted_by_locus.get(locus.gene_id)
        _require(predicted is not None, f"no predicted junctions supplied for locus {locus.gene_id}")
        counts[locus.gene_id] = _SCORE.junction_counts(predicted, supported, locus)
    return counts


def receipt(runs, rows, policy=None):
    runs = list(runs)
    contract = _SCORE.require_coverage_policy(policy) if policy is not None else None
    result = {
        "panel_sha256": mvp_windows.PANEL_SHA256,
        "thresholds": dict(_TABLE.SOURCE_THRESHOLDS),
        "counts_are": COUNTS_ARE,
        "coverage_policy": contract or "NONE_SUPPLIED_SINGLE_RUN_ONLY",
        "pooled_across_runs": len(runs) > 1,
        "run_provenance_status": "UNVERIFIED",
        "junctions": len(rows),
        "supported_junctions": len(supported_set(rows)),
        "runs": [dict(run=r.run_id, source=r.source, dataset=_DRIVER.RUN_DATASETS[r.run_id],
                      use_uf=r.use_uf, table_sha256=r.table_sha256,
                      supported_junctions=len(supported_set(r.rows))) for r in runs],
    }
    if contract is not None:
        bound = sorted({r.source for r in runs})
        pooled = per_source(runs, policy)
        result.update(
            structural_sources_bound=bound,
            # an approved source with no run bound is MISSING, never zero support
            structural_sources_missing=[s for s in contract["structural_sources"] if s not in bound],
            per_source={s: dict(dataset=contract["source_sets"][s],
                                runs=sorted(r.run_id for r in runs if r.source == s),
                                junctions=len(pooled[s]),
                                supported_junctions=len(supported_set(pooled[s])))
                        for s in bound})
    return result
