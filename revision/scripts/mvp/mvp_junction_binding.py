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
    PROTOCOL_B1 A47 (v1.37) and `mvp_score.require_coverage_policy` accepts it,
    and even then only WITHIN one source: A-ONT1 is the sole structural source
    (A-HiFi has no downloadable FLNC product and none is produced here), and an
    unavailable source stays missing rather than contributing zero counts.
    Single-run binding stays available for preparatory work.

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
    return RunTable(run_id=run_id, source=source, use_uf=use_uf,
                    table_path=table_path, table_sha256=_sha256(table_path),
                    rows=tuple(rows))


def bind(runs, policy=None):
    """Bound junction rows: one validated run, or runs pooled under an approved policy.

    Without a policy this stays single-run, because read-count thresholds make
    pooling a coverage decision rather than arithmetic. With the A47 contract
    the runs of one structural source are pooled by summing their read counts
    per junction; per-run rows are unchanged and still available to the caller.
    """
    runs = list(runs)
    _require(runs, "no junction table supplied; missing evidence is not negative evidence")
    identities = [(r.run_id, r.table_sha256) for r in runs]
    _require(len(set(identities)) == len(identities),
             "duplicate run identity; file labels do not establish distinct runs")
    if policy is None:
        _require(len(runs) == 1,
                 "pooling two or more runs requires an approved coverage policy; "
                 "pass the A47 contract to bind(runs, policy=...)")
        return list(runs[0].rows)
    contract = _SCORE.require_coverage_policy(policy)
    approved = set(contract["structural_sources"])
    sources = sorted({r.source for r in runs})
    _require(set(sources) <= approved,
             f"run source(s) {sorted(set(sources) - approved)} carry no structural authority "
             f"under coverage policy {contract['policy_id']}")
    _require(len(sources) == 1,
             f"cross-source pooling is not permitted under {contract['pooling']!r}; got {sources}")
    return pool(runs)


def pool(runs):
    """Sum read counts per junction across runs and re-derive `supported`."""
    thresholds = _TABLE.SOURCE_THRESHOLDS
    pooled = {}
    for run in runs:
        for row in run.rows:
            key = (row["chrom"], row["intron_start0"], row["intron_end0"], row["tx_strand"])
            acc = pooled.setdefault(key, dict(row, n_ONT=0, n_HiFi=0))
            acc["n_ONT"] += row["n_ONT"]
            acc["n_HiFi"] += row["n_HiFi"]
    for row in pooled.values():
        row["supported"] = (row["n_HiFi"] >= thresholds["HiFi"]
                            or row["n_ONT"] >= thresholds["ONT"])
    return [pooled[key] for key in sorted(pooled)]


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
    return {
        "panel_sha256": mvp_windows.PANEL_SHA256,
        "thresholds": dict(_TABLE.SOURCE_THRESHOLDS),
        "counts_are": COUNTS_ARE,
        "coverage_policy": contract or "NONE_SUPPLIED_SINGLE_RUN_ONLY",
        "pooled_across_runs": len(runs) > 1,
        "run_provenance_status": "UNVERIFIED",
        "junctions": len(rows),
        "supported_junctions": len(supported_set(rows)),
        "runs": [dict(run=r.run_id, source=r.source, use_uf=r.use_uf,
                      table_sha256=r.table_sha256) for r in runs],
    }
