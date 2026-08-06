#!/usr/bin/env python3
"""Run TransGenic completion mode over the three staged external annotations.

Why this script exists. Every published completion-mode result prompts TransGenic with
TAIR10's own primary transcript, which the model returns unchanged at ~99.4% of loci —
there is nothing to repair, so polishing was never measurable. Task 1 staged three
external annotations (GeMoMa, BRAKER3, EGAPx) that genuinely disagree with TAIR10; this
script prompts the model with each of them, in the one invocation that reproduces the
published benchmark, and hands `predictions/{tool}_completed.gff3` to Task 5's scorer.

This is not the naive two-step "run the model, write a file" the original task brief
sketched (`31_run_polishing_benchmark.sh`). Everything below was learned the hard way,
by actually running the pipeline on the GPU host, and is recorded in
`.superpowers/sdd/2026-08-05-polishing-benchmark/gpu-environment.md`:

1. `--batch-size 1` is non-negotiable. Batched generation pads unequal-length GSF
   prefixes and the model consumes the padding as content: bs1 loses 0/18 loci, bs2
   loses 1/18, bs4 loses 3/18, and bs32 — the CLI default on the GPU host's pinned
   commit `5d9929e` — loses 18/18 while still printing "Output written to:" and exiting
   0. So `--batch-size` is a hardcoded constant here, not a CLI flag: nothing about this
   script lets a caller silently reintroduce that loss.
2. The GPU host (`gpu` / pgl-gpu) has no GPFS mount and runs a pinned checkout of the
   package (commit `5d9929e`) that must not be updated — a newer checkout of
   `examples/prompt_mode.py` was measured to produce zero output on the same inputs.
   This script only ever invokes the host's *existing* `examples/prompt_mode.py`
   over SSH; it never touches the host's source tree.
3. `examples/prompt_mode.py` must run from a directory containing a populated
   `HFmodels/` (a hardcoded relative `cache_dir="./HFmodels"` inside
   `datasets.py:189`, combined with the script forcing itself offline) — so every
   remote command below does `cd ~/transgenic` first.
4. Raw `prompt_mode.py` output is not what gets scored. The published
   `standardized_results/` went through `transgenic_comparison/standardize_gff.py`
   (coordinate validation against the genome's `.fai`, 500 kb gene / 100 kb mRNA span
   caps, interval merge) — step 2 below.
5. `--num-beams 2` is the published Methods setting (`num_beams=2, do_sample=False`).
   **Correction (fix round 1):** the pinned host's `prompt_mode.py` calls
   `model.generate(..., num_return_sequences=1)`, so it does *not* export both beam
   hypotheses as separate gene records the way an earlier draft of this file assumed —
   a clean 18-locus run produced 18 gene rows, not 36. The top-beam filter (step 3
   below) is therefore a no-op safety net on this invocation, not a load-bearing step;
   kept because it is free insurance against a future checkpoint/invocation that does
   export both beams, but Task 5 must not assume two records per locus from this
   pipeline's output.
6. **(Round 3) `genome2GSFDataset` (`src/transgenic/datasets/preprocess.py`) drops the
   last gene of every file it processes, silently.** It finalizes and inserts a gene
   into the database only when it encounters the *next* `gene` line; the file loop ends
   and `con.close()` runs with no final flush, so the last gene in file order is never
   inserted, with zero parse errors, before generation is ever reached. Deterministic
   and content-independent — confirmed against two real runs (GeMoMa's and EGAPx's
   last ChrM gene were each exactly the locus found missing). This script accounts for
   it (see `build_local_subset`'s `structurally_unreachable_locus` and
   `run_remote_chunk`'s `reachable_genes_in`) rather than patching `preprocess.py` on
   the pinned host, which would break comparability with the published predictions.

7. **(Round 4) `genome2GSFDataset` appends to an existing DuckDB.** Pointing `--db` at
   a path that already exists adds the new loci alongside the ones already there,
   silently, and generation then runs over the union. Measured on the real host while
   verifying the resume fix: a second run of the `egapx/ChrM` chunk emitted 22 gene rows
   for 11 loci — exactly 2 per `GM=` — and its log reads `Generating predictions... 22`
   for a 12-gene chunk. Every gate in this script measures *loss*, so an inflated count
   passes all of them, and a re-run that genuinely produced only half its loci would be
   topped back up to a passing number by the stale half. `build_prompt_mode_command`
   therefore deletes the remote `.db` and raw output before each run, and
   `run_remote_chunk` fails a chunk whose gene-row count differs from its distinct
   `GM=` count.

This is the fourth defect this benchmark has surfaced in `prompt_mode.py` and its
dependencies, alongside point 1 (the batch-32 default that silently zeroes all output),
point 3 (the `cache_dir="./HFmodels"` working-directory trap) and point 6 (the silently
dropped last gene) — all four are worth reporting upstream on their own, independent of
this benchmark.

Pipeline (per tool): prompt_mode (remote, GPU) -> standardize_gff -> top-beam filter.
Each tool runs one chromosome at a time (Chr1-5, ChrC, ChrM) so that a run that dies
partway through does not have to restart: a chromosome chunk whose provenance record
already says "ok" is skipped on the next invocation, and only the remaining chromosomes
run. Full-locus resume (mid-chromosome) is out of scope — `prompt_mode.py` itself has no
such mechanism, and a single chromosome is a small enough unit (minutes to ~2h) that
losing an in-progress one to a crash is an acceptable, disclosed granularity.

Every step that can fail loudly does: a chunk whose output covers far fewer genes than
its input aborts the tool's run (not just that chunk) rather than being merged into a
result that looks complete; a value that cannot be computed (e.g. the host's git commit,
if SSH fails) is recorded as "N/A (<reason>)", never silently as 0 or omitted.

Usage:
    python 33_run_polishing_benchmark.py --tool gemoma braker3 egapx
    python 33_run_polishing_benchmark.py --tool egapx --chromosomes ChrM   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# --------------------------------------------------------------------------------------
# Constants. Everything that gpu-environment.md marks as "non-negotiable" is a constant
# here, not a CLI flag, so no invocation of this script can silently reintroduce the loss
# those measurements found.
# --------------------------------------------------------------------------------------

HOST = "gpu"  # ~/.ssh/config: pgl-gpu, 134.197.50.123
REMOTE_TRANSGENIC_DIR = "~/transgenic"
REMOTE_BENCH_DIR = "~/polishing_benchmark"
REMOTE_GENOME = f"{REMOTE_BENCH_DIR}/genome/Athaliana_167_TAIR10.fa"
REMOTE_INPUTS_DIR = f"{REMOTE_BENCH_DIR}/inputs"
REMOTE_PREDICTIONS_DIR = f"{REMOTE_BENCH_DIR}/predictions"

MODEL = "jlomas/HyenaTransgenic-768L12A6-400M"
BATCH_SIZE = 1  # non-negotiable; see module docstring point 1.
NUM_BEAMS = 2  # Methods: num_beams=2, do_sample=False
MAX_LENGTH = 2048
DEVICE = "cuda"

TOOLS = ("gemoma", "braker3", "egapx")
CHROMOSOMES = ("Chr1", "Chr2", "Chr3", "Chr4", "Chr5", "ChrC", "ChrM")
SCRATCH_PREFIXES = ("_smoke", "_bs")  # the team lead's GPU-host scratch files

EXPECTED_GENOME_MD5 = "ac1d3ca8af4f02bca3d750a339b65fec"
# Constraint #2 (gpu-environment.md): the host must stay at this exact commit. Syncing
# forward to GPFS HEAD was measured to produce zero output on every input (both GeMoMa
# and TAIR10 smoke runs); this commit is the one verified to reproduce the published
# prediction. Enforced, not just recorded — see get_and_verify_host_commit (I4).
EXPECTED_HOST_COMMIT = "5d9929ea2189e653aac5fc7e2fef234651e96ae3"
DEFAULT_MAX_LOSS_FRACTION = 0.05  # published run lost 3/27,416 = 0.01%; 5% is a generous
                                  # "something is clearly wrong" trigger, not a target.
                                  # Raising it above this default requires an explicit,
                                  # provenance-recorded acknowledgement (C3) — see main().
# R4-3: the ratio gate above cannot see a single genuinely-missing locus on any but the
# smallest chromosome, so the exact per-locus count is gated separately, and its default
# is zero: every locus prompted must come back, except the one per chunk known to be
# structurally unreachable. Raising it needs the same acknowledgement C3 requires, for
# the same reason — a loosened gate must never be invisible to a future reader.
DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI = 0

# R4-3: unexplained loss is a different quantity from total loss and needs its own,
# much tighter scale. `reachable_genes_in` already excludes the one locus per chunk that
# genome2GSFDataset structurally cannot emit, so anything still missing is either a real
# generation failure or something we do not understand — and the published run shows what
# "normal" looks like there: 3 of 27,416 loci absent, one of them the structural drop, so
# 2 genuinely unexplained, 0.007%. A gate at 0.05 would never fire on that scale; a gate
# at zero would abort a *correct* 19-hour run at the very end with no way forward short of
# editing this file. 0.001 sits between: it tolerates ~27 unexplained loci genome-wide,
# roughly thirteen times the published rate, while still catching a systematic failure.
DEFAULT_MAX_UNEXPLAINED_FRACTION = 0.001

# I10/N5: top-level feature types that are never protein-coding gene loci (EGAPx carries
# 2,048 lnc_RNA + 378 pseudogene rows in the staged A. thaliana file), and the gbkey
# values that flag an otherwise-generic "transcript"-typed row as non-coding (77 such
# rows, all misc_RNA). Both match 32_score_polishing.py's own NONCODING_TYPES/
# NONCODING_GBKEYS exactly — see build_local_subset() — so the denominator this script
# counts reconciles with what Tasks 2/3 already count, instead of silently diverging.
NONCODING_TOP_LEVEL_TYPES = frozenset({"lnc_RNA", "pseudogene"})
NONCODING_GBKEYS = frozenset({"misc_RNA", "ncRNA"})

ROOT = Path(__file__).resolve().parents[3]  # /data/gpfs/assoc/pgl/data/Transgenic
BENCH = ROOT / "polishing_benchmark"
LOCAL_INPUTS = BENCH / "inputs"
LOCAL_PREDICTIONS = BENCH / "predictions"
LOCAL_CHUNKS = LOCAL_PREDICTIONS / "_chunks"
# I8: gpu-environment.md specifies transgenic_comparison/genomes specifically (not the
# top-level genomes/ directory) — the two .fai files are identical today, but pointing
# at the one actually named in the spec removes a "true today, not asserted" dependency.
LOCAL_GENOME_DIR = ROOT / "transgenic_comparison" / "genomes"
STANDARDIZE_SCRIPT = ROOT / "transgenic_comparison" / "standardize_gff.py"
BEAM1_FILTER_SCRIPT = Path(__file__).resolve().parent / "13_beam1_filter.py"


# --------------------------------------------------------------------------------------
# Scratch-file guard
# --------------------------------------------------------------------------------------

def is_scratch_name(name: str) -> bool:
    """True if a path's basename looks like the team lead's GPU-host scratch files.

    Anything beginning `_smoke` or `_bs` on the host is explicitly out of bounds
    ("ignore them, and have the runner refuse to treat them as inputs").
    """
    base = Path(str(name)).name
    return any(base.startswith(prefix) for prefix in SCRATCH_PREFIXES)


def assert_not_scratch(path: object) -> None:
    if is_scratch_name(str(path)):
        raise ValueError(f"refusing to treat a scratch file as a pipeline path: {path}")


# --------------------------------------------------------------------------------------
# GFF3 attribute parsing — a small, self-contained helper (not imported from
# 32_score_polishing.py, to keep this script's only dependency on that module at the
# CLI/subprocess boundary, not a Python import boundary).
# --------------------------------------------------------------------------------------

_ATTR_RE_CACHE: dict[str, re.Pattern] = {}


def _gff_attr(attributes: str, key: str) -> str | None:
    """Read `key=value` out of a GFF3 attributes column, anchored on the key itself so
    it never matches inside a different attribute name that happens to end in `key`."""
    pattern = _ATTR_RE_CACHE.get(key)
    if pattern is None:
        pattern = _ATTR_RE_CACHE[key] = re.compile(rf"(?:^|;)\s*{key}=([^;\n]+)")
    m = pattern.search(attributes)
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------
# Parsing stderr/stdout text from prompt_mode.py
# --------------------------------------------------------------------------------------

_PARSE_ERROR_RE = re.compile(r"Parsing errors:\s*(\d+)\s*/\s*(\d+)\s*sequences skipped")
_OUTPUT_WRITTEN_RE = re.compile(r"Output written to:\s*(\S+)")


@dataclass(frozen=True)
class ParsingErrors:
    """`Parsing errors: N/M sequences skipped`, or the fact that the line never appeared.

    prompt_mode.py only prints this line when error_count > 0, so "not found" is
    ambiguous between "zero errors" and "the process never reached its summary block" —
    callers must check `output_written` (see below) to tell those apart, not this alone.
    """
    skipped: int
    total: int
    found: bool


def parse_parsing_errors(text: str) -> ParsingErrors:
    """Find the parse-error summary in combined stdout+stderr text.

    Both streams are searched (concatenated by the caller) because the version of
    prompt_mode.py pinned on the GPU host predates this repo's current source, and
    which stream it prints its summary lines to has not been independently confirmed
    for that exact commit — see gpu-environment.md. Searching both is strictly safer
    than guessing.
    """
    m = _PARSE_ERROR_RE.search(text)
    if not m:
        return ParsingErrors(skipped=0, total=0, found=False)
    return ParsingErrors(skipped=int(m.group(1)), total=int(m.group(2)), found=True)


def parse_output_written(text: str) -> str | None:
    """Find the `Output written to: <path>` marker. Present even for a zero-gene output
    (the batch-size-32 defect), so it confirms the process reached its end, not success.
    """
    m = _OUTPUT_WRITTEN_RE.search(text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------
# GFF3 counting helpers
# --------------------------------------------------------------------------------------

def count_gff3_features(path: Path, feature: str = "gene") -> int:
    """Count top-level rows of `feature` type in a GFF3 file. 0 for a missing file."""
    if not path.exists():
        return 0
    n = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3 and cols[2] == feature:
                n += 1
    return n


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Loss detection — "fail loudly" per the brief
# --------------------------------------------------------------------------------------

class LossTooHigh(RuntimeError):
    """Raised when a step's output covers far fewer loci than its input."""


def check_loss(genes_in: int, genes_out: int, max_loss_fraction: float, context: str) -> None:
    """Raise if the fraction of input genes missing from the output exceeds the
    threshold. `genes_in == 0` is itself an error (nothing to measure a loss rate
    against), not a silent pass.
    """
    if genes_in <= 0:
        raise LossTooHigh(f"{context}: genes_in is {genes_in} (nothing was staged?)")
    lost = genes_in - genes_out
    fraction = lost / genes_in
    if fraction > max_loss_fraction:
        raise LossTooHigh(
            f"{context}: {genes_out}/{genes_in} genes survived "
            f"({fraction:.1%} lost, threshold {max_loss_fraction:.1%}) — "
            f"stopping rather than writing a result that looks complete"
        )


# --------------------------------------------------------------------------------------
# Remote command construction (pure — testable without SSH)
# --------------------------------------------------------------------------------------

def build_remote_env_setup() -> str:
    return (
        "export MAMBA_ROOT_PREFIX=$HOME/micromamba && "
        'eval "$($HOME/bin/micromamba shell hook -s bash)" && '
        "micromamba activate transgenic"
    )


def build_local_subset(tool: str, chrom: str, input_path: Path, subset_path: Path) -> dict:
    """Build the per-(tool, chromosome) input subset locally, before ever touching the
    GPU host (I10).

    This replaced a remote `awk '$1==c'` one-liner that forwarded *every* row for the
    chromosome — including EGAPx's `lnc_RNA`/`pseudogene` rows and their children —
    into the host's `genome2GSFDataset`. That preprocessing step does not filter by
    feature type, so the database the host built could contain more loci than this
    script's own denominator, and the two numbers could never reconcile with Task 2/3's
    convention of excluding these types entirely. Filtering here, in local,
    unit-testable Python, closes that gap and removes an SSH-only code path with no test
    coverage of its own. (R4-1: this function's own "genes" count is now also the single
    source of truth `ensure_chunk` uses to decide whether a chunk needs to be re-run —
    an earlier, separate raw-count function used for that check disagreed with this
    one for every EGAPx chromosome, permanently defeating resume for that tool; it has
    been removed rather than reconciled, so there is exactly one definition of "how many
    genes are in this chunk" from now on.)

    Three passes:
    1. Exclude explicit non-coding top-level loci (`lnc_RNA`/`pseudogene`) and
       `transcript`-typed rows flagged non-coding by `gbkey` (N5) — matching
       `32_score_polishing.py`'s `NONCODING_TYPES`/`NONCODING_GBKEYS` exactly.
    2. Transitively pull in descendants of anything excluded in pass 1 (a fixed point
       over `Parent` chains — real GFF3 hierarchies are only 2-3 levels deep).
    3. (N1) Drop a `gene` row that has no surviving `mRNA`/`transcript` child after
       passes 1-2. EGAPx nests `gene(gbkey=lncRNA) -> lnc_RNA -> exon`: once the
       `lnc_RNA` is excluded in pass 1, the gene becomes a childless shell the model
       can never be prompted from — counting it into `genes_in` would bake a
       guaranteed, unmeasured loss into the denominator before inference even starts.
       Measured on the real staged EGAPx file: 6-7% of nuclear-chromosome gene rows
       are exactly this shell, enough on its own to trip a 5% loss gate before a
       single locus is generated.

    Returns `{"genes": <int>, "excluded_feature_counts": {type: count}, "
    structurally_unreachable_locus": <str | None>}`. A dropped childless gene is folded
    into `excluded_feature_counts["gene"]` alongside any other excluded row type, not
    tracked separately, since it is counted the same way.

    `structurally_unreachable_locus` (N6, round 3) is the identifier of the *last* kept
    `gene` row in this subset, in file order — the one locus `genome2GSFDataset` will
    silently drop before generation regardless of content (see the inline comment at
    the write loop below). This is not excluded from the subset itself (the host still
    needs to see it, in case a future fix changes this) — callers use it to adjust the
    denominator `check_loss` gates on and to annotate the missing-loci manifest with a
    reason, rather than surprise a perfect run with a "1 locus lost" result on every
    single chromosome.
    """
    assert_not_scratch(input_path)
    assert_not_scratch(subset_path)
    rows: list[list[str]] = []
    with input_path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[0] != chrom:
                continue
            rows.append(cols)

    def _expand_to_descendants(excluded: set[str]) -> None:
        changed = True
        while changed:
            changed = False
            for cols in rows:
                parent = _gff_attr(cols[8], "Parent")
                if not parent or parent not in excluded:
                    continue
                fid = _gff_attr(cols[8], "ID")
                if fid and fid not in excluded:
                    excluded.add(fid)
                    changed = True

    # Pass 1: explicit non-coding top-level loci + gbkey-flagged "transcript" rows.
    id_gbkey: dict[str, str] = {}
    for cols in rows:
        fid = _gff_attr(cols[8], "ID")
        if fid:
            gbkey = _gff_attr(cols[8], "gbkey")
            if gbkey:
                id_gbkey[fid] = gbkey

    excluded_ids: set[str] = set()
    for cols in rows:
        fid = _gff_attr(cols[8], "ID")
        noncoding_type = cols[2] in NONCODING_TOP_LEVEL_TYPES
        noncoding_gbkey = cols[2] == "transcript" and id_gbkey.get(fid) in NONCODING_GBKEYS
        if fid and (noncoding_type or noncoding_gbkey):
            excluded_ids.add(fid)

    # Pass 2: descendants of anything pass 1 excluded.
    _expand_to_descendants(excluded_ids)

    # Pass 3 (N1): genes left with no surviving mRNA/transcript child.
    gene_ids = {_gff_attr(cols[8], "ID") for cols in rows if cols[2] == "gene"}
    gene_ids.discard(None)
    genes_with_transcript: set[str] = set()
    for cols in rows:
        if cols[2] not in ("mRNA", "transcript"):
            continue
        fid = _gff_attr(cols[8], "ID")
        if fid in excluded_ids:
            continue
        parent = _gff_attr(cols[8], "Parent")
        if parent in gene_ids:
            genes_with_transcript.add(parent)
    childless_genes = gene_ids - genes_with_transcript - excluded_ids
    excluded_ids |= childless_genes
    _expand_to_descendants(excluded_ids)  # nothing coding should be left, but re-check

    excluded_feature_counts: dict = {}
    genes = 0
    last_gene_locus: str | None = None
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w") as out:
        for cols in rows:
            fid = _gff_attr(cols[8], "ID")
            parent = _gff_attr(cols[8], "Parent")
            is_excluded = (cols[2] in NONCODING_TOP_LEVEL_TYPES
                           or (fid is not None and fid in excluded_ids)
                           or (parent is not None and parent in excluded_ids))
            if is_excluded:
                excluded_feature_counts[cols[2]] = excluded_feature_counts.get(cols[2], 0) + 1
                continue
            if cols[2] == "gene":
                genes += 1
                # N6 (round 3): genome2GSFDataset (preprocess.py) only inserts a gene
                # into the database when it encounters the *next* gene line ("finalize
                # the previous gene and start a new one"); the file loop ends and
                # con.close() runs with no final flush, so the last gene of every file
                # it processes is silently dropped, with zero parse errors, before
                # generation is ever reached. Deterministic and content-independent —
                # confirmed against two real smoke-test runs (GeMoMa ChrM's last gene
                # was exactly the one found missing; so was EGAPx ChrM's). Since this
                # subset file preserves the staged input's row order, the last kept
                # `gene` row here IS the one genome2GSFDataset will drop — knowable
                # before the run, not just after.
                last_gene_locus = _first_attribute_value(cols[8])
            out.write("\t".join(cols) + "\n")

    return {
        "genes": genes,
        "excluded_feature_counts": excluded_feature_counts,
        "structurally_unreachable_locus": last_gene_locus,
    }


def build_prompt_mode_command(remote_gff: str, remote_output: str, remote_db: str, remote_log: str) -> str:
    """The exact remote command for one (tool, chromosome) chunk.

    `--batch-size 1` and `--num-beams 2` are literal constants in this string, not
    interpolated from any caller-supplied value — see the module docstring.

    R4-4 (round 4, found while verifying R4-1 on the real host): the stale remote
    DuckDB is deleted first, because `genome2GSFDataset` *appends* to an existing one.
    Re-running a chunk against a leftover `.db` therefore generates every locus again on
    top of the ones already there — measured on the real GPU, a second run of
    `egapx/ChrM` produced 22 gene rows for 11 loci, exactly 2 per `GM=`, and the log
    confirms the cause ("Generating predictions... 22" where the chunk has 12 genes).
    Nothing downstream would have called that wrong: `check_loss` only ever looks for
    *loss*, so an inflated count sails through, and a re-run that genuinely produced
    only half its loci would be masked by the stale half back to a passing number. The
    raw output is removed for the same reason — so a re-run is a re-run, not an
    accumulation, which is the premise `ensure_chunk` is built on.
    """
    for value in (remote_gff, remote_output, remote_db, remote_log):
        assert_not_scratch(value)
    return (
        f"{build_remote_env_setup()} && cd {REMOTE_TRANSGENIC_DIR} && "
        f"rm -f {remote_db} {remote_output} && "
        f"python3 examples/prompt_mode.py "
        f"--genome {REMOTE_GENOME} --gff {remote_gff} --output {remote_output} "
        f"--db {remote_db} --model {MODEL} --batch-size {BATCH_SIZE} "
        f"--num-beams {NUM_BEAMS} --max-length {MAX_LENGTH} --device {DEVICE} "
        f"> {remote_log} 2>&1"
    )


def build_chunk_remote_command(tool: str, chrom: str) -> dict:
    """Remote paths + the inference-only command for one chunk.

    The input subset is built locally and pushed separately (build_local_subset /
    push_and_verify below) — this no longer includes a remote subsetting step (I10).
    """
    remote_subset = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_subset.gff3"
    remote_output = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_raw.gff3"
    remote_db = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}.db"
    remote_log = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_raw.log"
    return {
        "remote_subset": remote_subset,
        "remote_output": remote_output,
        "remote_db": remote_db,
        "remote_log": remote_log,
        "command": build_prompt_mode_command(remote_subset, remote_output, remote_db, remote_log),
    }


def build_ssh_argv(remote_command: str, host: str = HOST) -> list[str]:
    return ["ssh", host, remote_command]


def build_rsync_fetch_argv(remote_path: str, local_path: Path, host: str = HOST) -> list[str]:
    return ["rsync", "-az", f"{host}:{remote_path}", str(local_path)]


def build_rsync_push_argv(local_path: Path, remote_path: str, host: str = HOST) -> list[str]:
    return ["rsync", "-az", str(local_path), f"{host}:{remote_path}"]


# --------------------------------------------------------------------------------------
# Injectable execution primitives. Production code uses the `default_*` implementations;
# tests inject fakes so the whole orchestration layer below is exercisable without SSH,
# a GPU, or the GPU host being reachable.
# --------------------------------------------------------------------------------------

def default_ssh_run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True)


def default_rsync_fetch(remote_path: str, local_path: Path, host: str = HOST) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(build_rsync_fetch_argv(remote_path, local_path, host=host),
                           capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync fetch failed ({remote_path} -> {local_path}): {proc.stderr}")


def default_rsync_push(local_path: Path, remote_path: str, host: str = HOST) -> None:
    proc = subprocess.run(build_rsync_push_argv(local_path, remote_path, host=host),
                           capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync push failed ({local_path} -> {remote_path}): {proc.stderr}")


def _read_remote_md5(remote_path: str, *, ssh_run, host: str) -> str:
    proc = ssh_run(build_ssh_argv(f"md5sum {remote_path}", host=host))
    remote_md5 = (proc.stdout or "").split()[0] if proc.returncode == 0 and (proc.stdout or "").strip() else None
    if remote_md5 is None:
        raise RuntimeError(f"could not read remote checksum for {remote_path}: {proc.stderr}")
    return remote_md5


def fetch_and_verify(remote_path: str, local_path: Path, *,
                      ssh_run=default_ssh_run, fetch=default_rsync_fetch, host: str = HOST) -> str:
    """Fetch a remote file and verify it by checksum. Returns the verified md5.

    `fetch` always has the signature `(remote_path, local_path, host=...)` — both the
    real `default_rsync_fetch` and any test fake must accept `host`, even if a fake
    ignores it, so this call site never needs to special-case which one it was given.
    """
    fetch(remote_path, local_path, host=host)
    if not local_path.exists():
        raise RuntimeError(f"fetch reported success but local file is missing: {local_path}")
    remote_md5 = _read_remote_md5(remote_path, ssh_run=ssh_run, host=host)
    local_md5 = md5_file(local_path)
    if remote_md5 != local_md5:
        raise RuntimeError(
            f"checksum mismatch after fetch: {remote_path} ({remote_md5}) != {local_path} ({local_md5})"
        )
    return local_md5


def push_and_verify(local_path: Path, remote_path: str, *,
                     ssh_run=default_ssh_run, push=default_rsync_push, host: str = HOST) -> str:
    """Push a local file to the host and verify it by checksum (the push-side mirror of
    `fetch_and_verify`) — used to ship the locally-built, filtered input subset (I10)
    without trusting rsync's own exit code alone."""
    local_md5 = md5_file(local_path)
    push(local_path, remote_path, host=host)
    remote_md5 = _read_remote_md5(remote_path, ssh_run=ssh_run, host=host)
    if remote_md5 != local_md5:
        raise RuntimeError(
            f"checksum mismatch after push: {local_path} ({local_md5}) != {remote_path} ({remote_md5})"
        )
    return local_md5


# --------------------------------------------------------------------------------------
# Chunk provenance (resume / idempotency)
# --------------------------------------------------------------------------------------

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# R4-2: every chunk-provenance field `run_tool_pipeline` reads unconditionally off an
# "ok" chunk result. A record predating any of them must not resume as done — see
# chunk_is_done. Register new fields here when adding them, or a future resume crashes
# with a KeyError instead of failing loudly.
REQUIRED_CHUNK_PROVENANCE_KEYS = (
    "genes_in",
    "genes_out",
    "reachable_genes_in",
    "structurally_unreachable_locus",
    "missing_loci_unexplained",
)


def chunk_provenance_path(tool: str, chrom: str) -> Path:
    return LOCAL_CHUNKS / f"{tool}_{chrom}_raw.provenance.json"


def chunk_output_path(tool: str, chrom: str) -> Path:
    return LOCAL_CHUNKS / f"{tool}_{chrom}_raw.gff3"


def chunk_is_done(tool: str, chrom: str, expected_genes_in: int) -> bool:
    """A chunk is done only if its provenance says "ok", was computed against the same
    genes_in we would compute today (guards against a stale marker surviving a change to
    the staged input), and the output file on disk still has the gene count that was
    recorded when the chunk actually finished (I7: a chunk truncated on disk after its
    provenance was written — e.g. a filesystem fault, a killed process after the
    provenance write — must not resume as "done"; it must be noticed here, not only if
    and when the loss happens to clear the final 5% gate).
    """
    prov = load_json(chunk_provenance_path(tool, chrom))
    if not prov or prov.get("status") != "ok":
        return False
    if prov.get("genes_in") != expected_genes_in:
        return False
    # R4-2: a provenance record written by an older revision of this script can be
    # missing fields that `run_tool_pipeline` now reads unconditionally — round 3's
    # `reachable_genes_in` (summed directly, `KeyError` on a stale record) and round 4's
    # `missing_loci_unexplained`. A latent hard crash on resume is exactly what a script
    # whose purpose is surviving interruption must not have, so an incomplete record is
    # not-done, the same as a genes_in mismatch or a missing subset file below.
    # REQUIRED_CHUNK_PROVENANCE_KEYS is the contract, not a list of the two fields that
    # happened to break first: every field a later round teaches run_tool_pipeline to
    # read must be registered there, or this check silently stops covering it.
    if any(key not in prov for key in REQUIRED_CHUNK_PROVENANCE_KEYS):
        return False
    out_path = chunk_output_path(tool, chrom)
    if not out_path.exists():
        return False
    if count_gff3_features(out_path, "gene") != prov.get("genes_out"):
        return False
    # N4: the missing-loci manifest is built from this chunk's *subset* file, not its
    # output — a resumed run whose _chunks/ got cleaned would otherwise "succeed" with
    # the output/provenance intact but no subset to compute the manifest's input side
    # from. Treat that as not-done too, so re-running rebuilds the subset instead of
    # silently degrading the manifest.
    subset_path = LOCAL_CHUNKS / f"{tool}_{chrom}_subset.gff3"
    if not subset_path.exists():
        return False
    return True


# --------------------------------------------------------------------------------------
# Per-chunk execution
# --------------------------------------------------------------------------------------

def run_remote_chunk(tool: str, chrom: str, max_loss_fraction: float, *,
                      ssh_run=default_ssh_run, fetch=default_rsync_fetch, push=default_rsync_push,
                      host: str = HOST,
                      max_unexplained_missing_loci: int = DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI) -> dict:
    """Run one (tool, chromosome) chunk on the GPU host and fetch its output back.

    Never raises for an ordinary loss violation or remote failure — those are recorded
    in the returned dict as status="failed" so the caller can decide whether to stop the
    whole tool's run while still leaving a provenance record a future resume can use.
    Only truly unexpected local errors (e.g. a filesystem error writing the provenance
    file) propagate.

    The host's git commit is captured and *enforced* here, per chunk, at the moment this
    chunk actually runs (I4) — not once for the whole tool's invocation, since a resumed
    run can span days and a single "commit as of now" would misrepresent chromosomes
    that ran earlier. There is no way to skip this check: constraint #2 ("the host must
    stay at the pinned commit") is exactly the kind of silent drift this script exists
    to catch, so — unlike the genome/input staging checksums — it is never gated behind
    `--skip-preflight`.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    result: dict = {
        "tool": tool, "chromosome": chrom,
        "started_at": started_at, "finished_at": None, "wall_seconds": None,
        "status": "failed", "reason": None,
        "genes_in": None, "reachable_genes_in": None, "genes_out": None,
        "structurally_unreachable_locus": None, "noncoding_features_excluded": None,
        # R4-3: an exact per-locus accounting, not a ratio — see the gate below. Stays
        # None (never 0) for a chunk that failed before its output could be compared,
        # so "no unexplained misses" is never confused with "never measured".
        "missing_loci_unexplained": None, "missing_loci_unexplained_examples": None,
        "distinct_loci_out": None,
        "host_git_commit": None,
        "parsing_errors_found": None, "parsing_errors_skipped": None, "parsing_errors_total": None,
        "output_written": None,
        "remote_subset": None, "remote_output": None, "remote_log": None,
        "ssh_returncode": None,
    }

    def _finish(status: str, reason: str | None = None) -> dict:
        result["status"] = status
        result["reason"] = reason
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["wall_seconds"] = time.time() - t0
        return result

    try:
        result["host_git_commit"] = get_and_verify_host_commit(ssh_run=ssh_run, host=host)
    except Exception as e:  # noqa: BLE001 - a pinned-commit failure is a chunk failure, not a crash
        return _finish("failed", f"host commit check failed: {e}")

    local_subset = LOCAL_CHUNKS / f"{tool}_{chrom}_subset.gff3"
    subset_info = build_local_subset(tool, chrom, LOCAL_INPUTS / f"{tool}_Athaliana.gff3", local_subset)
    result["genes_in"] = subset_info["genes"]
    result["noncoding_features_excluded"] = subset_info["excluded_feature_counts"]
    # N6 (round 3): the last gene in file order is structurally unreachable regardless
    # of content (preprocess.py's genome2GSFDataset never flushes it — see
    # build_local_subset's docstring). "Do not patch preprocess.py; account for the bug,
    # don't fix it" — so this chunk's own loss gate is computed against the genes it
    # could actually produce, not the raw count Task 2/3 reconciliation still uses.
    result["structurally_unreachable_locus"] = subset_info["structurally_unreachable_locus"]
    result["reachable_genes_in"] = subset_info["genes"] - (
        1 if subset_info["structurally_unreachable_locus"] else 0
    )

    paths = build_chunk_remote_command(tool, chrom)
    result["remote_subset"] = paths["remote_subset"]
    result["remote_output"] = paths["remote_output"]
    result["remote_log"] = paths["remote_log"]

    try:
        push_and_verify(local_subset, paths["remote_subset"], ssh_run=ssh_run, push=push, host=host)
    except Exception as e:  # noqa: BLE001
        return _finish("failed", f"push/verify of input subset failed: {e}")

    proc = ssh_run(build_ssh_argv(paths["command"], host=host))
    result["ssh_returncode"] = proc.returncode

    local_out = chunk_output_path(tool, chrom)
    local_log = LOCAL_CHUNKS / f"{tool}_{chrom}_raw.log"

    if proc.returncode != 0:
        # I5: fetch the log even on failure. All of prompt_mode.py's own stdout/stderr
        # was redirected into the remote log file, so `proc.stderr` here (the SSH
        # command's own stderr) is typically empty — without fetching the log itself,
        # the only diagnostic left would be "remote command exited 1" with nothing
        # after the colon.
        log_excerpt = "(remote log not fetched)"
        try:
            fetch(paths["remote_log"], local_log, host=host)
            if local_log.exists():
                log_excerpt = local_log.read_text()[-2000:]
        except Exception as log_err:  # noqa: BLE001 - best-effort diagnostic, never masks the real failure
            log_excerpt = f"(could not fetch remote log: {log_err})"
        return _finish(
            "failed",
            f"remote command exited {proc.returncode}; ssh stderr: {(proc.stderr or '')[:500]!r}; "
            f"remote log tail: {log_excerpt}",
        )

    try:
        fetch_and_verify(paths["remote_output"], local_out, ssh_run=ssh_run, fetch=fetch, host=host)
        # Best-effort: the log is diagnostic, not load-bearing for scoring, so its
        # absence downgrades to a recorded field rather than failing the chunk.
        try:
            fetch_and_verify(paths["remote_log"], local_log, ssh_run=ssh_run, fetch=fetch, host=host)
        except Exception as log_err:  # noqa: BLE001 - deliberately broad, see comment above
            result["log_fetch_error"] = str(log_err)
    except Exception as fetch_err:  # noqa: BLE001
        return _finish("failed", f"fetch/verify of remote output failed: {fetch_err}")

    combined_text = local_log.read_text() if local_log.exists() else ""
    perr = parse_parsing_errors(combined_text)
    result["parsing_errors_found"] = perr.found
    if perr.found:
        # prompt_mode.py only prints this line when error_count > 0, so a genuinely
        # clean chunk (0 errors) legitimately also has found=False — genes_in/genes_out
        # (always numeric, checked below) are the authoritative signal either way; this
        # field is diagnostic detail, not the primary loss check.
        result["parsing_errors_skipped"] = perr.skipped
        result["parsing_errors_total"] = perr.total
    else:
        result["parsing_errors_skipped"] = "N/A (summary line absent)"
        result["parsing_errors_total"] = "N/A (summary line absent)"
    result["output_written"] = parse_output_written(combined_text) is not None

    genes_out = count_gff3_features(local_out, "gene")
    result["genes_out"] = genes_out

    if result["reachable_genes_in"] > 0:
        try:
            check_loss(result["reachable_genes_in"], genes_out, max_loss_fraction, context=f"{tool}/{chrom}")
        except LossTooHigh as e:
            return _finish("failed", str(e))
    # else: a chunk with exactly one gene has, by construction, nothing reachable at
    # all (its only gene is the structurally-unreachable last one) — there is no loss
    # rate to measure, so this is a vacuous pass rather than a "genes_in is 0" error.
    # Real chromosomes have thousands (or, for the smallest, dozens) of genes, so this
    # only ever matters for a minimal test fixture, not a real run.

    # R4-3 (chunk level): which loci this chunk was asked about and never answered for,
    # named — not a rate. `check_loss` above is a ratio against a denominator round 3
    # deliberately shrank by the known structural loss, and a ratio cannot see a single
    # real miss on anything but a tiny chromosome: at the 5% default it tolerates ~309
    # lost loci on EGAPx Chr1 (6,194 filtered). This compares the subset actually pushed
    # against the GM= values actually emitted, so one genuinely-never-generated locus is
    # caught on the chromosome that produced it rather than ~6 GPU-hours later at the
    # tool-level manifest gate. The tool-level check is the strict superset of this one
    # (raw chunk output ⊇ the standardized, top-beam-filtered final file), so this adds
    # no failure that the end of the run would not have reached anyway — only an earlier,
    # better-attributed one.
    prompted_loci = _gff_first_attr_values(local_subset, "gene")
    emitted_loci = _gff_gm_values(local_out)
    result["distinct_loci_out"] = len(emitted_loci)

    # R4-4: every gate here counts loss, so a chunk that came back with *more* gene rows
    # than distinct loci had no check at all. That is the observable signature of the
    # stale-remote-DB accumulation `build_prompt_mode_command`'s `rm -f` now prevents
    # (22 rows / 11 GM= values, measured), and it is worth failing on independently of
    # that fix: it is also what a future invocation exporting both beam hypotheses would
    # produce, and gpu-environment.md is explicit that this pipeline emits exactly one
    # record per locus (`num_return_sequences=1`; 18 rows for 18 loci, verified twice).
    # If that ever stops being true, this must be a loud stop, not a silently doubled
    # denominator downstream.
    if genes_out != len(emitted_loci):
        return _finish(
            "failed",
            f"{tool}/{chrom}: output has {genes_out} gene rows for "
            f"{len(emitted_loci)} distinct GM= loci — this invocation emits exactly one "
            f"record per locus, so a mismatch means duplicated records (a stale remote "
            f"database accumulating across runs) or an invocation change that exports "
            f"more than one hypothesis per locus. Refusing to record a count nothing "
            f"downstream would flag, since every other gate here only measures loss."
        )

    unreachable = result["structurally_unreachable_locus"]
    unexplained = sorted(locus for locus in (prompted_loci - emitted_loci) if locus != unreachable)
    result["missing_loci_unexplained"] = len(unexplained)
    result["missing_loci_unexplained_examples"] = unexplained[:10]
    if len(unexplained) > max_unexplained_missing_loci:
        return _finish(
            "failed",
            f"{tool}/{chrom}: {len(unexplained)} locus/loci were prompted but never "
            f"emitted, for a reason other than the known structural loss "
            f"({unreachable!r}) — e.g. {unexplained[:5]}. The {max_loss_fraction:.1%} "
            f"ratio gate passed this chunk; an exact per-locus check does not. Raising "
            f"--max-unexplained-missing-loci (with --acknowledge-high-loss-threshold) "
            f"records the decision in provenance and re-runs this chunk (R4-3)."
        )

    return _finish("ok")


def ensure_chunk(tool: str, chrom: str, max_loss_fraction: float, *,
                  ssh_run=default_ssh_run, fetch=default_rsync_fetch, push=default_rsync_push,
                  host: str = HOST, force: bool = False,
                  max_unexplained_missing_loci: int = DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI) -> dict:
    """Run a chunk, or return its already-recorded result if it previously succeeded.

    A resumed chunk makes no SSH calls at all — it returns the exact record written the
    last time this chunk actually ran.

    R4-1: the "has the staged input changed" fingerprint is now computed the same way
    `run_remote_chunk` computes `genes_in` — via `build_local_subset`'s own post-filter
    count. It previously came from a separate helper that counted raw, unfiltered gene
    rows, which disagreed with the filtered count for any chromosome with excluded
    non-coding or childless rows: always for EGAPx (lnc_RNA/pseudogene/childless-shell
    exclusions are never zero there), never for GeMoMa or BRAKER3 (which have none). A
    completed EGAPx chunk's provenance `genes_in` (the filtered count) could therefore
    never match that fingerprint (the raw count), so `chunk_is_done` always reported it
    as not-done and `ensure_chunk` re-ran every EGAPx chunk from scratch on every
    invocation — silently defeating resumability for the one tool where a ~6-hour re-run
    is most costly. The old helper is removed rather than reconciled, so there is now
    exactly one definition of "how many genes are in this chunk". Building the subset
    here (idempotent, local, no network) also keeps `_chunks/*_subset.gff3` fresh even
    on a pure resume.
    """
    local_subset = LOCAL_CHUNKS / f"{tool}_{chrom}_subset.gff3"
    subset_info = build_local_subset(tool, chrom, LOCAL_INPUTS / f"{tool}_Athaliana.gff3", local_subset)
    genes_in = subset_info["genes"]
    if not force and chunk_is_done(tool, chrom, genes_in):
        prov = dict(load_json(chunk_provenance_path(tool, chrom)))
        prov["resumed"] = True
        return prov
    result = run_remote_chunk(tool, chrom, max_loss_fraction, ssh_run=ssh_run, fetch=fetch, push=push,
                               host=host, max_unexplained_missing_loci=max_unexplained_missing_loci)
    result["resumed"] = False
    write_json(chunk_provenance_path(tool, chrom), result)
    return result


# --------------------------------------------------------------------------------------
# Merge / standardize / top-beam filter
# --------------------------------------------------------------------------------------

def merge_gff3_files(paths: list[Path], out_path: Path) -> int:
    """Concatenate per-chromosome raw outputs into one file. Returns the merged gene
    count. Raises if any expected chunk file is missing (never silently merges fewer
    chromosomes than requested)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("##gff-version 3\n")
        for p in paths:
            assert_not_scratch(p)
            if not p.exists():
                raise FileNotFoundError(f"expected chunk output missing: {p}")
            with p.open() as fh:
                for line in fh:
                    if line.startswith("##gff-version"):
                        continue
                    out.write(line)
    return count_gff3_features(out_path, "gene")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def standardize_output(raw_path: Path, out_path: Path, tool: str,
                        genome_dir: Path = LOCAL_GENOME_DIR,
                        script_path: Path = STANDARDIZE_SCRIPT) -> None:
    """Run the same validation pass that produced `standardized_results/`, by calling
    `standardize_gff()` directly rather than through its CLI wrapper.

    The CLI's `--input-file` mode infers species/tool from a `results/<tool>_results/
    <file>` directory layout (see `run_lsativa_rerun_pipeline.sh`); our flat
    `polishing_benchmark/predictions/` does not follow it, and letting it mis-infer the
    species would silently drop the `.fai`-based chromosome-length check — the same
    class of silent-degradation bug this whole script exists to avoid. Calling the
    function directly with an explicit `species_prefix="A_thaliana"` and `genome_dir`
    sidesteps that entirely.

    I8: asserts the `.fai` this coordinate validation depends on actually exists, rather
    than letting `standardize_gff()`'s own `load_chrom_lengths()` fail silently — it
    returns `{}` for a missing file, prints a `[WARN]` this script never captures, and
    falls back to a 500,000,000 bp `MAX_COORDINATE` ceiling: no meaningful validation at
    all for a ~30 Mb chromosome.
    """
    fai_path = Path(genome_dir) / "Athaliana_167_TAIR10.fa.fai"
    if not fai_path.exists():
        raise FileNotFoundError(
            f"standardize_output requires a live .fai for coordinate validation against "
            f"real chromosome lengths, found none at {fai_path} — without it, "
            f"standardize_gff() silently degrades to a 500,000,000 bp MAX_COORDINATE "
            f"fallback and validates nothing meaningful for a ~30 Mb chromosome"
        )
    mod = _load_module("standardize_gff", script_path)
    mod.standardize_gff(str(raw_path), str(out_path), "A_thaliana", tool, genome_dir=str(genome_dir))


def filter_top_beam(in_path: Path, out_path: Path, script_path: Path = BEAM1_FILTER_SCRIPT) -> None:
    """Filter to the first gene record per GM= value (beam rank 1), reusing
    `13_beam1_filter.py`'s own `main()` rather than reimplementing the rule.

    On this invocation (fixed round 1 correction), `prompt_mode.py` calls
    `model.generate(..., num_return_sequences=1)`, so it does not actually export two
    beam hypotheses per locus — this filter is a no-op safety net here, not a
    load-bearing step. Kept anyway: free insurance against a future checkpoint or
    invocation that does export both beams, at negligible cost.
    """
    mod = _load_module("beam1_filter", script_path)
    mod.main(str(in_path), str(out_path))


# --------------------------------------------------------------------------------------
# Missing-loci manifest (I3)
# --------------------------------------------------------------------------------------

def _first_attribute_value(attributes: str) -> str | None:
    """Replicate the pinned host's own `GM=` derivation exactly
    (`preprocess.py:314`: `attributes.split(';')[0].split('=')[1]`) — the value of
    whichever attribute comes *first* on the line, not necessarily `ID=`.

    This matters because GeMoMa's gene rows lead with `Name=` (`Name=Ath_00001;
    ID=gene_0;...`), so its real `GM=` is the Name value, not the ID — BRAKER3 and
    EGAPx both lead with `ID=` and are unaffected (see N2).
    """
    first = attributes.split(";", 1)[0]
    if "=" not in first:
        return None
    return first.split("=", 1)[1].strip()


def _gff_ids(path: Path, feature: str) -> set[str]:
    """Every `ID=` value on rows of the given feature type in a GFF3 file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != feature:
                continue
            fid = _gff_attr(cols[8], "ID")
            if fid:
                ids.add(fid)
    return ids


def _gff_first_attr_values(path: Path, feature: str) -> set[str]:
    """Every row's first-attribute value (the host's own `GM=` derivation rule, see
    `_first_attribute_value`) for rows of the given feature type."""
    values: set[str] = set()
    if not path.exists():
        return values
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != feature:
                continue
            v = _first_attribute_value(cols[8])
            if v:
                values.add(v)
    return values


def _gff_gm_values(path: Path) -> set[str]:
    """Every `GM=` value on `gene` rows of a completion-mode output file — the exact
    gene id string of the input locus it was prompted from (see `prompt_mode.py`'s
    `gm_id` and `32_score_polishing.py`'s own docstring on this same convention)."""
    values: set[str] = set()
    if not path.exists():
        return values
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            gm = _gff_attr(cols[8], "GM")
            if gm:
                values.add(gm)
    return values


STRUCTURALLY_UNREACHABLE_REASON = (
    "structurally unreachable: last gene in this chromosome chunk's preprocessing "
    "input, never inserted by genome2GSFDataset (preprocess.py finalizes a gene only "
    "when the *next* gene line is seen; the file loop ends and con.close() runs with "
    "no final flush) — not a generation failure, known and predictable before the run"
)
UNEXPLAINED_MISSING_REASON = "never generated (unexplained)"


def write_missing_loci_manifest(chunk_results: list[dict], final_path: Path, out_path: Path) -> dict:
    """Write every locus the input subset carried but the final output does not — a
    locus the pipeline never emitted at all (I3).

    This is the artifact that makes it impossible for a never-generated locus to be
    scored as "still wrong": Task 5 can check a locus against this file before scoring
    it as damage, rather than that guarantee living only in good intentions.

    N2: pairing is on `GM=`, but `GM=` is not the input `ID=` — the pinned host derives
    it as the *first* attribute value on the gene line, whatever key that is
    (`_first_attribute_value`). GeMoMa's gene rows lead with `Name=`, so a perfect run
    used to report every GeMoMa locus as "missing" (`GM=Ath_00001` looked up as
    `gene_0`) while passing every synthetic test fixture, where `GM=g1` happens to equal
    `ID=g1`. Fixed by deriving the input key the same way the host derives `GM=`, with
    the `ID=` set kept only as a cross-check (recorded, not used for matching).

    N4: a missing subset file (e.g. a cleaned `_chunks/` on a resumed run) is an error
    here, not a silent empty set — `chunk_is_done` also refuses to call a chunk done
    without its subset present, so this should only ever fire on a genuinely broken
    on-disk state.

    N6 (round 3): each chunk's `structurally_unreachable_locus` (see
    `build_local_subset`/`run_remote_chunk`) is a *known*, content-independent loss —
    the last gene in file order, which `genome2GSFDataset` never flushes — not a real
    generation failure. Every line written here carries a reason distinguishing that
    predictable case from a genuinely unexplained miss, which is the whole point of
    this file: Task 5 must not fold the two together into "still wrong".

    Also raises if every input locus comes back "missing" — overwhelmingly more likely
    to be exactly the N2 key-derivation class of bug than a real 100% loss (which
    `check_loss` would already have caught well before this point), so this is treated
    as its own error rather than written out as if it were a real, if bad, result.

    Returns `{"total": int, "structurally_unreachable": int, "unexplained": int}`.
    """
    input_keys: set[str] = set()
    input_ids: set[str] = set()  # cross-check only, never used to decide "missing"
    known_unreachable: set[str] = set()
    for r in chunk_results:
        subset = LOCAL_CHUNKS / f"{r['tool']}_{r['chromosome']}_subset.gff3"
        if not subset.exists():
            raise FileNotFoundError(
                f"missing-loci manifest needs the input subset that produced "
                f"{r['tool']}/{r['chromosome']}, not found: {subset}"
            )
        input_keys |= _gff_first_attr_values(subset, "gene")
        input_ids |= _gff_ids(subset, "gene")
        locus = r.get("structurally_unreachable_locus")
        if locus:
            known_unreachable.add(locus)

    output_gms = _gff_gm_values(final_path)
    missing = sorted(input_keys - output_gms)

    if input_keys and len(missing) == len(input_keys):
        raise RuntimeError(
            f"missing-loci manifest reports all {len(input_keys)} input loci as never "
            f"generated (ID-based cross-check set was {len(input_ids)}) — almost "
            f"certainly a GM= key-derivation mismatch (N2), not a real 100% loss "
            f"(check_loss would already have stopped the run for that); refusing to "
            f"write a manifest this implausible"
        )

    structurally_unreachable_count = sum(1 for m in missing if m in known_unreachable)
    unexplained_count = len(missing) - structurally_unreachable_count

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("locus\treason\n")
        for m in missing:
            reason = STRUCTURALLY_UNREACHABLE_REASON if m in known_unreachable else UNEXPLAINED_MISSING_REASON
            out.write(f"{m}\t{reason}\n")

    return {
        "total": len(missing),
        "structurally_unreachable": structurally_unreachable_count,
        "unexplained": unexplained_count,
    }


# --------------------------------------------------------------------------------------
# Provenance for a whole tool's run
# --------------------------------------------------------------------------------------

def get_host_git_commit(*, ssh_run=default_ssh_run, host: str = HOST) -> str:
    proc = ssh_run(build_ssh_argv(f"cd {REMOTE_TRANSGENIC_DIR} && git rev-parse HEAD", host=host))
    commit = (proc.stdout or "").strip()
    if proc.returncode != 0 or not commit:
        return f"N/A (could not read host git commit: {(proc.stderr or 'unknown error').strip()})"
    return commit


def get_and_verify_host_commit(*, ssh_run=default_ssh_run, host: str = HOST) -> str:
    """I4: constraint #2 ("the host must stay at the pinned commit") was recorded but
    never enforced — `verify_host_genome` below hard-fails on md5 drift, but nothing
    compared the host's git commit to `EXPECTED_HOST_COMMIT`, even though syncing
    forward was measured to produce zero output on every input. This raises instead of
    just recording, and — unlike the genome/input checksum preflight — is never skipped.
    """
    commit = get_host_git_commit(ssh_run=ssh_run, host=host)
    if commit.startswith("N/A"):
        raise RuntimeError(f"could not establish host git commit: {commit}")
    if commit != EXPECTED_HOST_COMMIT:
        raise RuntimeError(
            f"host git commit drifted: {commit} != expected {EXPECTED_HOST_COMMIT} — "
            f"gpu-environment.md: this exact commit reproduces the published benchmark "
            f"(15/15 shared loci structurally identical); a newer checkout was measured "
            f"to produce zero output on every input. Reproduce the published result on "
            f"the new commit before trusting any prediction made against it."
        )
    return commit


def verify_host_genome(*, ssh_run=default_ssh_run, host: str = HOST) -> str:
    """Preflight: the host's staged genome must match the md5 gpu-environment.md
    recorded. Returns the verified md5, or raises if it has drifted."""
    remote_md5 = _read_remote_md5(REMOTE_GENOME, ssh_run=ssh_run, host=host)
    if remote_md5 != EXPECTED_GENOME_MD5:
        raise RuntimeError(
            f"host genome checksum drifted: {remote_md5} != expected {EXPECTED_GENOME_MD5} "
            f"({REMOTE_GENOME}) — re-verify before trusting any prediction made against it"
        )
    return remote_md5


def verify_host_input(tool: str, *, ssh_run=default_ssh_run, host: str = HOST) -> str:
    """Preflight: the host's staged input for `tool` must be byte-identical to the
    GPFS-staged copy Task 1 produced (both are supposed to be the same file, copied)."""
    local_path = LOCAL_INPUTS / f"{tool}_Athaliana.gff3"
    assert_not_scratch(local_path)
    local_md5 = md5_file(local_path)
    remote_path = f"{REMOTE_INPUTS_DIR}/{tool}_Athaliana.gff3"
    remote_md5 = _read_remote_md5(remote_path, ssh_run=ssh_run, host=host)
    if remote_md5 != local_md5:
        raise RuntimeError(
            f"host input for {tool} does not match the GPFS-staged copy: "
            f"{remote_md5} != {local_md5} ({remote_path} vs {local_path})"
        )
    return local_md5


# --------------------------------------------------------------------------------------
# Whole-tool pipeline
# --------------------------------------------------------------------------------------

def _is_full_genome(chromosomes: tuple[str, ...]) -> bool:
    return set(chromosomes) == set(CHROMOSOMES)


def output_stub(tool: str, chromosomes: tuple[str, ...]) -> str:
    """`{tool}` for the canonical full-genome run, `{tool}_{chr...}` otherwise (C2).

    Without this, a `--chromosomes ChrM` invocation run after a completed full-genome
    run would overwrite `{tool}_completed.gff3` (27k loci) with a 12-gene file under the
    exact same name, and if that invocation then tripped the loss gate, the stale
    provenance claiming 27,416 loci would be left next to a 12-gene prediction with no
    signal anything was wrong.
    """
    if _is_full_genome(chromosomes):
        return tool
    return f"{tool}_{'-'.join(chromosomes)}"


def run_tool_pipeline(tool: str, *, chromosomes: tuple[str, ...] = CHROMOSOMES,
                       max_loss_fraction: float = DEFAULT_MAX_LOSS_FRACTION,
                       ssh_run=default_ssh_run, fetch=default_rsync_fetch, push=default_rsync_push,
                       host: str = HOST, force: bool = False, skip_preflight: bool = False,
                       acknowledge_high_loss_threshold: bool = False,
                       max_unexplained_missing_loci: int = DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI,
                       invocation_args: dict | None = None) -> dict:
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}, expected one of {TOOLS}")
    if max_loss_fraction > DEFAULT_MAX_LOSS_FRACTION and not acknowledge_high_loss_threshold:
        # C3: --max-loss-fraction was a live escape hatch that left no trace in
        # provenance — a run at 50% looked byte-for-byte like a run at 5%. Refusing to
        # proceed above the default without an explicit, *recorded* acknowledgement
        # means a loosened gate can never be invisible to a future reader.
        raise ValueError(
            f"max_loss_fraction={max_loss_fraction} exceeds the default "
            f"{DEFAULT_MAX_LOSS_FRACTION} and acknowledge_high_loss_threshold was not "
            f"set — see task-4-findings-round1.md C3"
        )
    if (max_unexplained_missing_loci > DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI
            and not acknowledge_high_loss_threshold):
        # R4-3 rides C3's mechanism rather than inventing a second one: tolerating a
        # locus that was prompted and never came back is exactly as much of a loosened
        # gate as raising the ratio, and must be exactly as visible afterwards.
        raise ValueError(
            f"max_unexplained_missing_loci={max_unexplained_missing_loci} exceeds the "
            f"default {DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI} and "
            f"acknowledge_high_loss_threshold was not set — see "
            f"task-4-findings-round4.md R4-3"
        )

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    # These verify the *staging* (does the host still have the genome/input this
    # invocation expects) and are cheap enough to run once per invocation, even one that
    # turns out to resume-and-finish without executing any new chunk.
    genome_md5 = "N/A (preflight skipped)" if skip_preflight else verify_host_genome(ssh_run=ssh_run, host=host)
    input_md5 = "N/A (preflight skipped)" if skip_preflight else verify_host_input(tool, ssh_run=ssh_run, host=host)

    chunk_results = []
    for chrom in chromosomes:
        result = ensure_chunk(tool, chrom, max_loss_fraction, ssh_run=ssh_run, fetch=fetch, push=push,
                               host=host, force=force,
                               max_unexplained_missing_loci=max_unexplained_missing_loci)
        chunk_results.append(result)
        if result.get("status") != "ok":
            raise LossTooHigh(
                f"{tool}/{chrom} did not complete cleanly ({result.get('reason')}); "
                f"re-run the same command to resume — completed chromosomes are skipped"
            )

    # host_git_commit is captured and enforced per chunk, at the moment each one
    # actually ran (see run_remote_chunk/get_and_verify_host_commit) — not once for this
    # invocation — because a resumed run can span days and a single "commit as of now"
    # would misrepresent chunks that ran earlier.
    commits = sorted({r.get("host_git_commit") for r in chunk_results if r.get("host_git_commit")})
    if len(commits) == 1:
        host_commit = commits[0]
    else:
        host_commit = f"N/A (chunks disagree or are missing a recorded commit: {commits})"

    stub = output_stub(tool, chromosomes)
    chunk_paths = [chunk_output_path(tool, c) for c in chromosomes]
    raw_path = LOCAL_PREDICTIONS / f"{stub}_raw.gff3"
    genes_raw_merged = merge_gff3_files(chunk_paths, raw_path)  # I1: the first of two loss channels

    standardized_path = LOCAL_PREDICTIONS / f"{stub}_standardized.gff3"
    standardize_output(raw_path, standardized_path, tool)
    genes_standardized = count_gff3_features(standardized_path, "gene")  # I1: the second

    # C1: the handoff file must never be written before the gate that validates it. It
    # is built as `.partial`, gated by check_loss, and only atomically renamed into
    # place as the very last statement of this function (N3) — so a crash or walltime
    # kill anywhere between here and the end leaves no canonical `_completed.gff3` at
    # all, rather than one with no manifest or provenance beside it (C1's exact failure
    # mode, otherwise still reachable out-of-band through this window).
    final_path = LOCAL_PREDICTIONS / f"{stub}_completed.gff3"
    final_partial = LOCAL_PREDICTIONS / f"{stub}_completed.gff3.partial"
    filter_top_beam(standardized_path, final_partial)

    # N6 (round 3): genes_in_total is the true candidate-locus count (what Task 2/3's
    # own denominator reconciles against); reachable_genes_in_total additionally
    # excludes the one structurally-unreachable locus per chunk (see build_local_subset)
    # so the gate is not tripped, and a perfect run does not read as "1 lost", by a loss
    # this runner cannot avoid regardless of model quality.
    genes_in_total = sum(r["genes_in"] for r in chunk_results)
    reachable_genes_in_total = sum(r["reachable_genes_in"] for r in chunk_results)
    genes_out_total = count_gff3_features(final_partial, "gene")
    if reachable_genes_in_total > 0:
        check_loss(reachable_genes_in_total, genes_out_total, max_loss_fraction,
                   context=f"{tool} (final, post-filter)")
    # else: every chunk was a single, structurally-unreachable gene — nothing to
    # measure a loss rate against; see the identical rationale in run_remote_chunk.

    # N3: the manifest is built from final_partial, not final_path — install happens
    # last. (Deliberate: a failed run — check_loss raised above — never reaches this
    # point at all, so no manifest is written for a failed run either; there is no
    # "final" result yet to describe one against, and writing one against a partial
    # file that will never become canonical would itself be a misleading artifact.)
    missing_loci_path = LOCAL_PREDICTIONS / f"{stub}_missing_loci.txt"
    missing_loci = write_missing_loci_manifest(chunk_results, final_partial, missing_loci_path)

    # R4-3: reachable_genes_in_total shrinks the ratio check_loss gates on by exactly
    # the known structural loss per chunk — correct, but it narrows the band in which a
    # *real* additional generation failure on a small chromosome still trips the ratio
    # (e.g. one genuine miss alongside the known one used to read as 2/n against the
    # old, larger denominator; it can now read as 1/(n-1), which may no longer exceed
    # max_loss_fraction). write_missing_loci_manifest already tells the two apart
    # correctly — this is the gate that actually acts on that distinction, so a real,
    # unexplained miss is never treated as complete just because the ratio alone looked
    # fine. This still leaves final_partial and the manifest on disk as diagnostics
    # (matching C1's own choice for the ratio gate), but installs nothing canonical.
    # The chunk-level gate in run_remote_chunk already applies this same threshold to
    # each chunk's *raw* output; this one applies it to what actually survives
    # standardisation and the top-beam filter, which is the file Task 5 scores. A locus
    # dropped by standardize_gff() (invalid coordinates, an implausible span) reaches
    # only this gate, so the two are not redundant.
    if missing_loci["unexplained"] > max_unexplained_missing_loci:
        raise LossTooHigh(
            f"{tool}: {missing_loci['unexplained']} locus/loci went missing for a "
            f"reason other than the known per-chunk structural loss (see "
            f"{missing_loci_path}) — stopping rather than writing a result that looks "
            f"complete. check_loss's ratio alone cannot be trusted to catch this on a "
            f"small chromosome once the known structural loss is excluded from its "
            f"denominator (R4-3)."
        )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance = {
        "tool": tool,
        "host": host,
        "host_git_commit": host_commit,
        "model": MODEL,
        "model_invocation": {
            "batch_size": BATCH_SIZE, "num_beams": NUM_BEAMS,
            "max_length": MAX_LENGTH, "device": DEVICE,
        },
        # C3: every CLI argument the invocation actually used, not just the hardcoded
        # model-invocation constants above — so --max-loss-fraction, --force,
        # --acknowledge-high-loss-threshold etc. are never invisible in provenance.
        "invocation_args": invocation_args or {},
        "genome_md5": genome_md5,
        "input_md5": input_md5,
        "chromosomes": list(chromosomes),
        "chunks": chunk_results,
        "genes_in_total": genes_in_total,
        "reachable_genes_in_total": reachable_genes_in_total,
        "genes_raw_merged": genes_raw_merged,
        "genes_standardized": genes_standardized,
        "genes_out_total": genes_out_total,
        # N6: split so a reader never has to reparse the manifest file to tell a known,
        # predictable loss from a genuinely unexplained one.
        "missing_loci_count": missing_loci["total"],
        "missing_loci_structurally_unreachable": missing_loci["structurally_unreachable"],
        "missing_loci_unexplained": missing_loci["unexplained"],
        "missing_loci_path": str(missing_loci_path),
        "started_at": started_at, "finished_at": finished_at,
        "wall_seconds": time.time() - t0,
        "raw_path": str(raw_path), "standardized_path": str(standardized_path),
        "final_path": str(final_path),
        "status": "ok",
    }
    # N3: provenance is written before the canonical file is installed, for the same
    # reason the manifest is built from final_partial above — os.replace() below is the
    # single statement that makes this run's result visible as "the answer" at all, and
    # it happens only once everything else it should be found next to already exists.
    write_json(LOCAL_PREDICTIONS / f"{stub}_provenance.json", provenance)
    os.replace(final_partial, final_path)
    return provenance


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", nargs="+", choices=list(TOOLS) + ["all"], default=["all"])
    ap.add_argument("--chromosomes", nargs="+", default=list(CHROMOSOMES),
                     help="Restrict to specific chromosomes (e.g. for a smoke test). "
                          "A non-full set writes {tool}_{chromosomes}_completed.gff3, "
                          "never the canonical {tool}_completed.gff3 (C2).")
    ap.add_argument("--max-loss-fraction", type=float, default=DEFAULT_MAX_LOSS_FRACTION)
    ap.add_argument("--max-unexplained-missing-loci", type=int,
                     default=DEFAULT_MAX_UNEXPLAINED_MISSING_LOCI,
                     help="How many prompted loci may come back missing for a reason "
                          "other than the known per-chunk structural loss before the "
                          "run stops (default 0, at both chunk and tool level). Raising "
                          "it requires --acknowledge-high-loss-threshold and re-runs any "
                          "chunk that already failed on it (R4-3).")
    ap.add_argument("--acknowledge-high-loss-threshold", action="store_true",
                     help="Required to use --max-loss-fraction or "
                          "--max-unexplained-missing-loci above their defaults; the "
                          "acknowledgement itself is recorded in provenance (C3).")
    ap.add_argument("--force", action="store_true", help="Ignore prior chunk provenance and redo everything.")
    ap.add_argument("--skip-preflight", action="store_true",
                     help="Skip the genome/input checksum preflight (for offline dry runs/tests). "
                          "Does not affect the host git-commit check, which is never skippable.")
    args = ap.parse_args(argv)

    # C3: the full parsed CLI invocation goes into every tool's provenance verbatim, not
    # just the hardcoded model-invocation constants — so a run's exact arguments are
    # never invisible to a future reader.
    invocation_args = vars(args)

    tools = TOOLS if "all" in args.tool else tuple(args.tool)
    # Tools are independent: one tool's failure (or an overnight crash partway through
    # it) must not stop the others from finishing and being written to disk. Each is
    # itself resumable (see run_tool_pipeline/ensure_chunk), so re-running this same
    # command later retries only what failed.
    any_failed = False
    for tool in tools:
        print(f"=== {tool} ===", file=sys.stderr)
        try:
            summary = run_tool_pipeline(
                tool, chromosomes=tuple(args.chromosomes), max_loss_fraction=args.max_loss_fraction,
                force=args.force, skip_preflight=args.skip_preflight,
                acknowledge_high_loss_threshold=args.acknowledge_high_loss_threshold,
                max_unexplained_missing_loci=args.max_unexplained_missing_loci,
                invocation_args=invocation_args,
            )
        except Exception as e:  # noqa: BLE001 - report and move on to the next tool
            print(f"FAILED: {tool}: {e}", file=sys.stderr)
            any_failed = True
            continue
        print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=2), file=sys.stderr)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
