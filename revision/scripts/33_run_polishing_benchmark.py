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
5. `--num-beams 2` (the published Methods setting) exports both beam hypotheses as
   separate gene records; step 3 below filters to the top beam per locus, the same
   rule as `13_beam1_filter.py` / `27_rescore_prompted_topbeam.py`. Scoring an
   unfiltered two-beam file halves precision.

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
DEFAULT_MAX_LOSS_FRACTION = 0.05  # published run lost 3/27,416 = 0.01%; 5% is a generous
                                  # "something is clearly wrong" trigger, not a target.

ROOT = Path(__file__).resolve().parents[3]  # /data/gpfs/assoc/pgl/data/Transgenic
BENCH = ROOT / "polishing_benchmark"
LOCAL_INPUTS = BENCH / "inputs"
LOCAL_PREDICTIONS = BENCH / "predictions"
LOCAL_CHUNKS = LOCAL_PREDICTIONS / "_chunks"
LOCAL_GENOME_DIR = ROOT / "genomes"  # Athaliana_167_TAIR10.fa.fai lives here
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


def count_genes_for_chromosome(path: Path, chrom: str) -> int:
    """Count `gene` rows on one seqid, without needing the remote per-chromosome subset
    file at all — the staged input is byte-identical on GPFS and on the host (Task 1),
    so this is computed once, locally, against the canonical staged copy."""
    if not path.exists():
        return 0
    n = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3 and cols[0] == chrom and cols[2] == "gene":
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


def build_chromosome_subset_command(remote_input_gff: str, chrom: str, remote_subset_path: str) -> str:
    assert_not_scratch(remote_input_gff)
    assert_not_scratch(remote_subset_path)
    return f"awk -F'\\t' -v c={chrom} '$1==c' {remote_input_gff} > {remote_subset_path}"


def build_prompt_mode_command(remote_gff: str, remote_output: str, remote_db: str, remote_log: str) -> str:
    """The exact remote command for one (tool, chromosome) chunk.

    `--batch-size 1` and `--num-beams 2` are literal constants in this string, not
    interpolated from any caller-supplied value — see the module docstring.
    """
    for value in (remote_gff, remote_output, remote_db, remote_log):
        assert_not_scratch(value)
    return (
        f"{build_remote_env_setup()} && cd {REMOTE_TRANSGENIC_DIR} && "
        f"python3 examples/prompt_mode.py "
        f"--genome {REMOTE_GENOME} --gff {remote_gff} --output {remote_output} "
        f"--db {remote_db} --model {MODEL} --batch-size {BATCH_SIZE} "
        f"--num-beams {NUM_BEAMS} --max-length {MAX_LENGTH} --device {DEVICE} "
        f"> {remote_log} 2>&1"
    )


def build_chunk_remote_command(tool: str, chrom: str) -> dict:
    """All remote paths + the combined subset-then-infer command for one chunk."""
    remote_input = f"{REMOTE_INPUTS_DIR}/{tool}_Athaliana.gff3"
    remote_subset = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_subset.gff3"
    remote_output = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_raw.gff3"
    remote_db = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}.db"
    remote_log = f"{REMOTE_PREDICTIONS_DIR}/{tool}_{chrom}_raw.log"
    subset_cmd = build_chromosome_subset_command(remote_input, chrom, remote_subset)
    infer_cmd = build_prompt_mode_command(remote_subset, remote_output, remote_db, remote_log)
    return {
        "remote_input": remote_input,
        "remote_subset": remote_subset,
        "remote_output": remote_output,
        "remote_db": remote_db,
        "remote_log": remote_log,
        "command": f"{subset_cmd} && {infer_cmd}",
    }


def build_ssh_argv(remote_command: str, host: str = HOST) -> list[str]:
    return ["ssh", host, remote_command]


def build_rsync_fetch_argv(remote_path: str, local_path: Path, host: str = HOST) -> list[str]:
    return ["rsync", "-az", f"{host}:{remote_path}", str(local_path)]


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
    proc = ssh_run(build_ssh_argv(f"md5sum {remote_path}", host=host))
    remote_md5 = (proc.stdout or "").split()[0] if proc.returncode == 0 and (proc.stdout or "").strip() else None
    if remote_md5 is None:
        raise RuntimeError(f"could not read remote checksum for {remote_path}: {proc.stderr}")
    local_md5 = md5_file(local_path)
    if remote_md5 != local_md5:
        raise RuntimeError(
            f"checksum mismatch after fetch: {remote_path} ({remote_md5}) != {local_path} ({local_md5})"
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


def chunk_provenance_path(tool: str, chrom: str) -> Path:
    return LOCAL_CHUNKS / f"{tool}_{chrom}_raw.provenance.json"


def chunk_output_path(tool: str, chrom: str) -> Path:
    return LOCAL_CHUNKS / f"{tool}_{chrom}_raw.gff3"


def chunk_is_done(tool: str, chrom: str, expected_genes_in: int) -> bool:
    """A chunk is done only if its provenance says "ok" AND was computed against the
    same genes_in we would compute today — guards against a stale marker surviving a
    change to the staged input.
    """
    prov = load_json(chunk_provenance_path(tool, chrom))
    if not prov or prov.get("status") != "ok":
        return False
    if prov.get("genes_in") != expected_genes_in:
        return False
    return chunk_output_path(tool, chrom).exists()


# --------------------------------------------------------------------------------------
# Per-chunk execution
# --------------------------------------------------------------------------------------

def run_remote_chunk(tool: str, chrom: str, max_loss_fraction: float, *,
                      ssh_run=default_ssh_run, fetch=default_rsync_fetch, host: str = HOST,
                      skip_preflight: bool = False) -> dict:
    """Run one (tool, chromosome) chunk on the GPU host and fetch its output back.

    Never raises for an ordinary loss violation or remote failure — those are recorded
    in the returned dict as status="failed" so the caller can decide whether to stop the
    whole tool's run while still leaving a provenance record a future resume can use.
    Only truly unexpected local errors (e.g. a filesystem error writing the provenance
    file) propagate.

    The host's git commit is captured here, per chunk, at the moment this chunk actually
    runs — not once for the whole tool's invocation — so that a resumed run (which may
    span days) records what the host was actually running for *each* prediction, not
    just whatever commit happened to be checked out when the last chunk finished. A
    chunk loaded from a prior "ok" provenance record keeps its own recorded commit
    unchanged; this function is only called for a chunk that is actually executing.
    """
    genes_in = count_genes_for_chromosome(LOCAL_INPUTS / f"{tool}_Athaliana.gff3", chrom)
    host_commit = ("N/A (preflight skipped)" if skip_preflight
                   else get_host_git_commit(ssh_run=ssh_run, host=host))
    paths = build_chunk_remote_command(tool, chrom)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    proc = ssh_run(build_ssh_argv(paths["command"], host=host))
    wall_seconds = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    result = {
        "tool": tool, "chromosome": chrom,
        "genes_in": genes_in,
        "host_git_commit": host_commit,
        "remote_output": paths["remote_output"], "remote_log": paths["remote_log"],
        "ssh_returncode": proc.returncode,
        "started_at": started_at, "finished_at": finished_at, "wall_seconds": wall_seconds,
        "status": "failed", "reason": None,
        "genes_out": None, "parsing_errors_skipped": None, "parsing_errors_total": None,
        "output_written": None,
    }

    if proc.returncode != 0:
        result["reason"] = f"remote command exited {proc.returncode}: {(proc.stderr or '')[:500]}"
        return result

    local_out = chunk_output_path(tool, chrom)
    local_log = LOCAL_CHUNKS / f"{tool}_{chrom}_raw.log"
    try:
        fetch_and_verify(paths["remote_output"], local_out, ssh_run=ssh_run, fetch=fetch, host=host)
        # Best-effort: the log is diagnostic, not load-bearing for scoring, so its
        # absence downgrades to "N/A" rather than failing the chunk.
        try:
            fetch_and_verify(paths["remote_log"], local_log, ssh_run=ssh_run, fetch=fetch, host=host)
        except Exception as log_err:  # noqa: BLE001 - deliberately broad, see comment above
            result["log_fetch_error"] = str(log_err)
    except Exception as fetch_err:  # noqa: BLE001
        result["reason"] = f"fetch/verify of remote output failed: {fetch_err}"
        return result

    combined_text = local_log.read_text() if local_log.exists() else ""
    perr = parse_parsing_errors(combined_text)
    result["parsing_errors_skipped"] = perr.skipped
    result["parsing_errors_total"] = perr.total
    result["output_written"] = parse_output_written(combined_text) is not None

    genes_out = count_gff3_features(local_out, "gene")
    result["genes_out"] = genes_out

    try:
        check_loss(genes_in, genes_out, max_loss_fraction, context=f"{tool}/{chrom}")
    except LossTooHigh as e:
        result["reason"] = str(e)
        return result

    result["status"] = "ok"
    return result


def ensure_chunk(tool: str, chrom: str, max_loss_fraction: float, *,
                  ssh_run=default_ssh_run, fetch=default_rsync_fetch, host: str = HOST,
                  force: bool = False, skip_preflight: bool = False) -> dict:
    """Run a chunk, or return its already-recorded result if it previously succeeded.

    A resumed chunk makes no SSH calls at all — it returns the exact record written the
    last time this chunk actually ran.
    """
    genes_in = count_genes_for_chromosome(LOCAL_INPUTS / f"{tool}_Athaliana.gff3", chrom)
    if not force and chunk_is_done(tool, chrom, genes_in):
        prov = load_json(chunk_provenance_path(tool, chrom))
        prov = dict(prov)
        prov["resumed"] = True
        return prov
    result = run_remote_chunk(tool, chrom, max_loss_fraction, ssh_run=ssh_run, fetch=fetch, host=host,
                               skip_preflight=skip_preflight)
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
    """
    mod = _load_module("standardize_gff", script_path)
    mod.standardize_gff(str(raw_path), str(out_path), "A_thaliana", tool, genome_dir=str(genome_dir))


def filter_top_beam(in_path: Path, out_path: Path, script_path: Path = BEAM1_FILTER_SCRIPT) -> None:
    """Filter to the first gene record per GM= value (beam rank 1), reusing
    `13_beam1_filter.py`'s own `main()` rather than reimplementing the rule."""
    mod = _load_module("beam1_filter", script_path)
    mod.main(str(in_path), str(out_path))


# --------------------------------------------------------------------------------------
# Provenance for a whole tool's run
# --------------------------------------------------------------------------------------

def get_host_git_commit(*, ssh_run=default_ssh_run, host: str = HOST) -> str:
    proc = ssh_run(build_ssh_argv(f"cd {REMOTE_TRANSGENIC_DIR} && git rev-parse HEAD", host=host))
    commit = (proc.stdout or "").strip()
    if proc.returncode != 0 or not commit:
        return f"N/A (could not read host git commit: {(proc.stderr or 'unknown error').strip()})"
    return commit


def verify_host_genome(*, ssh_run=default_ssh_run, host: str = HOST) -> str:
    """Preflight: the host's staged genome must match the md5 gpu-environment.md
    recorded. Returns the verified md5, or raises if it has drifted."""
    proc = ssh_run(build_ssh_argv(f"md5sum {REMOTE_GENOME}", host=host))
    remote_md5 = (proc.stdout or "").split()[0] if proc.returncode == 0 and (proc.stdout or "").strip() else None
    if remote_md5 is None:
        raise RuntimeError(f"could not read host genome checksum: {proc.stderr}")
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
    proc = ssh_run(build_ssh_argv(f"md5sum {remote_path}", host=host))
    remote_md5 = (proc.stdout or "").split()[0] if proc.returncode == 0 and (proc.stdout or "").strip() else None
    if remote_md5 is None:
        raise RuntimeError(f"could not read host input checksum for {tool}: {proc.stderr}")
    if remote_md5 != local_md5:
        raise RuntimeError(
            f"host input for {tool} does not match the GPFS-staged copy: "
            f"{remote_md5} != {local_md5} ({remote_path} vs {local_path})"
        )
    return local_md5


# --------------------------------------------------------------------------------------
# Whole-tool pipeline
# --------------------------------------------------------------------------------------

def run_tool_pipeline(tool: str, *, chromosomes: tuple[str, ...] = CHROMOSOMES,
                       max_loss_fraction: float = DEFAULT_MAX_LOSS_FRACTION,
                       ssh_run=default_ssh_run, fetch=default_rsync_fetch, host: str = HOST,
                       force: bool = False, skip_preflight: bool = False) -> dict:
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}, expected one of {TOOLS}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    # These verify the *staging* (does the host still have the genome/input this
    # invocation expects) and are cheap enough to run once per invocation, even one that
    # turns out to resume-and-finish without executing any new chunk.
    genome_md5 = "N/A (preflight skipped)" if skip_preflight else verify_host_genome(ssh_run=ssh_run, host=host)
    input_md5 = "N/A (preflight skipped)" if skip_preflight else verify_host_input(tool, ssh_run=ssh_run, host=host)

    chunk_results = []
    for chrom in chromosomes:
        result = ensure_chunk(tool, chrom, max_loss_fraction, ssh_run=ssh_run, fetch=fetch,
                               host=host, force=force, skip_preflight=skip_preflight)
        chunk_results.append(result)
        if result.get("status") != "ok":
            raise LossTooHigh(
                f"{tool}/{chrom} did not complete cleanly ({result.get('reason')}); "
                f"re-run the same command to resume — completed chromosomes are skipped"
            )

    # host_git_commit is captured per chunk, at the moment each one actually ran (see
    # run_remote_chunk) — not once for this invocation — because a resumed run can span
    # days and a single "commit as of now" would misrepresent chunks that ran earlier.
    commits = sorted({r.get("host_git_commit") for r in chunk_results if r.get("host_git_commit")})
    if len(commits) == 1:
        host_commit = commits[0]
    else:
        host_commit = f"N/A (chunks disagree or are missing a recorded commit: {commits})"

    chunk_paths = [chunk_output_path(tool, c) for c in chromosomes]
    raw_path = LOCAL_PREDICTIONS / f"{tool}_raw.gff3"
    merge_gff3_files(chunk_paths, raw_path)

    standardized_path = LOCAL_PREDICTIONS / f"{tool}_standardized.gff3"
    standardize_output(raw_path, standardized_path, tool)

    final_path = LOCAL_PREDICTIONS / f"{tool}_completed.gff3"
    filter_top_beam(standardized_path, final_path)

    genes_in_total = sum(r["genes_in"] for r in chunk_results)
    genes_out_total = count_gff3_features(final_path, "gene")
    check_loss(genes_in_total, genes_out_total, max_loss_fraction, context=f"{tool} (final, post-filter)")

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance = {
        "tool": tool,
        "host": host,
        "host_git_commit": host_commit,
        "model": MODEL,
        "cli_args": {
            "batch_size": BATCH_SIZE, "num_beams": NUM_BEAMS,
            "max_length": MAX_LENGTH, "device": DEVICE,
        },
        "genome_md5": genome_md5,
        "input_md5": input_md5,
        "chromosomes": list(chromosomes),
        "chunks": chunk_results,
        "genes_in_total": genes_in_total,
        "genes_out_total": genes_out_total,
        "started_at": started_at, "finished_at": finished_at,
        "wall_seconds": time.time() - t0,
        "raw_path": str(raw_path), "standardized_path": str(standardized_path),
        "final_path": str(final_path),
        "status": "ok",
    }
    write_json(LOCAL_PREDICTIONS / f"{tool}_provenance.json", provenance)
    return provenance


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", nargs="+", choices=list(TOOLS) + ["all"], default=["all"])
    ap.add_argument("--chromosomes", nargs="+", default=list(CHROMOSOMES),
                     help="Restrict to specific chromosomes (e.g. for a smoke test).")
    ap.add_argument("--max-loss-fraction", type=float, default=DEFAULT_MAX_LOSS_FRACTION)
    ap.add_argument("--force", action="store_true", help="Ignore prior chunk provenance and redo everything.")
    ap.add_argument("--skip-preflight", action="store_true",
                     help="Skip the genome/input checksum preflight (for offline dry runs/tests).")
    args = ap.parse_args(argv)

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
            )
        except Exception as e:  # noqa: BLE001 - report and move on to the next tool
            print(f"FAILED: {tool}: {e}", file=sys.stderr)
            any_failed = True
            continue
        print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=2), file=sys.stderr)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
