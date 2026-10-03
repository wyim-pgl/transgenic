#!/usr/bin/env python3
"""Build the B6 observation cache (plan §7.5) from a merged C0 snapshot.

    b6_build_obs_cache.py --evidence b6_evidence_merged.db --family train --roles DATASET_ROLES.b6_v1.tsv \
        --reference Athaliana=TAIR10.fa [--reference SP=FA ...] --corpus b5_full_b6_v1.db --freeze b6_freeze.json \
        --zero train_track:Athaliana:zero.train_track.Athaliana.tsv [--zero ...] --out cache_train_v1/ \
        [--memory-limit 120GB] [--threads N] [--temp-dir DIR]

``--family eval`` (Z. mays, from b6_build_eval_c0.py + merge) takes no ``--corpus``/``--freeze`` and only
eval_input zero lists.  Nothing is written before ``b6_bindings.verify_upstream(stage="build")`` passes (family,
typed provenance keys, leakage and A49.2 gates, per-species reference == C0 inputs.reference, zero-list population
and upstream shas); the published cache is then re-verified with ``verify_cache``.
Every input is sha-bound into the cache manifest (evidence snapshot = recorded sidecar/PROVENANCE sha; roles =
sidecar; zero lists + their QC JSON (``<tsv>.qc.json`` unless given as ``pop:species:tsv:qc``); reference; corpus;
code).  The cache directory must not exist or be empty (caches are immutable).
Rows are extracted and their shards written one (species, chrom) at a time: row memory is bounded by the largest
chromosome, plus species-wide state (the EST molecule map and the dropped-molecule id / library sets, which are
unioned across chromosomes before counting); the cache is byte-identical to a one-shot ``write_cache(extract_rows(...))``
build.

DuckDB resources: ``--memory-limit`` / ``--threads`` are applied to the one read-only evidence connection of the run
(without a limit DuckDB's buffer pool grows towards ~80% of the host's RAM, above a Slurm cgroup: the 9-species train
build was OOM-killed at 178 GB after six species).  Operators and intermediates spill under the limit, but a query's
result set counts against it and does not spill: the limit must exceed the largest per-chromosome result set plus
working memory, or that query errors instead of using RAM; the Python heap of one chromosome's rows comes on top, so
keep limit + heap below the allocation.  Spills go to ``--temp-dir`` (kept) or, by default, to an owned
``b6_build_obs_cache.spill-*`` directory inside the parent of ``--out``, printed to stderr at start and removed on
normal exit, pass or fail -- after an OOM kill or SIGKILL nothing runs, so delete it by hand.  Only these two
settings and the spill directory are set; ``preserve_insertion_order`` stays at DuckDB's default, the one the
production shards were built under.  The values DuckDB reports back are logged to stderr only.  Shards and every
other written value are unchanged by these options; the manifest differs from an older build only in this script's
own ``bindings.code_sha256`` (``CODE`` includes the script), which ``verify_cache`` does not compare.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
for p in (_HERE.parent, _HERE.parents[3] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transgenic.evidence import b6_bindings as BD  # noqa: E402
from transgenic.evidence import b6_obs_cache as OC  # noqa: E402
from transgenic.evidence import b6_track_spec as TS  # noqa: E402

CODE = [_HERE, Path(OC.__file__), Path(TS.__file__), Path(BD.__file__)]


def default_roles_loader(path):
    from transgenic.evidence import b6_roles  # Writer A (plan §3); fails loudly when absent

    return b6_roles.load_roles(path)


def parse_zero(items):
    out = {}
    for item in items or ():
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise SystemExit(f"--zero {item!r}: expected population:species:tsv[:qc]")
        pop, sp, tsv = parts[:3]
        qc = parts[3] if len(parts) == 4 else tsv + ".qc.json"
        if pop not in OC.POPULATIONS:
            raise SystemExit(f"--zero population {pop!r} not in {OC.POPULATIONS}")
        out.setdefault(pop, {})[sp] = (Path(tsv), Path(qc))
    return out


def parse_refs(items):
    out = {}
    for item in items or ():
        sp, sep, path = item.partition("=")
        if not sep or not sp or not path:
            raise SystemExit(f"--reference {item!r}: expected SPECIES=FASTA")
        if sp in out:
            raise SystemExit(f"--reference {sp} given twice")
        out[sp] = _Ref(path)
    return out


class _Ref:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = OC.sha256_file(self.path)


def lit(value):
    return "'" + str(value).replace("'", "''") + "'"


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


class DuckDBSettings:
    """DuckDB resource settings applied to the evidence connection (as b6_merge_evidence_db.py / b6_validate_corpus.py);
    ``applied`` holds the values DuckDB reports back.  SET works on a read-only connection; nothing here touches the
    cache: the values are logged to stderr only."""

    SETTINGS = ("memory_limit", "threads", "temp_directory")

    def __init__(self, memory_limit=None, threads=None, temp_directory=None):
        self.memory_limit, self.threads, self.temp_directory = memory_limit, threads, temp_directory
        self.applied = None

    def apply(self, con):
        if self.memory_limit is not None:
            con.execute(f"SET memory_limit = {lit(self.memory_limit)}")
        if self.threads is not None:                      # >= 1 by argparse (_positive_int)
            con.execute(f"SET threads = {int(self.threads)}")
        if self.temp_directory is not None:
            con.execute(f"SET temp_directory = {lit(self.temp_directory)}")
        self.applied = dict(zip(self.SETTINGS, con.execute(
            "SELECT " + ", ".join(f"current_setting('{s}')" for s in self.SETTINGS)).fetchone()))


def spill_dir(temp_dir, out):
    """(spill directory, owned): ``--temp-dir`` as given (kept), else a fresh ``b6_build_obs_cache.spill-*`` directory
    inside the parent of ``--out`` (the cache's GPFS, not the node's /tmp), to be removed by the caller.  Its path is
    printed to stderr first: a kill (OOM, SIGKILL) skips the cleanup and the directory must be deleted by hand."""
    if temp_dir is not None:
        spill = Path(temp_dir).absolute()
        print(f"duckdb spill directory: {spill} (--temp-dir; not removed)", file=sys.stderr, flush=True)
        return spill, False
    parent = Path(out).absolute().parent
    parent.mkdir(parents=True, exist_ok=True)          # CacheWriter creates it moments later anyway
    spill = Path(tempfile.mkdtemp(prefix="b6_build_obs_cache.spill-", dir=str(parent)))
    print(f"duckdb spill directory: {spill} (removed on normal exit; delete it by hand after a kill)",
          file=sys.stderr, flush=True)
    return spill, True


def extract_by_species(con, roles, family, species, writer, *, log=None):
    """Extract and write each species (sorted) one chromosome at a time through ``writer``; returns (drops, libraries)
    equal to the one-shot ``extract_rows``: drop molecule ids and library pairs are unioned across chromosomes, then
    counted/sorted.  The coverage and molecule-state checks run once, before anything is written."""
    OC.check_species_coverage(con, species)
    OC.check_molecule_states(con)
    out = log or sys.stderr
    drops, libs = {}, OC.population_libraries([], family)
    for sp in sorted(species):
        est_mol = OC.est_molecules(con, sp)             # species-level: EST members may align on other chromosomes
        sp_drops, sp_libs, n_sp = {}, {}, 0
        for chrom in OC.species_chroms(con, sp):
            rows, d, lb = OC.extract_rows(con, roles, family, species=sp, chrom=chrom, check_states=False,
                                          est_mol=est_mol, raw_drops=True)
            bad = {(r["species"], r["chrom"]) for r in rows} - {(sp, chrom)}
            if bad:
                raise OC.CacheError(f"{sp} {chrom}: extracted rows of other shard keys {sorted(bad)}")
            writer.add_rows(rows)
            n_sp += len(rows)
            print(f"b6_build_obs_cache: chrom {sp} {chrom}: {len(rows)} rows", file=out, flush=True)
            del rows
            for k, ids in d.items():
                sp_drops.setdefault(k, set()).update(ids)
            for pop, per in lb.items():
                for lsp, pairs in per.items():
                    sp_libs.setdefault(pop, {}).setdefault(lsp, set()).update(map(tuple, pairs))
        print(f"b6_build_obs_cache: species {sp}: {n_sp} rows", file=out, flush=True)
        foreign = sorted(({k[0] for k in sp_drops} | {lsp for per in sp_libs.values() for lsp in per}) - {sp})
        if foreign:
            raise OC.CacheError(f"{sp}: drops or libraries of other species {foreign}")
        drops.update((k, len(ids)) for k, ids in sp_drops.items())
        for pop, per in sp_libs.items():
            libs[pop].update((lsp, sorted(pairs)) for lsp, pairs in per.items())
    return dict(sorted(drops.items())), {pop: dict(sorted(per.items())) for pop, per in libs.items()}


def main(argv=None, *, roles_loader=default_roles_loader, leakage_rerun=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--family", required=True, choices=OC.FAMILIES)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--reference", action="append", required=True,
                    help="SPECIES=FASTA, one per covered species; verified against C0 inputs.reference")
    ap.add_argument("--corpus", help="B6 corpus (train family)")
    ap.add_argument("--freeze", help="corpus freeze JSON (train family; b6_corpus_repair record)")
    ap.add_argument("--zero", action="append", help="population:species:tsv[:qc]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory-limit", help="DuckDB memory_limit of the evidence connection, e.g. 120GB (default: "
                                           "DuckDB's host-memory share, ~80%% of the node, above a Slurm cgroup); "
                                           "result sets do not spill, so it must exceed the largest per-chromosome "
                                           "result set plus working memory (one chromosome's Python rows come on top)")
    ap.add_argument("--threads", type=_positive_int, help="DuckDB threads (default: DuckDB's)")
    ap.add_argument("--temp-dir", help="DuckDB spill directory, kept afterwards (default: b6_build_obs_cache.spill-* "
                                       "inside the parent of --out, removed on normal exit; after an OOM kill "
                                       "delete it by hand)")
    a = ap.parse_args(argv)
    if a.family == "train" and not (a.corpus and a.freeze):
        raise SystemExit("--family train needs --corpus and --freeze")
    if a.family == "eval" and (a.corpus or a.freeze):
        raise SystemExit("--family eval takes no --corpus/--freeze (no corpus membership at evaluation)")
    roles = roles_loader(a.roles)
    refs = parse_refs(a.reference)
    zero = parse_zero(a.zero)
    freeze = a.freeze
    want = {"train": "training", "eval": "eval_only"}[a.family]
    kw = {} if leakage_rerun is None else {"leakage_rerun": leakage_rerun}
    try:
        ident = BD.verify_upstream(accept={want}, evidence=a.evidence, roles=roles, zero_lists=zero, freeze=freeze,
                                   corpus=a.corpus, references=refs, stage="build", **kw)
    except BD.BindingError as exc:
        raise SystemExit(f"b6_build_obs_cache: upstream verification failed: {exc}") from exc
    import duckdb

    spill, own_spill = spill_dir(a.temp_dir, a.out)
    try:
        ev_sha, con = BD.open_snapshot(a.evidence, expected=ident["evidence_sha256"])
        try:
            duck = DuckDBSettings(a.memory_limit, a.threads, spill)
            try:
                duck.apply(con)                         # before anything is written: a rejected setting leaves no cache
            except duckdb.Error as exc:                 # e.g. a malformed --memory-limit rejected by SET
                raise SystemExit(f"b6_build_obs_cache: duckdb settings rejected (--memory-limit {a.memory_limit!r}, "
                                 f"--threads {a.threads!r}, --temp-dir {a.temp_dir!r}): {exc}") from exc
            print("b6_build_obs_cache: duckdb settings " + json.dumps(duck.applied, sort_keys=True, default=str),
                  file=sys.stderr, flush=True)
            writer = OC.CacheWriter(a.out, family=a.family)
            drops, libs = extract_by_species(con, roles, a.family, {sp for sp, _src in ident["coverage"]}, writer)
        finally:
            con.close()
    finally:
        if own_spill:
            shutil.rmtree(spill, ignore_errors=True)
    bindings = {"evidence_sha256": ev_sha, "roles_sha256": roles.sha256,
                "reference_sha256": ident["reference_sha256"],
                "code_sha256": {p.name: OC.sha256_file(p) for p in CODE}}
    if a.corpus:
        bindings["corpus_sha256"] = ident["corpus_sha256"]
        bindings["freeze_sha256"] = OC.sha256_file(a.freeze)
    man = writer.finish(libraries=libs, drops=drops, bindings=bindings, zero_lists=zero)
    BD.verify_cache(OC.ObsCache(a.out), ident)             # the published cache re-verifies against its upstream
    print(json.dumps({"out": a.out, "shards": len(man["shards"]), "rows": sum(s["rows"] for s in man["shards"]),
                      "drops": man["drops"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
