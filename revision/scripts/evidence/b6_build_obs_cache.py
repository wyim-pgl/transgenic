#!/usr/bin/env python3
"""Build the B6 observation cache (plan §7.5) from a merged C0 snapshot.

    b6_build_obs_cache.py --evidence b6_evidence_merged.db --family train --roles DATASET_ROLES.b6_v1.tsv \
        --reference Athaliana=TAIR10.fa [--reference SP=FA ...] --corpus b5_full_b6_v1.db --freeze b6_freeze.json \
        --zero train_track:Athaliana:zero.train_track.Athaliana.tsv [--zero ...] --out cache_train_v1/

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
"""
from __future__ import annotations

import argparse
import json
import sys
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
    ev_sha, con = BD.open_snapshot(a.evidence, expected=ident["evidence_sha256"])
    try:
        writer = OC.CacheWriter(a.out, family=a.family)
        drops, libs = extract_by_species(con, roles, a.family, {sp for sp, _src in ident["coverage"]}, writer)
    finally:
        con.close()
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
