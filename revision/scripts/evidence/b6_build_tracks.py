#!/usr/bin/env python3
"""Dose-response evidence archive (plan §7.9; A46 rule 13).

For every evaluation window of a sha-bound window manifest, every fraction in {1, 0.5, 0.1, 0.01, 0} and every draw
d in 0..4 (aug_seed_d = 20260914 + d), resolve a fixed draw against the window's evidence owner, re-rasterise and
rebuild the chain memory (``eval_context`` -> ``rasterise`` / ``build_chains``), and archive per
(locus, orientation, fraction, draw): the retained molecule ids per source, chain truncation counts and the sha256
of the track and chain arrays.  ``b6_bindings.verify_inputs`` runs first (evidence family and gates, the reference
== C0 inputs.reference of the window species, zero lists, the cache binding); nothing is archived before it passes.  Ranks depend only on (aug_seed_d, species, molecule_id), so they are reused across
fractions and modes within a draw; the de novo mode is the all-zero input and is archived once per window.

    b6_build_tracks.py --cache cache_train_v1 --evidence merged.db --freeze freeze.json --corpus b5_full_b6_v1.db \
        --reference TAIR10.fa --panel b6_panels_v1.tsv \
        --panel-sha 79367154... --panel-id final --windows final_windows.tsv --generator centred30720-v1 \
        --roles DATASET_ROLES.b6_v1.tsv --out dose_final_v1.jsonl [--arrays dose_final_v1_arrays/]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for p in (_HERE.parent, _HERE.parents[3] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transgenic.evidence import b6_aug as AG  # noqa: E402
from transgenic.evidence import b6_bindings as BD  # noqa: E402
from transgenic.evidence import b6_chains as CH  # noqa: E402
from transgenic.evidence import b6_context as CX  # noqa: E402
from transgenic.evidence import b6_obs_cache as OC  # noqa: E402
from transgenic.evidence import b6_track_spec as TS  # noqa: E402
from transgenic.evidence import b6_tracks as TR  # noqa: E402


def arr_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dose_records(cache, ctx, *, draws=TS.DOSE_DRAWS, fractions=TS.DOSE_FRACTIONS, arrays=None):
    owner = AG.Owner.from_pairs(ctx.species, cache.owner_libraries("eval_input", ctx.species))
    locus = ctx.panel_locus[0] if ctx.panel_locus else ctx.row
    for d in draws:
        for f in fractions:
            draw = AG.resolve_fixed(TS.dose_aug_seed(d), owner, f)
            t, qc = TR.rasterise_with_qc(cache, ctx, draw)
            tile = CH.build_chains(cache, ctx, draw)
            rec = {"row": ctx.row, "locus": locus, "is_rc": ctx.is_rc, "fraction": f, "draw": d,
                   "aug_seed": TS.dose_aug_seed(d), "mode": "evidence",
                   "retained": qc["retained_molecules"], "chain_truncated": tile.qc["truncated"],
                   "chain_filled": tile.qc["filled"], "tracks_sha256": arr_sha(t),
                   "chains_sha256": arr_sha(tile.chain_pos, tile.chain_feat, tile.chain_mask)}
            if arrays is not None:
                name = f"{hashlib.sha256(ctx.row.encode()).hexdigest()[:16]}.f{f}.d{d}.npz"
                np.savez_compressed(Path(arrays) / name, tracks=t, chain_pos=tile.chain_pos,
                                    chain_feat=tile.chain_feat, chain_mask=tile.chain_mask)
                rec["arrays"] = name
            yield rec
    off = AG.resolve_off(owner)
    t = TR.rasterise(cache, ctx, off)
    tile = CH.build_chains(cache, ctx, off)
    yield {"row": ctx.row, "locus": locus, "is_rc": ctx.is_rc, "fraction": None, "draw": None, "aug_seed": None,
           "mode": "de_novo", "retained": {}, "chain_truncated": {}, "chain_filled": {},
           "tracks_sha256": arr_sha(t), "chains_sha256": arr_sha(tile.chain_pos, tile.chain_feat, tile.chain_mask)}


def main(argv=None, *, roles_loader=None, leakage_rerun=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--cache", "--evidence", "--reference", "--panel", "--panel-sha", "--panel-id", "--windows",
              "--generator", "--roles", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--freeze", help="corpus freeze JSON (training-family evidence, e.g. A. thaliana)")
    ap.add_argument("--corpus", help="B6 corpus (training-family evidence)")
    ap.add_argument("--arrays")
    a = ap.parse_args(argv)
    if roles_loader is None:
        from transgenic.evidence import b6_roles

        roles_loader = b6_roles.load_roles
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    if a.arrays:
        Path(a.arrays).mkdir(parents=True, exist_ok=False)
    cache = OC.ObsCache(a.cache)
    ref = CX.Reference(a.reference)
    panel = CX.Panel(a.panel, a.panel_sha, a.panel_id)
    wm = CX.WindowManifest(a.windows)
    roles = roles_loader(a.roles)
    species = sorted({w.species for w in wm.windows})
    if len(species) != 1:
        raise SystemExit(f"one species per dose-response run (one --reference), got {species}")
    kw = {} if leakage_rerun is None else {"leakage_rerun": leakage_rerun}
    ident = BD.verify_inputs(accept={"training", "eval_only"}, evidence=a.evidence, roles=roles, cache=cache,
                             freeze=a.freeze,
                             corpus=a.corpus, references={species[0]: ref}, panel=panel, window_manifest=wm, **kw)
    n = 0
    with open(out, "x") as fh:
        header = {"archive": "b6-dose-response-v1", "spec": TS.SPEC_SHAS, "cache_manifest_sha256": cache.manifest_sha256,
                  "reference_sha256": ref.sha256, "panel_sha256": panel.sha256, "window_manifest_sha256": wm.sha256,
                  "roles_sha256": roles.sha256, "fractions": list(TS.DOSE_FRACTIONS), "draws": list(TS.DOSE_DRAWS),
                  "identity": ident}
        fh.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        for w in wm.windows:
            ctx = CX.eval_context(w, reference=ref, panel=panel, roles=roles, generator=a.generator, cache=cache,
                                  window_manifest=wm)
            for rec in dose_records(cache, ctx, arrays=a.arrays):
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
                n += 1
    Path(str(out) + ".sha256").write_text(f"{OC.sha256_file(out)}  {out.name}\n")
    print(json.dumps({"out": str(out), "records": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
