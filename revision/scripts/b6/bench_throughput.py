#!/usr/bin/env python3
"""B6 loader throughput bench (plan §7.8, CP2 for Writer B): loader samples/s at the 129,024-nt tier vs the model's
optimizer-step sample rate.  Gate (plan default): loader >= 1.5 x model samples/s.

    bench_throughput.py --corpus b5_full_b6_v1.db --cache cache_train_v1 --tier 129024 --workers 8 \
        --samples 400 --model-samples-per-s <from Writer C's memory/step bench> --out bench_throughput.json

Runs B6Dataset + b6_collate_fn in ``--workers`` processes (spawn), each worker owning its own corpus/cache handles
(no shared state; draws are pure functions of (aug_seed, row, epoch)).  CPU only; no torch needed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
for p in (_ROOT / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GATE = 1.5


def _load_dataset_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("b6_dataset_bench", _ROOT / "src" / "transgenic" / "datasets" /
                                                  "b6_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _char_encoder(seq):
    import numpy as np

    lut = np.full(256, 11, np.int64)
    for c, v in zip(b"ACGTN", (7, 8, 9, 10, 11)):
        lut[c] = v
    return lut[np.frombuffer(seq.upper().encode("ascii"), np.uint8)], 4


def _worker(args):
    corpus, cache_dir, keys, batch, mode, aug_seed = args
    sys.path.insert(0, str(_ROOT / "src"))
    from transgenic.evidence import b6_context as CX
    from transgenic.evidence import b6_obs_cache as OC

    DS = _load_dataset_module()
    cache = OC.ObsCache(cache_dir)
    ds = DS.B6Dataset(CX.B6Corpus(corpus, cache), cache, mode=mode, arm="B6", aug_seed=aug_seed,
                      encode_dna=_char_encoder, training=False)
    t0 = time.perf_counter()
    n = 0
    for i in range(0, len(keys), batch):
        DS.b6_collate_fn([ds[k] for k in keys[i:i + batch]])
        n += len(keys[i:i + batch])
    return n, time.perf_counter() - t0


def run(corpus, cache_dir, *, tier, workers, samples, batch=1, mode="train_track", aug_seed=20261037):
    import duckdb

    split = {"train_track": "train", "valid_track": "valid"}[mode]
    con = duckdb.connect(str(corpus), read_only=True)
    rns = [r[0] for r in con.execute("SELECT rn FROM geneList WHERE split = ? AND fin - start = ? ORDER BY rn",
                                     [split, int(tier)]).fetchall()]
    con.close()
    if not rns:
        raise SystemExit(f"no {split} rows at tier {tier}")
    keys = [(rns[i % len(rns)], i // len(rns)) for i in range(samples)]
    parts = [(str(corpus), str(cache_dir), keys[w::workers], batch, mode, aug_seed) for w in range(workers)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_worker, parts)
    wall = time.perf_counter() - t0
    n = sum(r[0] for r in res)
    return {"tier": tier, "workers": workers, "samples": n, "wall_s": wall, "loader_samples_per_s": n / wall,
            "per_worker_s": [r[1] for r in res]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tier", type=int, default=129024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--model-samples-per-s", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = run(a.corpus, a.cache, tier=a.tier, workers=a.workers, samples=a.samples, batch=a.batch)
    rec["model_samples_per_s"] = a.model_samples_per_s
    rec["ratio"] = rec["loader_samples_per_s"] / a.model_samples_per_s
    rec["gate"] = GATE
    rec["pass"] = rec["ratio"] >= GATE
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec, sort_keys=True))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
