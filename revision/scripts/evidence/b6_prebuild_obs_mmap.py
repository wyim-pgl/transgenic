#!/usr/bin/env python3
"""Pre-build the memory-mapped derived shard files of a B6 observation cache (run once, before training).

For every manifest shard: verify the parquet sha256 against the manifest, then build ``<mmap dir>/<shard sha256>.arrow``
(or check an existing one's recorded source sha256, rows and version).  The cache directory is not written.  The mmap
dir defaults to ``<cache parent>/<cache name>.mmap_v1`` (``--mmap-dir`` or env B6_OBS_CACHE_MMAP_DIR override it; the
trainer must see the same directory).  Prints one JSON line per shard and a summary line.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transgenic.evidence import b6_obs_cache as OC  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="observation cache directory (holds manifest.json)")
    ap.add_argument("--mmap-dir", default=None, help="derived file directory (default: env or <cache>.mmap_v1)")
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(rec):
        rec = dict(rec, elapsed_s=round(time.time() - t0, 1))
        print(json.dumps(rec, sort_keys=True), flush=True)

    recs = OC.prebuild_mmap(args.cache, mmap_dir=args.mmap_dir, log=log)
    print(json.dumps({"shards": len(recs), "built": sum(r["built"] for r in recs),
                      "rows": sum(r["rows"] for r in recs), "derived_bytes": sum(r["derived_bytes"] for r in recs),
                      "elapsed_s": round(time.time() - t0, 1)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
