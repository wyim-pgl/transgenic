"""Provenance for the partial pretrained initialisation of the Hyena encoder (protocol A34).

The B5 recipe asks for an encoder of `d_model` 768 and 12 layers, while every public HyenaDNA checkpoint is 128 or
256 wide and at most 8 layers deep. `HyenaEncoder.__init__` therefore builds the target model first and copies what
it can from the checkpoint: identical key and shape verbatim, mismatched shapes tiled and cropped by
`_adapt_tensor_shape`, and keys the checkpoint does not have (the top layers) left at the target model's random
initialisation. Measured for `LongSafari/hyenadna-large-1m-seqlen-hf` at 768x12 on 2026-09-03:
339 target tensors, 227 loaded (80 exact, 147 adapted), 112 missing — all of them `layers.8`–`layers.11`.

Two independent reviews (Codex, Kimi K3, 2026-09-03) agreed that the initialisation itself must not change: it is
the published 400M recipe and the reported 92 % F1 was obtained with it, so altering it would change the numbers and
break parity. What must change is the record. A stderr line is not evidence for a pre-registered study, and a load
that silently degrades (a renamed checkpoint, a missing file, a different revision) currently continues into a run
whose encoder is mostly random. This module writes the audit as an artifact and fails closed when the load is not
the expected one.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, List, Optional, Sequence

# Frozen expectations per (checkpoint, target size). A load that does not match exactly is refused.
EXPECTED: Dict[str, Dict[str, int]] = {
    "LongSafari/hyenadna-large-1m-seqlen-hf@768x12": {
        "target_tensors": 339, "loaded": 227, "exact": 80, "adapted": 147, "missing": 112, "missing_layers_max": 11,
    },
}

NOTE = ("partial pretrained initialisation: tensors present in the checkpoint with a matching shape are copied "
        "verbatim, mismatched shapes are tiled and cropped per dimension, and keys absent from the checkpoint keep "
        "the target model's random initialisation (published 400M recipe, unchanged)")


def expectation_key(checkpoint: str, d_model: int, n_layer: int) -> str:
    return f"{checkpoint}@{d_model}x{n_layer}"


def _layer_index(key: str) -> Optional[int]:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def summarise(rows: Sequence[Dict]) -> Dict:
    """Counts and the layer range of the missing tensors, from the per-tensor audit."""
    exact = sum(1 for r in rows if r["status"] == "exact")
    adapted = sum(1 for r in rows if r["status"] == "adapted")
    missing_rows = [r for r in rows if r["status"] == "missing"]
    layers = sorted({i for i in (_layer_index(r["key"]) for r in missing_rows) if i is not None})
    outside = sorted({r["key"] for r in missing_rows if _layer_index(r["key"]) is None})
    total = len(rows)
    return {"target_tensors": total, "loaded": exact + adapted, "exact": exact, "adapted": adapted,
            "missing": len(missing_rows), "missing_layers": layers, "missing_outside_layers": outside,
            "loaded_fraction": (exact + adapted) / total if total else 0.0}


def check_expected(summary: Dict, expected: Optional[Dict[str, int]]) -> List[str]:
    """Violations that must stop the run. A load of zero tensors always fails, expectation or not."""
    v: List[str] = []
    if summary["loaded"] == 0:
        v.append("no pretrained tensor was loaded into the encoder")
    if summary["missing_outside_layers"]:
        v.append(f"{len(summary['missing_outside_layers'])} tensors missing outside the layer stack "
                 f"(e.g. {summary['missing_outside_layers'][0]}); the checkpoint keys do not match this architecture")
    if expected is None:
        return v
    for field in ("target_tensors", "loaded", "exact", "adapted", "missing"):
        if summary[field] != expected[field]:
            v.append(f"{field}: got {summary[field]}, expected {expected[field]}")
    if summary["missing_layers"] and summary["missing_layers"][-1] > expected["missing_layers_max"]:
        v.append(f"missing layers reach {summary['missing_layers'][-1]}, expected at most {expected['missing_layers_max']}")
    return v


def write_report(path: str, rows: Sequence[Dict], checkpoint: str, target: Dict, source: Dict,
                 violations: Optional[Sequence[str]] = None) -> str:
    """Write the per-tensor audit as a run artifact (A34: the Methods/supplementary record of the partial load)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    doc = {"checkpoint": checkpoint, "target": target, "source": source, "summary": summarise(rows),
           "violations": list(violations or []), "note": NOTE,
           "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tensors": [{"key": r["key"], "status": r["status"], "src": r["src"], "dst": r["dst"]} for r in rows]}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)
    return path
