#!/usr/bin/env python3
"""Decide how many epochs a TransGenic training run needs, from per-epoch validation loss.

Inputs (any number, mixed):
  --log  FILE   stderr/log of train/train_HyenaTransgenic.py
                (lines "epoch=N: train_ppl=..., train_epoch_loss=..., eval_ppl=..., eval_epoch_loss=...")
  --csv  FILE   columns epoch,eval_loss[,train_loss]  (header required)
  --jsonl FILE  one object per line with keys epoch, eval_loss[, train_loss]
Rules (frozen with configs/b5_400m_v1.json): checkpoint = minimum mean validation loss, patience 3.
Outputs JSON: per run best epoch, patience-rule stop epoch, plateau epoch (within --plateau-tol of the
best), overfitting onset, power-law extrapolation, and a recommended max_epochs = stop-or-plateau + margin.
No third-party dependencies; matplotlib is optional (--plot).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LINE_RE = re.compile(r"epoch=(\d+):.*?train(?:_epoch)?_loss=(?:tensor\()?([0-9.eE+-]+).*?eval(?:_epoch)?_loss=(?:tensor\()?([0-9.eE+-]+)")


@dataclass
class Run:
    name: str
    epochs: List[int]
    eval_loss: List[float]
    train_loss: Optional[List[float]]


def parse_trainer_log(text: str, name: str = "log") -> Run:
    epochs, ev, tr = [], [], []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)) + 1)  # trainer counts from 0; report 1-based
        tr.append(float(m.group(2)))
        ev.append(float(m.group(3)))
    if not epochs:
        raise ValueError(f"{name}: no 'epoch=N: ... eval_epoch_loss=' lines found")
    return Run(name, epochs, ev, tr)


def load_run(path: Path, kind: Optional[str] = None) -> Run:
    kind = kind or {".csv": "csv", ".jsonl": "jsonl"}.get(path.suffix.lower(), "log")
    name = path.stem
    if kind == "log":
        return parse_trainer_log(path.read_text(errors="replace"), name)
    rows: List[Dict[str, str]] = []
    if kind == "csv":
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
    elif kind == "jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        raise ValueError(f"unknown input kind {kind}")
    if not rows or "epoch" not in rows[0] or "eval_loss" not in rows[0]:
        raise ValueError(f"{path}: need columns epoch,eval_loss")
    rows.sort(key=lambda r: int(r["epoch"]))
    epochs = [int(r["epoch"]) for r in rows]
    ev = [float(r["eval_loss"]) for r in rows]
    has_tr = all(r.get("train_loss") not in (None, "") for r in rows)
    tr = [float(r["train_loss"]) for r in rows] if has_tr else None
    return Run(name, epochs, ev, tr)


def fit_power_law(epochs: List[int], loss: List[float]) -> Dict[str, float]:
    """Fit L(e) = a + b * e^-c by grid search over c with closed-form least squares for a, b."""
    best = None
    for i in range(1, 400):
        c = i / 100.0  # 0.01 .. 3.99
        x = [e ** -c for e in epochs]
        n = len(x)
        sx, sy = sum(x), sum(loss)
        sxx = sum(v * v for v in x)
        sxy = sum(v * w for v, w in zip(x, loss))
        den = n * sxx - sx * sx
        if abs(den) < 1e-12:
            continue
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        sse = sum((a + b * v - w) ** 2 for v, w in zip(x, loss))
        if best is None or sse < best["sse"]:
            best = {"a": a, "b": b, "c": c, "sse": sse}
    best["rmse"] = math.sqrt(best["sse"] / len(loss))
    return best


def epoch_where_gain_below(fit: Dict[str, float], rel_tol: float, max_search: int = 200) -> Optional[int]:
    """First epoch e at which the predicted improvement L(e-1)-L(e) falls below rel_tol * a."""
    a, b, c = fit["a"], fit["b"], fit["c"]
    if b <= 0 or a <= 0:
        return None
    for e in range(2, max_search + 1):
        gain = b * ((e - 1) ** -c - e ** -c)
        if gain < rel_tol * a:
            return e
    return None


def analyse(run: Run, patience: int, min_delta: float, plateau_tol: float, margin: int, max_epochs: int) -> Dict:
    n = len(run.epochs)
    out: Dict = {"name": run.name, "n_epochs": n}
    if n < 3:
        out["status"] = "insufficient_data"
        return out
    ev = run.eval_loss
    best_i = min(range(n), key=lambda i: ev[i])
    best = ev[best_i]
    out.update(best_epoch=run.epochs[best_i], best_eval_loss=best, final_eval_loss=ev[-1])

    # patience rule as the B5 trainer will apply it: stop after `patience` epochs without an improvement > min_delta (relative)
    stop = None
    cur_best, since = ev[0], 0
    for i in range(1, n):
        if ev[i] < cur_best * (1 - min_delta):
            cur_best, since = ev[i], 0
        else:
            since += 1
            if since >= patience:
                stop = run.epochs[i]
                break
    out["early_stop_epoch"] = stop

    # plateau: first epoch from which the eval loss stays within plateau_tol of the overall best
    plateau = None
    for i in range(n):
        if all(ev[j] <= best * (1 + plateau_tol) for j in range(i, n)):
            plateau = run.epochs[i]
            break
    out["plateau_epoch"] = plateau

    # overfitting onset: `patience` consecutive epochs above the best while train loss keeps falling (if known)
    overfit = None
    for i in range(best_i + 1, n):
        window = range(i - patience + 1, i + 1)
        if i - patience + 1 <= best_i:
            continue
        if all(ev[j] > best for j in window):
            if run.train_loss is None or all(run.train_loss[j] <= run.train_loss[j - 1] for j in window):
                overfit = run.epochs[i]
                break
    out["overfit_epoch"] = overfit

    k = min(3, n - 1)
    out["last_k_relative_improvement"] = (ev[-1 - k] - ev[-1]) / ev[-1 - k] if ev[-1 - k] else None
    fit = fit_power_law(run.epochs, ev)
    out["power_law"] = {kk: round(v, 6) for kk, v in fit.items()}
    out["extrapolated_diminishing_epoch"] = epoch_where_gain_below(fit, plateau_tol)

    anchor = stop if stop is not None else plateau
    rec = anchor + margin if anchor is not None else None
    if rec is not None:
        rec = min(rec, max_epochs)
    out["recommended_max_epochs"] = rec
    if stop is not None:
        out["verdict"] = f"patience-{patience} rule stops at epoch {stop}; best at {run.epochs[best_i]}"
        out["status"] = "converged"
    elif plateau is not None and plateau < run.epochs[-1]:
        out["verdict"] = f"no early stop; within {plateau_tol:.1%} of best from epoch {plateau}"
        out["status"] = "plateau"
    else:
        out["verdict"] = "still improving at the last epoch; extend the run"
        out["status"] = "improving"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], type=Path)
    ap.add_argument("--csv", action="append", default=[], type=Path)
    ap.add_argument("--jsonl", action="append", default=[], type=Path)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.0, help="relative improvement counted as progress")
    ap.add_argument("--plateau-tol", type=float, default=0.005, help="relative tolerance around the best loss")
    ap.add_argument("--margin", type=int, default=2, help="epochs added after the stop/plateau epoch")
    ap.add_argument("--max-epochs", type=int, default=22, help="hard cap (released recipe)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plot", type=Path)
    a = ap.parse_args(argv)
    runs = [load_run(p, "log") for p in a.log] + [load_run(p, "csv") for p in a.csv] + [load_run(p, "jsonl") for p in a.jsonl]
    if not runs:
        ap.error("give at least one --log/--csv/--jsonl")
    results = [analyse(r, a.patience, a.min_delta, a.plateau_tol, a.margin, a.max_epochs) for r in runs]
    recs = [r["recommended_max_epochs"] for r in results if r.get("recommended_max_epochs")]
    summary = {
        "runs": len(results),
        "recommended_max_epochs": max(recs) if recs else None,
        "rule": f"max over runs of (patience-{a.patience} stop epoch, else plateau epoch within {a.plateau_tol:.1%}) + {a.margin}, capped at {a.max_epochs}",
        "statuses": {r["name"]: r.get("status") for r in results},
    }
    report = {"summary": summary, "runs": results}
    text = json.dumps(report, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; no plot", file=sys.stderr)
            return 0
        fig, ax = plt.subplots(figsize=(6, 4))
        for r, res in zip(runs, results):
            ax.plot(r.epochs, r.eval_loss, marker="o", label=f"{r.name} eval")
            if r.train_loss:
                ax.plot(r.epochs, r.train_loss, ls="--", alpha=0.6, label=f"{r.name} train")
            if res.get("best_epoch"):
                ax.axvline(res["best_epoch"], color="grey", lw=0.8)
        ax.set_xlabel("epoch"); ax.set_ylabel("mean loss"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
