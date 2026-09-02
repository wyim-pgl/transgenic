"""Torch-free runtime pieces for the B5 trainer (issue #17): config loading, split selection,
best-validation early stopping, per-epoch checkpoint layout, argument parsing, benchmark summary.

Kept free of torch so the logic is unit-tested wherever duckdb is available; the trainer
(train/train_HyenaTransgenic.py) imports and uses these helpers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REQUIRED_CONFIG_KEYS = ("d_model_encoder", "d_model_decoder", "encoder_layers", "decoder_layers", "encoder_attention_heads",
                        "decoder_attention_heads", "attention_window", "dropout", "lr", "weight_decay", "effective_batch_size",
                        "max_epochs", "seeds", "encoder_model")
DEFAULT_PATIENCE = 3


def load_b5_config(path: str) -> Dict:
    with open(path) as fh:
        cfg = json.load(fh)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["d_model_encoder"] == 1152 or cfg["encoder_layers"] == 16:
        raise ValueError("this looks like the 1.17B wide configuration; B5 must use the 400M recipe (configs/b5_400m_v1.json)")
    cfg.setdefault("patience", DEFAULT_PATIENCE)
    cfg.setdefault("optimizer", "AdamW")
    cfg.setdefault("lr_warmup_fraction", 0.05)
    cfg.setdefault("encoder_n_layer", cfg["encoder_layers"])
    cfg.setdefault("window_policy", "sym6144-v1")
    cfg.setdefault("max_encoder_seqlen", 49152 if cfg["window_policy"] == "sym6144-v1" else 129024)
    return cfg


def model_kwargs(cfg: Dict) -> Dict:
    """Map the frozen JSON to HyenaTransgenicConfig keyword arguments."""
    return {
        "d_model": cfg["d_model_encoder"], "encoder_d_model": cfg["d_model_encoder"], "decoder_d_model": cfg["d_model_decoder"],
        "encoder_layers": cfg["encoder_layers"], "decoder_layers": cfg["decoder_layers"], "encoder_n_layer": cfg["encoder_n_layer"],
        "encoder_ffn_dim": cfg["d_model_encoder"] * 4, "decoder_ffn_dim": cfg["d_model_decoder"] * 4,
        "attention_window": list(cfg["attention_window"]), "dropout": cfg["dropout"],
        "encoder_attention_heads": cfg["encoder_attention_heads"], "decoder_attention_heads": cfg["decoder_attention_heads"],
        "encoder_model": cfg["encoder_model"], "max_encoder_seqlen": int(cfg.get("max_encoder_seqlen", 49152)),
    }


def accumulation_steps(cfg: Dict, batch_size: int) -> int:
    eff = int(cfg["effective_batch_size"])
    if eff % batch_size:
        raise ValueError(f"effective batch {eff} is not a multiple of the micro-batch {batch_size}")
    return eff // batch_size


class EarlyStopper:
    """Minimum mean validation loss with patience (compare loss to loss, never to perplexity)."""

    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.bad_epochs = 0

    def update(self, epoch: int, loss: float) -> Tuple[bool, bool]:
        """Return (is_new_best, should_stop)."""
        if self.best is None or loss < self.best * (1 - self.min_delta):
            self.best, self.best_epoch, self.bad_epochs = float(loss), epoch, 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience

    def state(self) -> Dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs, "patience": self.patience}

    def load_state(self, s: Dict) -> None:
        self.best, self.best_epoch, self.bad_epochs = s.get("best"), s.get("best_epoch"), int(s.get("bad_epochs", 0))


class CheckpointLayout:
    """<run>/epoch_NN.tmp -> epoch_NN (atomic rename), best symlink, TRAINING_DONE marker."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def epoch_dir(self, epoch: int) -> str:
        return os.path.join(self.run_dir, f"epoch_{epoch:02d}")

    def begin_epoch(self, epoch: int) -> str:
        tmp = self.epoch_dir(epoch) + ".tmp"
        if os.path.isdir(tmp):
            import shutil
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        return tmp

    def finish_epoch(self, epoch: int, eval_loss: Optional[float], train_loss: float, extra: Optional[Dict] = None, is_best: bool = False) -> str:
        tmp, final = self.epoch_dir(epoch) + ".tmp", self.epoch_dir(epoch)
        payload = {"epoch": epoch, "eval_loss": eval_loss, "train_loss": train_loss, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        payload.update(extra or {})
        with open(os.path.join(tmp, "eval.json"), "w") as fh:
            json.dump(payload, fh, indent=1)
        if os.path.isdir(final):
            import shutil
            shutil.rmtree(final)
        os.rename(tmp, final)
        if is_best:
            self.set_best(epoch)
        return final

    def set_best(self, epoch: int) -> None:
        link = os.path.join(self.run_dir, "best")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(self.epoch_dir(epoch)), link)

    def completed_epochs(self) -> List[int]:
        out = []
        for name in os.listdir(self.run_dir):
            if name.startswith("epoch_") and not name.endswith(".tmp") and os.path.isdir(os.path.join(self.run_dir, name)):
                out.append(int(name.split("_")[1]))
        return sorted(out)

    def latest_epoch(self) -> Optional[int]:
        eps = self.completed_epochs()
        return eps[-1] if eps else None

    def write_state(self, state: Dict) -> None:
        with open(os.path.join(self.run_dir, "trainer_state.json"), "w") as fh:
            json.dump(state, fh, indent=1)

    def read_state(self) -> Optional[Dict]:
        p = os.path.join(self.run_dir, "trainer_state.json")
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    def mark_done(self) -> None:
        with open(os.path.join(self.run_dir, "TRAINING_DONE"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S\n"))

    def resume_dir(self, mode: Optional[str]) -> Optional[str]:
        """'auto' -> latest completed epoch dir with an accelerate state, a path -> that path, None -> None."""
        if not mode:
            return None
        if mode != "auto":
            return mode
        for ep in reversed(self.completed_epochs()):
            d = self.epoch_dir(ep)
            if os.path.isdir(os.path.join(d, "accelerate_state")):
                return d
        return None


def split_row_numbers(db: str, split: str, excluded_species: Sequence[str] = ("Zmays",), require_labels: bool = True) -> List[int]:
    """Row numbers of one split from the frozen split column; refuses excluded species and NULL splits."""
    import duckdb
    con = duckdb.connect(db, read_only=True)
    try:
        cols = {r[1] for r in con.sql("PRAGMA table_info('geneList')").fetchall()}  # (cid, name, type, ...)
        if "split" not in cols:
            raise ValueError(f"{db} has no split column; it was not built by scripts/build_b5_database.py")
        n_null = con.sql("SELECT count(*) FROM geneList WHERE split IS NULL").fetchone()[0]
        if n_null:
            raise ValueError(f"{db}: {n_null} rows without a split value")
        for sp in excluded_species:
            n = con.sql("SELECT count(*) FROM geneList WHERE species_id = ?", params=[sp]).fetchone()[0]
            if n:
                raise ValueError(f"{db}: excluded species {sp} present ({n} rows)")
        q = "SELECT rn FROM geneList WHERE split = ?" + (" AND gff IS NOT NULL" if require_labels else "") + " ORDER BY rn"
        return [r[0] for r in con.sql(q, params=[split]).fetchall()]
    finally:
        con.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HyenaTransgenic (B5 recipe when --config is given)")
    p.add_argument("--db", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="frozen recipe JSON (configs/b5_400m_v1.json); omit for the legacy wide run")
    p.add_argument("--seed", type=int, default=None, help="B5 seed (123 primary, 456/789 confirmatory)")
    p.add_argument("--output-dir", type=str, default=None, help="run directory (epoch_NN/, best, TRAINING_DONE)")
    p.add_argument("--resume", type=str, default=None, help="'auto' or an epoch directory")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=None, help="derived from effective_batch_size when --config is given")
    p.add_argument("--max-epochs", type=int, default=None, help="override the config cap")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--benchmark-steps", type=int, default=0, help="run N optimizer steps, print throughput JSON, exit")
    p.add_argument("--save-every-n-steps", type=int, default=0)
    p.add_argument("--checkpoint-path", type=str, default="checkpoints_HyenaWide/", help="legacy path (no --config)")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args(argv)
    if a.config:
        if a.seed is None or a.output_dir is None:
            p.error("--config requires --seed and --output-dir")
    return a


def benchmark_summary(step_seconds: Sequence[float], tokens_per_step: Sequence[int], rows_in_train: int, rows_per_step: int,
                      peak_mem_gb: Optional[float] = None, warmup: int = 20) -> Dict:
    s = list(step_seconds)[warmup:] or list(step_seconds)
    t = list(tokens_per_step)[warmup:] or list(tokens_per_step)
    sec = sum(s) / len(s)
    toks = sum(t) / len(t)
    steps_per_epoch = max(1, -(-rows_in_train // rows_per_step))
    return {"steps_measured": len(s), "sec_per_step": sec, "tokens_per_sec": toks / sec if sec else None,
            "steps_per_epoch": steps_per_epoch, "hours_per_epoch": steps_per_epoch * sec / 3600.0, "peak_mem_gb": peak_mem_gb}
