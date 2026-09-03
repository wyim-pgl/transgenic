"""Tests for the torch-free B5 trainer runtime (issue #17)."""
import ast
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def rt():
    return _load(ROOT / "src" / "transgenic" / "training" / "b5_runtime.py", "b5_runtime")


def test_load_frozen_config_and_model_kwargs(rt):
    cfg = rt.load_b5_config(str(ROOT / "configs" / "b5_400m_v1.json"))
    kw = rt.model_kwargs(cfg)
    assert kw["encoder_d_model"] == 768 and kw["decoder_d_model"] == 1536 and kw["encoder_layers"] == 12 and kw["decoder_layers"] == 12
    assert kw["encoder_attention_heads"] == 6 and len(kw["attention_window"]) == 12 and kw["attention_window"][0] == 1024
    assert kw["encoder_ffn_dim"] == 3072 and kw["decoder_ffn_dim"] == 3072   # published recipe (d013418 used the config defaults)
    assert cfg["patience"] == 3 and cfg["optimizer"] == "AdamW" and cfg["seeds"]["primary"] == 123
    assert rt.accumulation_steps(cfg, 1) == 96 and rt.accumulation_steps(cfg, 4) == 24
    with pytest.raises(ValueError):
        rt.accumulation_steps(cfg, 5)


def test_wide_config_is_rejected(rt, tmp_path):
    cfg = json.load(open(ROOT / "configs" / "b5_400m_v1.json"))
    cfg["d_model_encoder"], cfg["encoder_layers"] = 1152, 16
    p = tmp_path / "wide.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        rt.load_b5_config(str(p))


def test_early_stopper_patience_three(rt):
    es = rt.EarlyStopper(patience=3)
    seq = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83]
    out = [es.update(i + 1, v) for i, v in enumerate(seq)]
    assert [o[0] for o in out] == [True, True, True, False, False, False]
    assert [o[1] for o in out] == [False, False, False, False, False, True]
    assert es.best_epoch == 3 and es.best == 0.8
    s = es.state()
    es2 = rt.EarlyStopper(patience=3)
    es2.load_state(s)
    assert es2.best_epoch == 3 and es2.bad_epochs == 3


def test_checkpoint_layout_atomic_rename_best_and_done(rt, tmp_path):
    lay = rt.CheckpointLayout(str(tmp_path / "seed123"))
    tmp = lay.begin_epoch(1)
    assert tmp.endswith("epoch_01.tmp") and os.path.isdir(tmp)
    (Path(tmp) / "model.safetensors").write_bytes(b"x")
    final = lay.finish_epoch(1, eval_loss=0.9, train_loss=1.0, is_best=True)
    assert final.endswith("epoch_01") and not os.path.exists(tmp)
    assert json.load(open(Path(final) / "eval.json"))["eval_loss"] == 0.9
    assert os.readlink(tmp_path / "seed123" / "best") == "epoch_01"
    lay.begin_epoch(2)
    lay.finish_epoch(2, eval_loss=0.95, train_loss=0.9, is_best=False)
    assert lay.completed_epochs() == [1, 2] and lay.latest_epoch() == 2
    assert os.readlink(tmp_path / "seed123" / "best") == "epoch_01"
    assert lay.resume_dir("auto") is None  # no accelerate_state saved
    os.makedirs(Path(lay.epoch_dir(2)) / "accelerate_state")
    assert lay.resume_dir("auto").endswith("epoch_02")
    lay.write_state({"epoch": 2}); assert lay.read_state()["epoch"] == 2
    lay.mark_done(); assert (tmp_path / "seed123" / "TRAINING_DONE").exists()


def test_split_row_numbers_from_b5_db(rt, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db = str(tmp_path / "t.db")
    con = duckdb.connect(db)
    con.sql("CREATE TABLE geneList (rn INT, species_id VARCHAR, split VARCHAR, gff VARCHAR, geneModel VARCHAR)")
    con.executemany("INSERT INTO geneList VALUES (?,?,?,?,?)", [
        [1, "Athaliana", "train", "g", "a"], [2, "Athaliana", "valid", "g", "b"], [3, "Osativa", "train", "g", "c"], [4, "Osativa", "test", "g", "d"]])
    con.close()
    assert rt.split_row_numbers(db, "train") == [1, 3] and rt.split_row_numbers(db, "valid") == [2]
    con = duckdb.connect(db); con.sql("INSERT INTO geneList VALUES (5, 'Zmays', 'train', 'g', 'Zm1')"); con.close()
    with pytest.raises(ValueError):
        rt.split_row_numbers(db, "train")


def test_split_row_numbers_refuses_legacy_db(rt, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db = str(tmp_path / "legacy.db")
    con = duckdb.connect(db); con.sql("CREATE TABLE geneList (rn INT, geneModel VARCHAR, gff VARCHAR)"); con.close()
    with pytest.raises(ValueError):
        rt.split_row_numbers(db, "train")


def test_parse_args_requires_seed_and_output_with_config(rt):
    a = rt.parse_args(["--db", "x.db", "--config", "c.json", "--seed", "456", "--output-dir", "runs/seed456", "--resume", "auto"])
    assert a.seed == 456 and a.resume == "auto" and a.benchmark_steps == 0
    with pytest.raises(SystemExit):
        rt.parse_args(["--db", "x.db", "--config", "c.json"])
    legacy = rt.parse_args(["--db", "x.db"])
    assert legacy.config is None


def test_benchmark_summary(rt):
    s = rt.benchmark_summary([1.0] * 30, [9600] * 30, rows_in_train=96000, rows_per_step=96, peak_mem_gb=20.5, warmup=10)
    assert s["steps_measured"] == 20 and s["steps_per_epoch"] == 1000 and s["hours_per_epoch"] == pytest.approx(1000 / 3600)
    assert s["tokens_per_sec"] == 9600


def test_resume_prefers_newest_state_and_config_check(rt, tmp_path):
    lay = rt.CheckpointLayout(str(tmp_path / "s"))
    lay.begin_epoch(1); os.makedirs(Path(lay.epoch_dir(1) + ".tmp") / "accelerate_state")
    json.dump({"global_step": 100}, open(Path(lay.epoch_dir(1) + ".tmp") / "meta.json", "w"))
    lay.finish_epoch(1, 0.9, 1.0, is_best=True)
    assert lay.resume_dir("auto").endswith("epoch_01")
    ls = Path(lay.latest_state_dir()); os.makedirs(ls / "accelerate_state"); json.dump({"global_step": 150, "epoch": 1, "step": 40}, open(ls / "meta.json", "w"))
    assert lay.resume_dir("auto").endswith("latest_state")          # mid-epoch state is newer
    json.dump({"global_step": 50}, open(ls / "meta.json", "w"))
    assert lay.resume_dir("auto").endswith("epoch_01")              # stale latest_state is ignored
    json.dump({"recipe": {"name": "x"}, "db": "/a.db", "seed": 123, "batch_size": 1, "accumulation_steps": 96, "max_epochs": 22, "patience": 3},
              open(tmp_path / "s" / "run_config.json", "w"))
    lay.check_run_config({"recipe": {"name": "x"}, "db": "/a.db", "seed": 123, "batch_size": 1, "accumulation_steps": 96, "max_epochs": 22, "patience": 3})
    with pytest.raises(RuntimeError):
        lay.check_run_config({"recipe": {"name": "y"}, "db": "/a.db", "seed": 123, "batch_size": 1, "accumulation_steps": 96, "max_epochs": 22, "patience": 3})
    with pytest.raises(RuntimeError):
        lay.check_run_config({"recipe": {"name": "x"}, "db": "/a.db", "seed": 456, "batch_size": 1, "accumulation_steps": 96, "max_epochs": 22, "patience": 3})


def test_split_row_numbers_filters_by_window_length(rt, tmp_path):
    """#18 needs per-tier throughput, so the split query can be restricted to one tile tier."""
    duckdb = pytest.importorskip("duckdb")
    db = tmp_path / "t.db"
    con = duckdb.connect(str(db))
    con.sql("CREATE TABLE geneList (rn INT, species_id VARCHAR, split VARCHAR, gff VARCHAR, start INT, fin INT, train_weight DOUBLE)")
    con.executemany("INSERT INTO geneList VALUES (?,?,?,?,?,?,?)",
                    [(1, "Athaliana", "train", "x", 0, 30720, 1.0), (2, "Athaliana", "train", "y", 0, 61440, 1.0),
                     (3, "Athaliana", "train", "z", 0, 129024, 1.0), (4, "Athaliana", "valid", "w", 0, 30720, 1.0)])
    con.close()
    assert rt.split_row_numbers(str(db), "train") == [1, 2, 3]
    assert rt.split_row_numbers(str(db), "train", window_len=30720) == [1]
    assert rt.split_row_numbers(str(db), "train", window_len=129024) == [3]
    assert rt.split_row_numbers(str(db), "valid", window_len=61440) == []


def test_parse_args_accepts_benchmark_tier(rt):
    a = rt.parse_args(["--db", "x.db", "--config", "c.json", "--seed", "123", "--output-dir", "o",
                       "--benchmark-steps", "50", "--benchmark-tier", "129024"])
    assert a.benchmark_steps == 50 and a.benchmark_tier == 129024
    assert rt.parse_args(["--db", "x.db"]).benchmark_tier is None


def test_oversized_batch_cap_follows_the_recipe(rt):
    """The trainer's guard must accept the tier it was configured for: 129,024-nt tiles are legal under tile6144-v3."""
    v3 = rt.load_b5_config(str(ROOT / "configs" / "b5_400m_win_v3.json"))
    v1 = rt.load_b5_config(str(ROOT / "configs" / "b5_400m_v1.json"))
    assert v3["max_encoder_seqlen"] == 129024 and v1["max_encoder_seqlen"] == 49152
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    # under a recipe the cap is the recipe's own window length; the 100,000 heuristic survives only on the legacy path
    assert "if ii.shape[1] > _max_seqlen" in src
    assert "elif ii.shape[0] * ii.shape[1] > 100_000" in src
    assert '_max_seqlen = int((b5_config or {}).get("max_encoder_seqlen", 49152))' in src


def test_trainer_does_not_require_wandb_at_import():
    """A benchmark or offline run passes --no-wandb, and the ACCESS container has no wandb; importing it at module
    load made every such run die with ModuleNotFoundError (measured on pgl-gpu 2026-09-02)."""
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    tree = ast.parse(src)
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top_imports if isinstance(n, ast.Import) for a in n.names}
    names |= {(n.module or "").split(".")[0] for n in top_imports if isinstance(n, ast.ImportFrom)}
    assert "wandb" not in names, "wandb must not be imported at module load"
    assert "def _wandb()" in src and "import wandb as _w" in src


def test_trainer_reads_a_real_cuda_property_and_tolerates_cpu():
    """`total_mem` does not exist on torch's device properties (`total_memory` does): the banner crashed every run
    before the first batch, and get_device_properties(0) also throws on a CPU-only host."""
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    assert "total_mem " not in src and ".total_mem /" not in src
    assert "gpu_props.total_memory" in src
    assert "if torch.cuda.is_available():" in src.split("gpu_props =")[0].rsplit("\n\n", 1)[-1]


def test_b5_run_never_skips_a_failing_batch():
    """Protocol A35. The loop caught every exception, logged one stderr line and continued: on the 129,024-nt tier
    the 4090 skipped 1,093 of 1,103 batches while the run looked healthy. Both reviews (Codex, Kimi K3) called for
    an immediate failure with an allowed-skip threshold of zero for the frozen-recipe path."""
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    tree = ast.parse(src)
    # the handler must not be a bare `except Exception` that continues
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and any(isinstance(n, ast.Continue) for n in ast.walk(node)):
            assert node.type is not None, "a bare except that continues is forbidden"
            names = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(node.type) if isinstance(n, ast.Attribute)}
            assert "Exception" not in names, "the batch handler must not swallow every exception and continue"
            assert "OutOfMemoryError" in names, "only CUDA OOM may be skipped, and only on the legacy path"
    # under a frozen recipe even an OOM raises, with the sample, the shapes and the memory state
    body = src.split("except torch.cuda.OutOfMemoryError as e:")[1][:2600]
    assert "if layout is not None:" in body and "raise RuntimeError(" in body
    assert "memory_allocated" in body and "protocol A35" in body


def test_accumulation_accounting_uses_completed_microbatches():
    """Both reviews: `(step + 1) % accumulation_steps` counts DataLoader indices, so a skipped batch silently
    changes the effective batch and the epoch-loss denominator."""
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    assert "(step + 1) % accumulation_steps" not in src, "optimizer steps must key off completed micro-batches"
    assert "micro_done" in src
    assert "total_loss / len(train_dl)" not in src, "the epoch loss must divide by the batches actually seen"
