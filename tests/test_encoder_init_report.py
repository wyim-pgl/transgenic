"""Encoder initialisation provenance (protocol A34): the partial pretrained load must be recorded, not just logged.

Measured 2026-09-03 with `configs/b5_400m_win_v3.json`: the target encoder has 339 tensors, the checkpoint
`LongSafari/hyenadna-large-1m-seqlen-hf` has 227 (d_model 256, n_layer 8 versus the recipe's 768 and 12), so
80 tensors load exactly, 147 are tiled/cropped into shape and 112 (all of `layers.8`–`layers.11`) keep the target
model's random initialisation. Both reviews (Codex, Kimi K3) agreed: keep the initialisation exactly as it is —
it is the published 400M recipe and changing it would change the numbers — but stop treating a silent stderr line
as the record, and fail closed when the load is not the expected one.

These tests cover the pure-python parts (audit summary, expectation check, report writing); the load itself needs
torch and the checkpoint and is exercised on the GPU box.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "transgenic" / "model" / "encoder_init.py"


def _load():
    if not TARGET.exists():
        pytest.fail(f"{TARGET} does not exist yet (RED state)", pytrace=False)
    mod = types.ModuleType("encoder_init")
    mod.__file__ = str(TARGET)
    sys.modules["encoder_init"] = mod
    exec(compile(TARGET.read_text(), str(TARGET), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def ei():
    return _load()


def _rows():
    """A miniature of the real audit: 2 exact, 2 adapted, 3 missing in the top layers."""
    return [
        {"key": "backbone.embeddings.word_embeddings.weight", "status": "adapted", "src": [16, 256], "dst": [16, 768]},
        {"key": "backbone.layers.0.mixer.in_proj.weight", "status": "adapted", "src": [768, 256], "dst": [2304, 768]},
        {"key": "backbone.layers.0.norm.weight", "status": "exact", "src": [768], "dst": [768]},
        {"key": "backbone.ln_f.weight", "status": "exact", "src": [768], "dst": [768]},
        {"key": "backbone.layers.8.mixer.in_proj.weight", "status": "missing", "src": None, "dst": [2304, 768]},
        {"key": "backbone.layers.9.mixer.in_proj.weight", "status": "missing", "src": None, "dst": [2304, 768]},
        {"key": "backbone.layers.11.mixer.out_proj.bias", "status": "missing", "src": None, "dst": [768]},
    ]


def test_summary_counts_and_missing_layer_range(ei):
    s = ei.summarise(_rows())
    assert s["target_tensors"] == 7 and s["loaded"] == 4
    assert s["exact"] == 2 and s["adapted"] == 2 and s["missing"] == 3
    assert s["missing_layers"] == [8, 9, 11]                 # the layers the checkpoint does not have
    assert s["missing_outside_layers"] == []                 # nothing missing outside the layer stack
    assert 0.0 < s["loaded_fraction"] < 1.0


def test_expectation_gate_accepts_the_known_shape_and_rejects_anything_else(ei):
    rows = _rows()
    exp = {"target_tensors": 7, "loaded": 4, "exact": 2, "adapted": 2, "missing": 3, "missing_layers_max": 11}
    assert ei.check_expected(ei.summarise(rows), exp) == []
    # a checkpoint that silently loads less must be caught
    fewer = [dict(r, status="missing", src=None) if r["status"] == "exact" else r for r in rows]
    v = ei.check_expected(ei.summarise(fewer), exp)
    assert v and any("exact" in x for x in v)
    # missing tensors outside the top layers are never expected (that would be a naming break, not a size gap)
    broken = [dict(r, key="backbone.embeddings.word_embeddings.weight", status="missing", src=None) if r["status"] == "adapted" else r for r in rows]
    v2 = ei.check_expected(ei.summarise(broken), exp)
    assert v2 and any("outside" in x for x in v2)


def test_zero_load_is_always_a_failure(ei):
    none_loaded = [dict(r, status="missing", src=None) for r in _rows()]
    s = ei.summarise(none_loaded)
    assert s["loaded"] == 0
    assert ei.check_expected(s, None), "a load of zero tensors must fail even without an expectation"


def test_report_is_written_as_an_artifact(ei, tmp_path):
    out = tmp_path / "encoder_init_report.json"
    rows = _rows()
    path = ei.write_report(str(out), rows, checkpoint="LongSafari/hyenadna-large-1m-seqlen-hf",
                           target={"d_model": 768, "n_layer": 12, "max_seq_len": 129024},
                           source={"d_model": 256, "n_layer": 8, "max_seq_len": 1000002})
    d = json.loads(Path(path).read_text())
    assert d["checkpoint"] == "LongSafari/hyenadna-large-1m-seqlen-hf"
    assert d["target"]["n_layer"] == 12 and d["source"]["n_layer"] == 8
    assert d["summary"]["missing"] == 3 and len(d["tensors"]) == 7
    assert d["timestamp"] and d["note"].startswith("partial")           # the record says it is partial, not full
    assert all(set(r) >= {"key", "status", "src", "dst"} for r in d["tensors"])


def test_expected_manifest_for_the_b5_recipe_is_committed(ei):
    """The numbers measured on 2026-09-03 are frozen next to the recipe so a drift is caught at load time."""
    exp = ei.EXPECTED["LongSafari/hyenadna-large-1m-seqlen-hf@768x12"]
    assert exp == {"target_tensors": 339, "loaded": 227, "exact": 80, "adapted": 147, "missing": 112, "missing_layers_max": 11}


def test_loader_uses_the_gate_and_never_falls_back_to_random():
    """The old handler caught every exception and continued with a randomly initialised encoder."""
    src = (ROOT / "src" / "transgenic" / "model" / "modeling_HyenaTransgenic.py").read_text()
    assert "using random init" not in src, "the silent random-init fallback must be gone"
    assert "_enc_init.check_expected" in src and "_enc_init.write_report" in src
    assert "TRANSGENIC_ALLOW_ENCODER_DRIFT" in src            # deliberate override, documented
    assert "raise RuntimeError(" in src.split("except Exception as exc:")[1][:400]
