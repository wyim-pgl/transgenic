"""Tests for 60_epoch_budget.py (written before the implementation)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "60_epoch_budget.py"
import types

eb = types.ModuleType("epoch_budget")
eb.__file__ = str(SCRIPT)
sys.modules["epoch_budget"] = eb
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), eb.__dict__)


def power_curve(n, a=1.20, b=1.5, c=0.9):
    return [a + b * (e ** -c) for e in range(1, n + 1)]


def test_parse_trainer_log_line_with_tensors():
    # Arrange: the exact stderr format of train/train_HyenaTransgenic.py
    text = ("epoch=0: train_ppl=tensor(3.2, device='cuda:0'), train_epoch_loss=tensor(1.1631, device='cuda:0'), "
            "eval_ppl=tensor(2.9), eval_epoch_loss=tensor(1.0647)\n"
            "junk line\n"
            "epoch=1: train_ppl=tensor(2.8), train_epoch_loss=tensor(1.0296), eval_ppl=tensor(2.7), eval_epoch_loss=tensor(0.9933)\n")
    # Act
    run = eb.parse_trainer_log(text)
    # Assert: epochs are reported 1-based
    assert run.epochs == [1, 2]
    assert run.eval_loss == pytest.approx([1.0647, 0.9933])
    assert run.train_loss == pytest.approx([1.1631, 1.0296])


def test_parse_csv_with_optional_train_column(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("epoch,eval_loss,train_loss\n1,1.0,1.2\n2,0.9,1.0\n3,0.85,0.9\n")
    run = eb.load_run(p)
    assert run.epochs == [1, 2, 3]
    assert run.eval_loss == [1.0, 0.9, 0.85]
    assert run.train_loss == [1.2, 1.0, 0.9]


def test_best_and_patience_stop_on_overfitting_curve():
    # eval improves to epoch 6 then rises; patience 3 -> stop after epoch 9
    ev = [1.0, 0.9, 0.8, 0.75, 0.72, 0.70, 0.71, 0.72, 0.74, 0.76]
    res = eb.analyse(eb.Run("x", list(range(1, 11)), ev, None), patience=3, min_delta=0.0, plateau_tol=0.005, margin=2, max_epochs=22)
    assert res["best_epoch"] == 6
    assert res["early_stop_epoch"] == 9
    assert res["overfit_epoch"] == 9  # three consecutive epochs above the best
    assert res["recommended_max_epochs"] == 9 + 2


def test_plateau_curve_without_overfitting_recommends_plateau_plus_margin():
    ev = power_curve(20, c=2.5)  # monotone, flattens by epoch ~9
    res = eb.analyse(eb.Run("x", list(range(1, 21)), ev, None), patience=3, min_delta=0.0, plateau_tol=0.005, margin=2, max_epochs=22)
    assert res["early_stop_epoch"] is None
    assert 7 <= res["plateau_epoch"] <= 10
    assert res["recommended_max_epochs"] == res["plateau_epoch"] + 2
    assert res["recommended_max_epochs"] <= 22


def test_power_law_fit_recovers_parameters_and_extrapolates():
    ev = power_curve(12, a=1.2, b=1.5, c=0.9)
    fit = eb.fit_power_law(list(range(1, 13)), ev)
    assert fit["a"] == pytest.approx(1.2, abs=0.02)
    assert fit["c"] == pytest.approx(0.9, abs=0.1)
    # marginal gain per epoch below 0.5% of the asymptote happens later than the data end
    e = eb.epoch_where_gain_below(fit, rel_tol=0.005, max_search=200)
    assert e is not None and e > 12


def test_short_run_reports_insufficient_data():
    res = eb.analyse(eb.Run("x", [1, 2], [1.0, 0.9], None), patience=3, min_delta=0.0, plateau_tol=0.005, margin=2, max_epochs=22)
    assert res["status"] == "insufficient_data"


def test_cli_end_to_end_json(tmp_path):
    p = tmp_path / "seed123.csv"
    rows = ["epoch,eval_loss"] + [f"{i},{v:.5f}" for i, v in enumerate(power_curve(10), 1)]
    p.write_text("\n".join(rows) + "\n")
    out = subprocess.run([sys.executable, str(SCRIPT), "--csv", str(p), "--patience", "3"], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    assert data["runs"][0]["name"] == "seed123"
    assert "recommended_max_epochs" in data["summary"]
