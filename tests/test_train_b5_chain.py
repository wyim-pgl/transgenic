"""The A28 job chain (deploy/deltaai/train_b5.slurm) must keep the trainer's real exit code.

The loop `if wait "$PID"; then ...; fi; TRAIN_RC=$?` recorded 0 after every crash, because the status
of an `if` whose condition failed is 0. A dead trainer was then resubmitted up to CHAIN_MAX times
instead of writing TRAINING_FAILED (Codex review 2026-09-09; reproduced with a child exiting 17).

The loop is extracted from the script text and run in bash against real children, so the test is
about the shell form itself, not a re-implementation of it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "deltaai" / "train_b5.slurm"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def _wait_loop() -> str:
    src = SCRIPT.read_text()
    m = re.search(r"^TRAIN_RC=0\nwhile :; do\n.*?^done\n", src, re.S | re.M)
    assert m, "wait loop not found in train_b5.slurm"
    return m.group(0)


def _run(child: str, signal_after: float | None = None) -> str:
    """Run the extracted loop under `set -euo pipefail` with `child` as the background trainer."""
    harness = "set -euo pipefail\n"
    harness += f"({child}) &\nTRAIN_PID=$!\n"
    harness += "trap 'echo USR1 >&2' USR1\n"
    if signal_after is not None:
        harness += f"(sleep {signal_after}; kill -USR1 $$) &\n"
    harness += _wait_loop()
    harness += 'echo "observed_rc=$TRAIN_RC"\n'
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30)
    return out.stdout + out.stderr


def test_wait_loop_keeps_a_failing_child_exit_code():
    assert "observed_rc=17" in _run("exit 17")


def test_wait_loop_reports_zero_for_a_clean_child():
    assert "observed_rc=0" in _run("exit 0")


def test_wait_loop_survives_usr1_and_keeps_the_final_code():
    # USR1 interrupts the first wait (status 138); the loop must re-wait and keep the child's real code.
    out = _run("sleep 1.5; exit 5", signal_after=0.3)
    assert "USR1" in out, "the trap did not fire; the test did not exercise the interrupted wait"
    assert "observed_rc=5" in out


def test_wait_loop_survives_usr1_with_a_clean_child():
    out = _run("sleep 1.5; exit 0", signal_after=0.3)
    assert "USR1" in out
    assert "observed_rc=0" in out


def test_failed_link_exits_nonzero():
    """sacct must show the link FAILED when the trainer died; a COMPLETED link hid the crash."""
    src = SCRIPT.read_text()
    branch = src.split('training failed (rc=$TRAIN_RC)')[1].split("elif")[0]
    assert "exit 1" in branch
