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


# ---------------------------------------------------------------------------------------------------
# The resubmit decision (2026-09-09 retry branch): a node's `unspecified launch failure` resubmits the
# same link with the node excluded, bounded by LAUNCH_RETRY_MAX; every other failure stays terminal.
# ---------------------------------------------------------------------------------------------------

def _decision_block() -> str:
    src = SCRIPT.read_text()
    m = re.search(r"^# chain: resubmit unless finished.*?^fi\n", src, re.S | re.M)
    assert m, "resubmit decision block not found in train_b5.slurm"
    return m.group(0)


def _decide(tmp_path: Path, *, rc: int, err_before: str = "", err_after: str = "", env: dict | None = None) -> dict:
    """Run the decision block with a stub sbatch; return exit code, stdout, stub args/env, marker state."""
    run = tmp_path / "run"; run.mkdir()
    err = run / "train.err"
    err.write_text(err_before)
    off = err.stat().st_size
    with err.open("a") as fh:
        fh.write(err_after)
    binq = tmp_path / "bin"; binq.mkdir()
    stub = binq / "sbatch"
    stub.write_text('#!/bin/bash\necho "SBATCH_ARGS: $*"\necho "SBATCH_ENV_EXCLUDE: ${EXCLUDE_NODES:-<unset>}"\necho "Submitted batch job 999"\n')
    stub.chmod(0o755)
    harness = "set -uo pipefail\n"
    harness += f'PATH="{binq}:$PATH"\nRUN_HOST="{run}"\nERR_OFF={off}\nTRAIN_RC={rc}\n'
    harness += 'WORK=/w; SEED=456; GPUS=4; SLURM_JOB_ID=1; SLURM_NODELIST=gh121\n'
    harness += 'CHAIN_N=${CHAIN_N:-1}; CHAIN_MAX=${CHAIN_MAX:-8}\n'
    harness += _decision_block()
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30,
                         env={**{"PATH": "/usr/bin:/bin"}, **(env or {})})
    return {"code": out.returncode, "out": out.stdout + out.stderr,
            "failed_marker": (run / "TRAINING_FAILED").exists()}


LAUNCH = "  what():  CUDA error: unspecified launch failure\n"


def test_launch_failure_in_this_attempt_resubmits_same_link_in_place_first(tmp_path):
    r = _decide(tmp_path, rc=1, err_after="x\n" + LAUNCH)
    assert "SBATCH_ARGS:" in r["out"] and "--exclude" not in r["out"] and "CHAIN_N=1," in r["out"]
    assert "LAUNCH_RETRY=1" in r["out"] and "--dependency=afterany:1" in r["out"]


def test_second_launch_failure_excludes_the_node(tmp_path):
    r = _decide(tmp_path, rc=1, err_after=LAUNCH, env={"LAUNCH_RETRY": "1"})
    assert "--exclude=gh121" in r["out"] and "LAUNCH_RETRY=2" in r["out"] and not r["failed_marker"]
    assert r["code"] != 0, "the dead link must still exit non-zero"
    assert not r["failed_marker"], "a retried link must not be marked TRAINING_FAILED"


def test_launch_failure_text_from_a_previous_attempt_is_not_a_retry(tmp_path):
    r = _decide(tmp_path, rc=3, err_before=LAUNCH, err_after="")
    assert "SBATCH_ARGS:" not in r["out"]
    assert r["failed_marker"] and r["code"] != 0


def test_other_failures_stay_terminal(tmp_path):
    r = _decide(tmp_path, rc=1, err_after="Traceback (most recent call last):\nValueError: boom\n")
    assert "SBATCH_ARGS:" not in r["out"] and r["failed_marker"] and r["code"] != 0


def test_retry_budget_is_bounded(tmp_path):
    r = _decide(tmp_path, rc=1, err_after=LAUNCH, env={"LAUNCH_RETRY": "2"})
    assert "SBATCH_ARGS:" not in r["out"] and r["failed_marker"]


def test_exclusions_accumulate_and_travel_in_the_environment(tmp_path):
    r = _decide(tmp_path, rc=1, err_after=LAUNCH, env={"EXCLUDE_NODES": "gh062", "LAUNCH_RETRY": "1"})
    assert "--exclude=gh062,gh121" in r["out"]
    assert "SBATCH_ENV_EXCLUDE: gh062,gh121" in r["out"], "the comma list must reach sbatch via the environment, not --export"
    assert "EXCLUDE_NODES=" not in r["out"].split("SBATCH_ARGS:")[1].split("\n")[0]


def test_signature_followed_by_a_large_log_still_matches(tmp_path):
    r = _decide(tmp_path, rc=1, err_after=LAUNCH + ("progress line\n" * 20000))
    assert "SBATCH_ARGS:" in r["out"] and "LAUNCH_RETRY=1" in r["out"]


def test_clean_link_resubmits_next_link_with_retry_reset_and_exclusions_kept(tmp_path):
    r = _decide(tmp_path, rc=0, env={"EXCLUDE_NODES": "gh062,gh121"})
    assert "CHAIN_N=2," in r["out"] and "LAUNCH_RETRY=0" in r["out"] and "--exclude=gh062,gh121" in r["out"]
    assert r["code"] == 0 and not r["failed_marker"]
