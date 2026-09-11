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
    m = re.search(r"^  TRAIN_RC=0\n  while :; do\n.*?^  done\n", src, re.S | re.M)
    assert m, "wait loop not found in train_b5.slurm"
    return "\n".join(line[2:] for line in m.group(0).splitlines()) + "\n"   # dedent the inner loop


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


def _decide(tmp_path: Path, *, rc: int, err_before: str = "", err_after: str = "", env: dict | None = None,
            forced_preempt: bool = False) -> dict:
    """Run the decision block with a stub sbatch; return exit code, stdout, stub args/env, marker state."""
    run = tmp_path / "run"; run.mkdir(exist_ok=True)
    if forced_preempt:
        (run / "FORCED_PREEMPT.1").write_text("")        # job-scoped marker; the harness uses SLURM_JOB_ID=1
    err = run / "train.err"
    err.write_text(err_before)
    off = err.stat().st_size
    with err.open("a") as fh:
        fh.write(err_after)
    binq = tmp_path / "bin"; binq.mkdir(exist_ok=True)
    stub = binq / "sbatch"
    stub.write_text('#!/bin/bash\necho "SBATCH_ARGS: $*"\necho "SBATCH_ENV_EXCLUDE: ${EXCLUDE_NODES:-<unset>}"\necho "Submitted batch job 999"\n')
    stub.chmod(0o755)
    harness = "set -euo pipefail\n"        # the real script runs under errexit: a stray non-zero status must not end it
    harness += f'PATH="{binq}:$PATH"\nRUN_HOST="{run}"\nERR_OFF={off}\nTRAIN_RC={rc}\n'
    harness += 'WATCHDOG=${WATCHDOG:-}\n'
    harness += 'WORK=/w; REPO_DIR=/w/repo; SEED=456; GPUS=4; SLURM_JOB_ID=1; SLURM_NODELIST=gh121\nFORCED="$RUN_HOST/FORCED_PREEMPT.$SLURM_JOB_ID"\n'
    harness += 'CHAIN_N=${CHAIN_N:-1}; CHAIN_MAX=${CHAIN_MAX:-8}\n'
    harness += _decision_block()
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30,
                         env={**{"PATH": "/usr/bin:/bin"}, **(env or {})})
    return {"code": out.returncode, "out": out.stdout + out.stderr,
            "failed_marker": (run / "TRAINING_FAILED").exists(),
            "forced_marker": (run / "FORCED_PREEMPT.1").exists()}


LAUNCH = "  what():  CUDA error: unspecified launch failure\n"


def test_launch_failure_after_the_in_allocation_relaunch_resubmits_same_link_elsewhere(tmp_path):
    # the in-allocation relaunch already retried on this node; the cross-allocation retry excludes it
    r = _decide(tmp_path, rc=1, err_after="x\n" + LAUNCH)
    assert "SBATCH_ARGS:" in r["out"] and "--exclude=gh121" in r["out"] and "CHAIN_N=1," in r["out"]
    assert "LAUNCH_RETRY=1" in r["out"] and "--dependency=afterany:1" in r["out"]
    assert r["code"] != 0 and not r["failed_marker"]


def test_a_second_cross_allocation_launch_failure_is_terminal(tmp_path):
    r = _decide(tmp_path, rc=1, err_after=LAUNCH, env={"LAUNCH_RETRY": "1"})
    assert "SBATCH_ARGS:" not in r["out"] and r["failed_marker"] and r["code"] != 0


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
    r = _decide(tmp_path, rc=1, err_after=LAUNCH, env={"EXCLUDE_NODES": "gh062"})
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


# ---------------------------------------------------------------------------------------------------
# Pre-limit signal forwarding (2026-09-09, jobs 3119780/3119848/3119984): SIGTERM must reach the rank processes only -- not the
# container, not the launcher, not the DataLoader workers that share the trainer's command line.
# ---------------------------------------------------------------------------------------------------

def _rank_finder() -> str:
    src = SCRIPT.read_text()
    m = re.search(r"^_descendants\(\) .*?^}\n", src, re.S | re.M)
    assert m, "_descendants/_ranks not found in train_b5.slurm"
    return m.group(0)


def test_usr1_targets_only_the_ranks_under_the_launcher(tmp_path):
    # Fake tree: container -> launcher ("accelerate launch") -> rank (trainer script) -> worker (same script).
    train = tmp_path / "train"; train.mkdir()
    rank = train / "train_HyenaTransgenic.py"
    rank.write_text('#!/bin/bash\nif [ "${1:-}" = --worker ]; then sleep 30; exit 0; fi\n'
                    'trap \'echo RANK_GOT_TERM > "$OUT/got"; exit 0\' TERM\n'
                    f'bash "{rank}" --worker &\nwhile :; do sleep 0.2; done\n')
    launcher = tmp_path / "accelerate"
    launcher.write_text(f'#!/bin/bash\n# accelerate launch --num_processes=1 train/train_HyenaTransgenic.py\nbash "{rank}" &\nwait\n')
    container = tmp_path / "container"
    # like the real job: bash -lc "... accelerate launch ... train/train_HyenaTransgenic.py ..." wraps the launcher,
    # and the launcher's own argv names the trainer script -- neither may be signalled (rehearsal 3119819).
    container.write_text(f'#!/bin/bash\nbash -lc "bash \\"{launcher}\\" launch --num_processes=1 {rank}" &\nwait\n')
    for f in (rank, launcher, container):
        f.chmod(0o755)
    out = tmp_path / "out"; out.mkdir()
    harness = f'export OUT="{out}"\nbash "{container}" > /dev/null 2>&1 < /dev/null &\nTRAIN_PID=$!\nsleep 1\n' + _rank_finder()
    harness += ('R=$(_ranks); echo "RANKS=$R"; n=0; for p in $R; do n=$((n+1)); done; echo "NRANKS=$n"\n'
                'for p in $R; do kill -TERM "$p"; done\nsleep 1\n'
                'wait "$TRAIN_PID"; echo "CONTAINER_RC=$?"\n'
                'for p in $(_descendants "$TRAIN_PID") "$TRAIN_PID"; do kill -KILL "$p" 2>/dev/null; done; true\n')
    res = subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30)
    assert "NRANKS=1" in res.stdout, res.stdout + res.stderr
    assert (out / "got").exists(), "the rank's TERM handler did not fire"
    assert "CONTAINER_RC=0" in res.stdout, "the container/launcher must end cleanly after the rank exits (138 = launcher was signalled)"


def test_forced_preemption_resubmits_the_next_link_and_clears_the_marker(tmp_path):
    # The watchdog killed a trainer that ignored USR1 (rc 137): not a failure, the next link resumes latest_state.
    r = _decide(tmp_path, rc=137, forced_preempt=True)
    assert "SBATCH_ARGS:" in r["out"] and "CHAIN_N=2," in r["out"] and "--dependency=afterany:1" in r["out"]
    assert not r["failed_marker"] and not r["forced_marker"] and r["code"] == 0


def test_forced_preemption_at_the_chain_limit_stays_terminal_and_consumes_the_marker(tmp_path):
    r = _decide(tmp_path, rc=137, forced_preempt=True, env={"CHAIN_N": "8", "CHAIN_MAX": "8"})
    assert "SBATCH_ARGS:" not in r["out"] and r["code"] != 0 and not r["forced_marker"]


def test_forced_preemption_wins_over_a_launch_error_signature_at_the_cap(tmp_path):
    # Codex review: with both a forced marker and a fresh launch-failure signature at CHAIN_N=CHAIN_MAX the old
    # ordering fell through to the launch retry and submitted the capped link again.
    r = _decide(tmp_path, rc=137, forced_preempt=True, err_after=LAUNCH, env={"CHAIN_N": "8", "CHAIN_MAX": "8"})
    assert "SBATCH_ARGS:" not in r["out"] and not r["forced_marker"]


def test_a_finished_watchdog_does_not_end_the_link_under_errexit(tmp_path):
    # rehearsal 3119991: `[ -n "$WATCHDOG" ] && kill $WATCHDOG` failed (the subshell had exited) and errexit
    # killed the batch script before the resubmit decision.
    r = _decide(tmp_path, rc=137, forced_preempt=True, env={"WATCHDOG": "999999"})
    assert "SBATCH_ARGS:" in r["out"] and "CHAIN_N=2," in r["out"] and r["code"] == 0


# ---------------------------------------------------------------------------------------------------
# Pre-queued chain (A45, 2026-09-11; Codex review): with CHAIN_PRESUBMITTED=1 no branch may call sbatch; a finished
# or failed run cancels the queued successors (found by --comment=seed$SEED), a node launch failure pushes the
# exclusion onto them with scontrol update, and clean/forced boundaries simply end the link.
# ---------------------------------------------------------------------------------------------------

def _decide_pre(tmp_path: Path, *, rc: int, err_after: str = "", env: dict | None = None, forced_preempt: bool = False,
                done: bool = False, successors: str = "7 seed456\n8 seed456\n9 seed789\n") -> dict:
    """_decide with CHAIN_PRESUBMITTED=1 and stub squeue/scancel/scontrol; the stub queue lists `successors`."""
    binq = tmp_path / "bin"; binq.mkdir(exist_ok=True)
    (binq / "squeue").write_text(f'#!/bin/bash\nprintf "%s" "{successors}"\n')
    (binq / "scancel").write_text('#!/bin/bash\necho "SCANCEL: $*"\n')
    (binq / "scontrol").write_text('#!/bin/bash\necho "SCONTROL: $*"\n')
    for f in ("squeue", "scancel", "scontrol"):
        (binq / f).chmod(0o755)
    if done:
        run = tmp_path / "run"; run.mkdir(exist_ok=True); (run / "TRAINING_DONE").write_text("")
    r = _decide(tmp_path, rc=rc, err_after=err_after, forced_preempt=forced_preempt,
                env={**{"CHAIN_PRESUBMITTED": "1", "USER": "t"}, **(env or {})})
    return r


def test_presubmitted_clean_link_does_not_resubmit(tmp_path):
    r = _decide_pre(tmp_path, rc=0)
    assert "SBATCH_ARGS:" not in r["out"] and "SCANCEL" not in r["out"] and r["code"] == 0
    assert "pre-queued next link continues" in r["out"]


def test_presubmitted_forced_preemption_does_not_resubmit_and_clears_the_marker(tmp_path):
    r = _decide_pre(tmp_path, rc=137, forced_preempt=True)
    assert "SBATCH_ARGS:" not in r["out"] and r["code"] == 0 and not r["forced_marker"] and not r["failed_marker"]


def test_presubmitted_finished_run_cancels_only_this_seeds_queued_links(tmp_path):
    r = _decide_pre(tmp_path, rc=0, done=True)
    assert "SCANCEL: 7" in r["out"] and "SCANCEL: 8" in r["out"] and "SCANCEL: 9" not in r["out"]
    assert "SBATCH_ARGS:" not in r["out"] and r["code"] == 0


def test_presubmitted_trainer_failure_marks_and_cancels_the_successors(tmp_path):
    r = _decide_pre(tmp_path, rc=1, err_after="Traceback (most recent call last):\nValueError: boom\n")
    assert r["failed_marker"] and r["code"] != 0 and "SCANCEL: 7" in r["out"] and "SBATCH_ARGS:" not in r["out"]


def test_presubmitted_launch_failure_excludes_the_node_on_the_queued_links(tmp_path):
    r = _decide_pre(tmp_path, rc=1, err_after="x\n" + LAUNCH, env={"EXCLUDE_NODES": "gh[062,093]"})
    assert "SBATCH_ARGS:" not in r["out"] and r["code"] != 0 and not r["failed_marker"]
    assert "SCONTROL: update JobId=7 ExcNodeList=gh[062,093],gh121" in r["out"]
    assert "SCONTROL: update JobId=8 ExcNodeList=gh[062,093],gh121" in r["out"] and "JobId=9" not in r["out"]


def test_presubmitted_mode_does_not_change_the_self_resubmitting_chain(tmp_path):
    # CHAIN_PRESUBMITTED unset: the 2026-09-09/10 behaviour is byte-for-byte the same decision
    r = _decide(tmp_path, rc=0)
    assert "SBATCH_ARGS:" in r["out"] and "CHAIN_N=2," in r["out"]


def _guard_block() -> str:
    src = SCRIPT.read_text()
    m = re.search(r'^if \[ -f "\$RUN_HOST/TRAINING_DONE" \]; then echo "training already finished.*?^export EXCLUDE_NODES\n', src, re.S | re.M)
    assert m, "start-of-link guard block not found in train_b5.slurm"
    return m.group(0)


def _guard_run(tmp_path: Path, *, running_jobs: str = "", job_id: str = "5", exc: str = "(null)",
               env: dict | None = None) -> subprocess.CompletedProcess:
    run = tmp_path / "run"; run.mkdir(exist_ok=True)
    binq = tmp_path / "bin"; binq.mkdir(exist_ok=True)
    # squeue -j ID -h -o %T: RUNNING for a listed id, empty for a purged one; SQUEUE_FAIL=<text> simulates an outage
    (binq / "squeue").write_text(f'#!/bin/bash\n[ -n "${{SQUEUE_FAIL:-}}" ] && {{ echo "$SQUEUE_FAIL"; exit 1; }}\nfor j in {running_jobs}; do [ "$j" = "$2" ] && echo RUNNING; done; true\n')
    (binq / "scontrol").write_text(f'#!/bin/bash\necho "JobId={job_id} ExcNodeList={exc} NumNodes=1"\n')
    for f in ("squeue", "scontrol"):
        (binq / f).chmod(0o755)
    harness = f'set -euo pipefail\nPATH="{binq}:$PATH"\nRUN_HOST="{run}"\nSLURM_JOB_ID={job_id}\n' + _guard_block() + 'echo "GUARD_PASSED exclude=[$EXCLUDE_NODES]"\n'
    return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30,
                          env={**{"PATH": "/usr/bin:/bin"}, **(env or {})})


def test_guard_runs_before_anything_touches_the_gpus():
    src = SCRIPT.read_text()
    guard = src.index('if [ -f "$RUN_HOST/TRAINING_DONE" ]; then echo "training already finished')
    assert guard < src.index("gpu warm-up ok") and guard < src.index("SYNC_DEST") and guard < src.index("_launch()")


def test_guard_finished_run_exits_zero_without_starting(tmp_path):
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "TRAINING_DONE").write_text("")
    r = _guard_run(tmp_path)
    assert r.returncode == 0 and "nothing to do" in r.stdout and "GUARD_PASSED" not in r.stdout


def test_guard_failed_run_is_refused(tmp_path):
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "TRAINING_FAILED").write_text("")
    r = _guard_run(tmp_path)
    assert r.returncode == 1 and "REFUSED" in r.stderr and "GUARD_PASSED" not in r.stdout


def test_guard_refuses_while_another_link_of_the_seed_is_running(tmp_path):
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "LINK_ACTIVE.4").write_text("")
    r = _guard_run(tmp_path, running_jobs="4")
    assert r.returncode == 3 and "is still RUNNING" in r.stderr and "GUARD_PASSED" not in r.stdout
    assert (tmp_path / "run" / "LINK_ACTIVE.4").exists(), "the running link's marker must be left alone"


def test_guard_ignores_a_stale_marker_and_records_its_own(tmp_path):
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "LINK_ACTIVE.4").write_text("")
    r = _guard_run(tmp_path, running_jobs="")
    assert r.returncode == 0 and "GUARD_PASSED" in r.stdout
    assert not (tmp_path / "run" / "LINK_ACTIVE.4").exists()
    # the EXIT trap removes this link's own marker at the end of the harness
    assert not (tmp_path / "run" / "LINK_ACTIVE.5").exists()


def test_guard_fails_closed_when_squeue_cannot_answer(tmp_path):
    # Codex 2026-09-11: an outage must not read as "not running"
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "LINK_ACTIVE.4").write_text("")
    r = _guard_run(tmp_path, env={"SQUEUE_FAIL": "slurm_load_jobs error: Unable to contact slurm controller"})
    assert r.returncode == 3 and "cannot confirm" in r.stderr and "GUARD_PASSED" not in r.stdout
    assert (tmp_path / "run" / "LINK_ACTIVE.4").exists()


def test_guard_treats_a_purged_job_id_as_stale(tmp_path):
    (tmp_path / "run").mkdir(); (tmp_path / "run" / "LINK_ACTIVE.4").write_text("")
    r = _guard_run(tmp_path, env={"SQUEUE_FAIL": "slurm_load_jobs error: Invalid job id specified"})
    assert r.returncode == 0 and "GUARD_PASSED" in r.stdout and not (tmp_path / "run" / "LINK_ACTIVE.4").exists()


def test_submit_chain_refuses_a_non_integer_seed(tmp_path):
    r = subprocess.run(["bash", str(ROOT / "deploy" / "deltaai" / "submit_chain.sh"), "45,6"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "SIF": str(SCRIPT), "REPO_DIR": str(ROOT)}, timeout=30)
    assert r.returncode == 3 and "SEED must be an integer" in r.stderr


def test_guard_seeds_exclusions_from_the_submission(tmp_path):
    r = _guard_run(tmp_path, exc="gh[062,093]")
    assert "GUARD_PASSED exclude=[gh[062,093]]" in r.stdout
    r2 = _guard_run(tmp_path)
    assert "GUARD_PASSED exclude=[]" in r2.stdout


# ---------------------------------------------------------------------------------------------------
# Runtime identity (A44, 2026-09-10): the first link records image/checkout/GPUs/CPUs/torch/policies and a later
# link under a different identity is refused unless RUNTIME_OVERRIDE=1.
# ---------------------------------------------------------------------------------------------------

def _identity_block() -> str:
    src = SCRIPT.read_text()
    m = re.search(r"^IDENT=.*?^echo \"runtime identity: .*?\n", src, re.S | re.M)
    assert m, "runtime identity block not found in train_b5.slurm"
    return m.group(0)


def _identity_run(tmp_path: Path, *, gpus: str, env: dict | None = None) -> subprocess.CompletedProcess:
    run = tmp_path / "run"; run.mkdir(exist_ok=True)
    sif = tmp_path / "img.sif"; sif.write_text("x"); (tmp_path / "img.sif.sha256").write_text("abc123  img.sif\n")
    repo = tmp_path / "repo"
    if not (repo / ".git").exists():
        repo.mkdir(); subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "f").write_text("1"); subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    binq = tmp_path / "bin"; binq.mkdir(exist_ok=True)
    (binq / "apptainer").write_text("#!/bin/bash\necho 2.14.0a0\n"); (binq / "apptainer").chmod(0o755)
    harness = f'set -euo pipefail\nPATH="{binq}:$PATH"\nRUN_HOST="{run}"\nSIF="{sif}"\nREPO_DIR="{repo}"\nGPUS={gpus}\n' + _identity_block()
    return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=30,
                          env={**{"PATH": "/usr/bin:/bin"}, **(env or {})})


def test_runtime_identity_is_recorded_then_enforced(tmp_path):
    first = _identity_run(tmp_path, gpus="4")
    assert first.returncode == 0 and "runtime identity: sif=img.sif sif_sha256=abc123" in first.stdout
    assert "gpus=4" in (tmp_path / "run" / "runtime_identity.txt").read_text()
    same = _identity_run(tmp_path, gpus="4")
    assert same.returncode == 0
    changed = _identity_run(tmp_path, gpus="1")
    assert changed.returncode == 3 and "REFUSED: runtime identity differs" in changed.stderr
    forced = _identity_run(tmp_path, gpus="1", env={"RUNTIME_OVERRIDE": "1"})
    assert forced.returncode == 0 and "gpus=1" in (tmp_path / "run" / "runtime_identity.txt").read_text()
