# C2 continuation controller independent review — 2026-10-12

**Disposition: changes requested — two P2 findings.**

Reviewed `d332dcfa1709faf3056b43ab9b9b4034f12df1ad` against parent `1f4bb61`, using the complete supplied diff, the binding `CONTINUATION_IMPLEMENTATION_PLAN.md`, and both implementation/interface reports. The older design was consulted with the implementation plan's explicit rulings taking precedence. This review covers the host controller and its producer interface; it does not replace the separate allocation-helper review or live scientific gates.

## P2 — promotion does not atomically refuse an existing destination

Location: `continue_c2_history.py:142–145` (`rename_new`).

The destination absence check and `os.rename` are separate operations. If a destination directory appears between them, POSIX rename replaces it when it is empty. The two controller locks serialize cooperating controllers, but they do not turn this filesystem operation into the required atomic no-replacement operation. This affects both original200-to-backup and candidate410-to-canonical promotion. The binding plan explicitly requires refusal of even an empty destination; the parent confirmed that requirement applies beyond cooperating actors.

A focused probe used real temporary directories and the real `os.rename`; a wrapper inserted an empty destination immediately before the syscall. `rename_new` returned successfully, the destination inode was replaced, and the source disappeared. Observed output:

```text
{'race_destination_replaced': True, 'source_disappeared': True, 'moved_evidence': 'preserve'}
```

Deterministic reproducer, from the repository root:

```python
import importlib.util, tempfile
from pathlib import Path

p = Path('notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/continue_c2_history.py')
s = importlib.util.spec_from_file_location('ctl_review', str(p))
ctl = importlib.util.module_from_spec(s)
s.loader.exec_module(ctl)
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source, destination = root / 'source', root / 'destination'
    source.mkdir()
    (source / 'evidence').write_text('preserve')
    real_rename, inserted_inode = ctl.os.rename, []
    def insert_before_syscall(src, dst):
        destination.mkdir()
        inserted_inode.append(destination.stat().st_ino)
        return real_rename(src, dst)
    ctl.os.rename = insert_before_syscall
    try:
        ctl.rename_new(source, destination)
    finally:
        ctl.os.rename = real_rename
    assert destination.stat().st_ino != inserted_inode[0]
    assert not source.exists()
```

Required correction: use an atomic no-replacement rename, such as Linux `renameat2(RENAME_NOREPLACE)` through Python3.6 stdlib `ctypes`, and fail closed if the kernel/filesystem does not support it. Retain the existing same-filesystem check and parent fsyncs. Add a race-boundary regression proving both source and intervening destination survive when the destination appears. Actual shared-Lustre support remains a deployment smoke check.

## P2 — the configured submitted script need not be manifest-pinned

Location: `continue_c2_history.py:413–414` (`pins`), with configuration accepted at `:266–268` and the unchecked script selected at `:465`.

Each helper manifest is checked, but its members are never required to include `phase['script']`. The script can be a different WORK-relative file, including a file inside the declared helper directory omitted from its manifest. The accepted immutable configuration therefore permits submission of changed executable bytes while every source/helper pin still passes. This can arise from a staging/configuration mismatch; it does not require a scheduler compromise or a malicious same-UID actor. The script runs before its own allocation-side checks could protect the host submission contract.

A focused fixture initialized the controller with `validation.script='validation/unpinned.sbatch'`, retained the unchanged helper manifest containing only `run.sbatch`, changed the unpinned script after initialization, and called the normal preparation/execution boundary with fake Slurm. The controller submitted that changed file and advanced to `waiting_validation`:

```text
{'phase': 'waiting_validation',
 'submitted_script': '/tmp/.../validation/unpinned.sbatch',
 'manifest_covers_submitted_script': False}
```

Deterministic reproducer, from the repository root; it imports fixtures without running the suite:

```python
import importlib.util
from pathlib import Path

p = Path('notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/test_continue_c2_history.py')
s = importlib.util.spec_from_file_location('test_ctl_review', str(p))
t = importlib.util.module_from_spec(s)
s.loader.exec_module(t)
fixture = t.ControllerTests()
fixture.setUp()
try:
    config = fixture.config()
    script = fixture.root / 'validation/unpinned.sbatch'
    script.write_text('#!/bin/bash\nexit 0\n')
    config['phases']['validation']['script'] = 'validation/unpinned.sbatch'
    state = fixture.root / 'operation'
    t.ctl.Controller.initialize(config, state)
    scheduler = t.FakeSlurm()
    controller = t.ctl.Controller(state, scheduler)
    script.write_text('#!/bin/bash\necho altered\n')
    controller.prepare_submission('validation')
    controller.execute_submission()
    assert controller.status()['phase'] == 'waiting_validation'
    assert scheduler.calls[0][1][-1] == str(script)
    assert 'unpinned.sbatch' not in (fixture.root / 'validation/manifest.sha256').read_text()
finally:
    fixture.doCleanups()
```

Required correction: bind each phase's exact script path to a required member of its pinned helper manifest and reject missing/out-of-directory coverage before persisting a submission intent. Checking a manifest's integrity alone does not establish coverage. Add missing-member and changed-script tests at the controller submission boundary. Independently reviewing complete deployment manifests remains useful but does not replace this invariant.

## Other reviewed boundaries and limits

- Durable submission intents survive uncertain responses; restart paths reconcile rather than retry `sbatch`. Token fallback to a unique exact `SubmitLine` comment agrees with the stated accounting behavior. Parent and batch success are mandatory, with only the stated extern OOM exception.
- The controller adopts the configured proof ID and has no initial-proof submission path. The pure numerical validator, validation400 positions/history bindings, actual observed rank3 slot6718, canonical candidate digest, and validation-ready envelope agree with the current science helper.
- Promotion preserves original200, its failure marker, diagnostic256, continuous410 and at400, and recognizes the specified rename crash layouts. The no-replacement race above remains a prerequisite to approval.
- The real producer's request/receipt structure agrees with the controller for canonical checkpoint digests, four-rank fresh verification, pending-gradient inventories, epoch event carry-forward, and final publication. Source `CompileHistory.save/load` confirms the host's identity/fingerprint/position serialization assumptions. Final acceptance requires exact4099, completed-epoch count/evidence, the fixed-step markers, hash-bound nonempty DONE, and normal parent/batch exit.
- Storage uses the independent100GiB and clone-peak conditions prescribed by the binding plan. Source/helper members stream their hashes; bounded control-file reads remain separate, and model/optimizer/RNG/gradient tensors are not deserialized on the host.
- The reported38-test suite was not rerun without cause. Only the two focused probes above were executed; both reproduced their findings. No source edits, remote commands, scheduler mutations, or subagents were used by this reviewer. The only review artifact written is this file. Live proof3369694 and validation400 remain scientific prerequisites; this note makes no claim that either has passed.

After both fixes, request focused regression evidence and independent diff review. The parent owns real host3.6/shared-filesystem checks, allocation interface integration, immutable staging, and any authorized deployment.
