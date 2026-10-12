# Independent D2.1 permission-fix review

**Verdict: PASS. D2.1 is resolved within the requested creation-time owner/mode scope.**

Reviewed `notes/seed123_execution_20260917/est_deployment/prepare_controller.py` and the new permission regression. `require_private` checks effective UID ownership and exact `stat.S_IMODE`, immediately after namespace creation, after each log creation, and before writing receipt content. Failure propagates before the CLI prints its receipt or submission argv; failed reservations are retained.

The regression covers eight combinations: incorrect observed owner or mode for namespace, stdout, stderr, and preparation receipt. Each requires rejection and retention, with no successful receipt content. This review does not claim hostile same-user race protection.

## Fresh independent evidence

Harness: `permission_fix_checks.py`. Evidence: `permission_fix_evidence_2_atyyxf/`.

- With `TMPDIR=/tmp`, targeted permission regression and positive preparation-contract test both passed: 2 tests, 0.111 seconds (`tests.stderr`).
- Additional unmocked CLI preparation on `/tmp` exited 0. Namespace mode was 0700; both empty logs and receipt were 0600; all owners matched UID 1000. Receipt status was PREPARED_NOT_SUBMITTED; no worker output existed.
- Fresh unmocked preparation in the `/mnt/f` evidence directory observed namespace mode 0777 and exited 1 with the private-owner/mode error. Stdout was empty: no usable argv was printed. No preparation receipt or logs were created. The failed reservation remains empty. Reuse exited 1 with FileExistsError and preserved its inode.
- Exact commands, outcomes and before/after matching hashes: `results.json`; CLI output: `positive.*`, `unsupported.*`, `refused-reuse.*`.

SHA256:

```text
prepare_controller.py 9f30e4c828abe84ce2b2df5e1ad8ea648dc32414207a9f13a37204ff3acb9b8f
test_deployment.py 96dd1ebeb9326fa33dbe840f8ca0a3f8145df1221168db796689417cf84e59c3
```

Only tiny synthetic fixtures were used. No source edits, jobs, real inputs, remote actions, commits, launcher/binder re-review, or full-suite rerun. Graft reported approximately 39,511 tokens saved.
