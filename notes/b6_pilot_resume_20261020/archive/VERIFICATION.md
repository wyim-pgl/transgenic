# Archive operation verification — 2026-10-20

- Existing core/CLI/storage helper hashes exactly match reviewed3b4eb0d.
- `bash -n archive.sbatch`: PASS.
- Python3.6 grammar parsing of guard and regression: PASS.
- Initial real tiny CLI transaction passed. Independent review then found oneP2: repeated generic CLI planning could select a newly appearing older checkpoint or unrelated pending transaction after the initial scope check.
- Fixed operation invokes the unchanged core prepare/transfer/cleanup APIs for literal epochs9,10,11 only, checks each prepared manifest against pinned epoch/name/bytes, and records exclusive manifests/verified receipts/cleanup results.
- Focused regression: `python3 notes/b6_pilot_resume_20261020/archive/test_archive_guard.py -v` →1testPASS,26.161seconds. It introduces epoch08 after preflight and proves it remains untouched, archives exactly09–11 with independent destination-byte verification, preserves12–14/latest/best, and rejects changed live position, out-of-scope epochs and a mismatched cleanup receipt. An initial execution exposed a loop-variable shadowing bug; renaming it to protected_epoch repaired that actual failure before the passing run.
- Actual DeltaAI HDD parent `/work/hdd/bilv/wcyim/transgenic` reports poolddn_hdd and project22169 with inheritanceP. New archive destination is not yet created. Quota remaining170.62GiB exceeds payload39.46GiB plus50GiB reserve.
- NVMe remaining94.37GiB; current allocation account balance1641Hours/deposited2501Hours. No training has been submitted during this preparation.
- Initial CPU-only proposal was rejected by actual `sbatch --test-only`: `Jobs requesting no GPU resources cannot be submitted on DeltaAI`. No job was submitted. QOSMinTRES being empty did not bypass this site policy.
- Revised allocation requests the site's minimum1GPU,1CPU,2GiB RAM,2hours onghx4-interactive. Work remains stdlib fileI/O only; noTorch/CUDA execution. At most2GPU-hours of allocation (interactive charging factor may increase charged account hours);1641account hours remain. NativeHDD/project22169P and host3.6 tiny transaction test passed during v1staging. Revised resources require a new immutablev2operation and fresh scheduler preflight before submission.
