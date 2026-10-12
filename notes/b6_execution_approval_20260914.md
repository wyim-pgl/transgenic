# B6 execution approvals — 2026-09-14 16:41 PDT

Author instructions in this session:

1. “EST blastx 실행” — authorizes execution of the previously selected route B EST DIAMOND blastx workflow, after input/DB/launcher/scheduler preflight and review of any new execution code.
2. “mvp 승인할께 진행해줘” — authorizes the exploratory MVP implementation and execution described in `revision/protocols/B6_MVP_PILOT_v1.md`.

MVP specification at approval: v1.2 draft, SHA256 `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`. This includes the 16:35 PDT corrections: genomic transcript strand composes SAM FLAG and ts per A39; equality with the starting seed123 model applies only at step zero. No claim that these two corrections have received an additional independent review.

## Scope

- MVP: Athaliana, EST-only 8 channels, exploratory labels; data builder/rasterizer and tests before model changes, then memory/generation/DDP gates before fine-tunes. Base estimate approximately 18 GPU-h; approximately 5.5 additional GPU-h for the conditional withholding check.
- DIAMOND route B: separate production-evidence preparation path; phase channels are not added to the 8-channel MVP by this approval.
- Do not interpret either instruction as blanket A46 approval, pre-registration freeze, full B6 training authorization, or permission to resume deferred seeds 456/789.
- Preserve the running/queued seed123 checkout and jobs. Use a separate checkout for MVP training.

## Progress

- Read-only DIAMOND launcher/input/taxonomy/provenance investigation started; no blastx job submitted yet.
- Read-only MVP days 1–2 implementation planning started; no MVP model or training job started yet.
- Existing work remains uncommitted; no freeze or commit performed by this record.
