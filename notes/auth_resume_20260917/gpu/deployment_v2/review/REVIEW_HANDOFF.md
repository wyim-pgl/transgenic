# Independent review required

This directory contains implementer-generated local verification, not an independent approval. No independent new-code review has yet been performed.

Review exact release_v3 and transport against IMPLEMENTATION.md and identity-verification.json. Concentrate on admission.py authorization schema, query provenance/timeouts/freshness, scheduler identity and census completeness, accounting allocation/step treatment, B5 history classification and maximum reservations, account unit conversion, admission races, and deploy.py's no-stage-effects ordering. Verify shell claim persistence and externally protected wrapper assumptions.

The old release_v2 independent review remains unchanged at ../../deployment_v1/review/INDEPENDENT_REVIEW.md. Its queued-age finding motivated this correction; its acceptance of other surfaces is not acceptance of these new bytes.

Critical new external state: old3163986 is FAILED with51allocated seconds, not zero-allocation CANCELLED. See old3163986-history.json. The new gate intentionally rejects it. Its accounting also reports288allocated CPUs versus64requested under exclusive placement. Do not reinterpret either condition to obtain a pass. Controller reconciliation and geometry clarification are prerequisites to execution authority.

No actual approved authorization, runtime admission, readiness, submission, cancellation, transfer, or training was performed by this implementation task. The source archive and operator pins remain unchanged. Local tests are synthetic and cannot substitute for an independent review or live compute-start queries.
