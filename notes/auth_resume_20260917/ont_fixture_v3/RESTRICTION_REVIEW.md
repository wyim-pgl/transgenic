# Narrow restriction review

Disposition: accepted as a restriction of the prior conditional site-v1-only review, not new fixture authorization or F4/full ONT approval. All prior controller conditions remain applicable.

Verified both 17-entry manifests against actual bytes. Exactly one payload and manifest hash changed: runtime_closure.py. The complete former v2 branch is replaced by an unconditional ValueError when the v1 branch is not selected; all preceding bytes, including v1 behavior, are identical. Candidate MANIFEST.sha256:

`2cd93e7b93182a51dc9198d4f697af7b9f36b826e0286d1538f26c96af09b904`

The test_amendments.py diff adds only test_unreviewed_v2_hierarchy_rejected_before_workload. Independently ran with `/usr/bin/python3 -I -B` (3.10.12): the new regression against the original v2 collector, substituted only in memory, fails with `ValueError not raised`; against v3, that regression and test_v1_cgroup_requires_job_hierarchy_memory_and_swap both pass. Retained controller logs show expected RED then 18 passing tests; the full suite was not independently rerun in this narrow review.

All 36 prior retained v2 file identities match review_v2/identities.json. The entire v2 tree is byte-identical before/after these checks; v3 payload hashes remain intact. No remote changes, jobs, installs or workload execution occurred. Required local graft orientation auto-refreshed graph metadata; no candidate source was edited.

Restriction review complete. Only one bounded, separately authorized documented site-v1 fixture remains eligible under the prior conditions.
