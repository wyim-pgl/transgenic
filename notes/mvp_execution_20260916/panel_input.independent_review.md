# Independent MVP panel-input review (2026-09-16)

## Verdict

**PASS for the scoped panel-input producer; no blocking correctness/spec findings.**
One low-severity deployment-documentation caveat is recorded below. This is not
approval of a production run, decoder integration, or scientific results.

Reviewed detached `0f9f777` against `8803afb`, all 183 producer lines and 191 test
lines, the committed implementation report/contract, the supplied contract, and
`revision/protocols/B6_MVP_PILOT_v1.md` (particularly §§1.2–1.4, 2, 5.1–5.2).
The protocol file still labels itself a draft; this review uses the requested
approved scope, not a claim to change that document's approval status.
The supplied diff and live Git diff have identical SHA256:
`283e4825b06b2d670d463a72670c8a275fd5b69d89ef442181ee9e2f060f43d8`.
Only the producer, its tests, and two notes differ from the base.

## Finding / deployment prerequisite

**LOW — explicitly document the Git-object deployment requirement.**
`revision/scripts/mvp/mvp_prepare_panel_inputs.py:40` calls the existing
`mvp_produce_counts.frozen_reference` (`revision/scripts/mvp/mvp_produce_counts.py:20-21,54-70`).
That gate requires Git and the immutable object
`92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`
to be accessible via `git -C ROOT show`, where ROOT is the producer source-tree
root. A native worktree/checkout with that object works; an immutable source
archive without accessible Git metadata **will not operate**, even if all its
source files and input bytes are otherwise correct. Merely having the manifest
working-copy file is insufficient. A shallow checkout missing the object also
fails. Failure is closed, before publication, but surfaces as
`subprocess.CalledProcessError`, not the reference-mismatch `ValueError`.

The contract's reference to an immutable authority is accurate but does not spell
out this operational dependency. Treat it as a deployment prerequisite to document,
not a correctness defect requiring an arbitrary reference-hash override. If a
Git-free deployment is required later, it needs a separately reviewed immutable
authority transport; no such alternative is implemented or approved here.

## Requirement trace

- **All 50 windows / fixed geometry:** `preflight:30-45` uses hash-pinned
  `mvp_windows.load_panel:94-136` and requires 50 loci. `prepare:120-149` uses
  `evaluation_window:139-160`, checks chromosome lengths without clamping,
  extracts genomic-forward FASTA sequence, and requires every locus to appear.
  Sequence hash, explicit half-open window, FASTA bytes, and track identity are
  recorded together. No training Dataset or masked training sequence is used.
- **Heldout-only evidence:** `panel_tracks:48-54` calls existing `rasterise_est`
  with `rn=None`, `population='excluded_heldout'`, `keep_fraction=1`,
  `source_supplied=True`, and no RC. Existing rasterizer policy includes
  `primary_count>0`. There is no retained-flank fallback or tile_est join.
  Training filters, rasterizer, builder, and DB schema are unchanged.
- **Provenance:** `selected_provenance:57-80` uses separate overlapping-block and
  donor/acceptor endpoint queries with the same molecule policy, so intron-only
  windows are not lost. It records original matching geometry, full molecule
  metadata including cause, and all member rows. Geometry means window-matching
  geometry, not every distant placement. Selected geometry/molecule counts are
  not asserted to equal biological support (including unknown-strand junctions).
- **N, availability, seeds:** existing `mvp_tracks.py:90-108` preserves
  transcription-order endpoint identity, masks N/n last across all eight channels,
  and returns float16. Producer compares arrays exactly for augmentation seeds
  20260915/16/17 and emits one reusable on tensor. Biological nonzero counts use
  channels 0–6 only; availability is reported separately; zero-support windows
  remain publishable. The off tensor is a documented consumer operation,
  `zeros_like(on)`, not a separately emitted file or a claimed decode result.
- **Identity:** preflight invokes the frozen-reference gate and existing
  acceptance/companion and corpus/evidence/panel-byte gates
  (`mvp_cli.py:122-186`, `mvp_cli_runtime.py:89-143`). Their read-only training/valid
  metadata reconciliation is an identity check, not panel evidence selection.
  Acceptance is technical exploratory acceptance, not production authorization.
- **Publication:** `prepare:102-168` rejects existing/symlink output directories,
  opens evidence read-only, writes fresh NPY/FASTA/JSONL artifacts exclusively,
  records hashes/sizes, and checks inputs/request/code plus outputs before and
  after manifest publication. COMPLETE binds the manifest digest and is last.
  Failed partial directories are not consumable. Consumers still must verify
  COMPLETE and all identities; this is not protection against malicious transient
  swaps or post-publication mutation, nor a filesystem crash-durability claim.

## Independent verification

From this real Git worktree:

```sh
OMP_NUM_THREADS=1 \
PYTHONPATH=/tmp/b6-p2-readiness-1535112/.testdeps:/tmp/b6-trackb-testdeps:src \
/home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  revision/scripts/tests/test_mvp_panel_inputs.py -q
# 11 passed in 28.97s
```

This includes the genuine frozen-reference synthetic-input rejection: with the
Git authority present it raises the expected reference-mismatch ValueError.
The 50-locus publication test uses real frozen coordinates but synthetic sequence
and mocked preflight; it does not establish accepted-real-data end-to-end success.

Also ran focused temporary-fixture probes using the same producer: appended bytes
to `00.npy` immediately before the pre-manifest output check and, separately,
before the post-manifest check. Both raised `ValueError` containing `mutated` and
left **no COMPLETE**. The committed tests already exercise request mutation at
both phases. `git diff 8803afb 0f9f777 --check` passed.

Controller's earlier archive result (25 passes, two environment failures) and the
implementation report's 27 rasterizer/panel plus 16 candidate passes are distinct
historical evidence, not independent passes claimed here. Archive frozen-reference
failure is explained by the Git requirement above. The real-BAM smoke failure
was missing `pysam`; I did not install dependencies or rerun that smoke/broad tests.
The implementation agent's deleted dependency environment was not reconstructed.

## Remaining integration boundary

Artifact consumer/decoder contract implementation remains under construction and
outside this verdict. It must enforce identities and the same sequence/window for
on/off and all three seeds. No model-generation, real-data performance, accepted
reference/DB full publication, or structural/scientific outcome is certified here.
No source edits, SSH, jobs, subagents, merge, or push were performed. Only this
report is committed. The requested graph-first lookup was used (approximately
956,198 graft tokens saved, tool estimate). Advisor consultation was unavailable
because its configured client version did not support the selected model.
