# Evidence collection drivers

These scripts run from `evidence/` in the project checkout, **not** from here — but this is where
they live. `evidence/<name>` is a symlink to the file in this directory, so the working tree keeps
its familiar paths while the code is version-controlled.

    evidence/longread_fetch.sh -> transgenic/revision/scripts/evidence/longread_fetch.sh

## Why they moved here (author decision 2026-09-03)

They enforce frozen protocol clauses, and until now they did it from outside the repository:

| script | clause it enforces |
|---|---|
| `longread_fetch.sh` | §3 / A12 (v1.3) / A14 (v1.5) instrument restriction, the `_subreads` rejection of #60, the `library_strategy` allow-list of #63, the empty-`instrument_model` refusal of #65 |
| `est_align.sbatch` | A18.3 dataset-role gate, A21 "only the screened FASTA is aligned", A36 length floor, and the §4 frozen minimap2 command |
| `univec_screen.sh` | A21 UniVec screening (this one was already here — it is the precedent the rest followed) |

The gap was recorded in `quarantine.md` §1b: *the protocol clause is inside the repository and the
code that enforces it is outside*. On 2026-09-03 `longread_fetch.sh` was edited three times in one
day — resumable transfer, ENA checksum verification, failure accounting — with no history of any of
it. That is what settled the decision.

## Editing them while a driver is running

A collection run holds these files open. Edit the copy here, then let the symlink do the rest;
if you must replace a file in place, write a temporary file and `mv` it over the target so the
replacement is atomic. A partially-written script that a running `bash` re-reads will misbehave in
ways that are hard to attribute.

Two mistakes worth not repeating, both made on 2026-09-03:

- A file written by a tool (Python, an editor) arrives **0644**. `mv`-ing it over a driver silently
  removes the execute bit and the next invocation dies with `Permission denied` — which, from a
  background launcher, looks exactly like the download failing again. Check `ls -l` after replacing.
- `pkill -f <pattern>` matches the shell that is running the pattern too. Kill by PID.

## Scripts that are not current

`rerun_training_after_finish.sh` waits for a completion string written by the very script it is
supposed to launch, so it never fires (recorded in `resume.md`). It is kept for history, not use.
