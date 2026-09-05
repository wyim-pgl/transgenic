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

## Long-read fetch V2 (A18.8)

`longread_fetch.sh` now resolves its own symlink and executes `longread_fetch_v2.sh`.
All five fetch callers migrate together. The live `evidence/` symlinks need no edits.

```bash
PLATFORM_RE=OXFORD_NANOPORE ./longread_fetch.sh ont label PRJNA594286
SUBMITTED_RE=maize ./longread_fetch.sh pacbio label PRJEB22122
./longread_fetch.sh ont label PRJDB38182 '^DRR807190$'
```

The optional fourth argument is **RUN_RE only** (also accepted as an environment variable).
`PLATFORM_RE`, `SUBMITTED_RE`, `MODEL_RE` and `STRAT_ALLOW` match their parsed columns.
Unset optional filters impose no constraint; explicitly empty regexes still require a nonempty
field. The transcript-strategy default remains `^(RNA-Seq|FL-cDNA|ncRNA-Seq|OTHER)$`.
Known nonmatches are excluded; otherwise missing filtered metadata is `UNRESOLVED`.
`MAX_READS` and the subreads exclusion remain in force. Invalid configuration, recognizable legacy
fourth arguments, and zero rows selected by column filters exit **6**, with a stderr diagnostic.
Selected runs excluded by the read-count/subreads policy are logged as skipped. Download,
conversion or count failures exit **5**; unavailable metadata/checksums exit **4** (5 takes
precedence if both occur). Sequential callers stop on errors, and parallel callers wait for and
propagate child errors before writing FINISHED.

Submitted FASTQ preference is retained. All URLs from the chosen source family are downloaded;
`submitted_md5` is paired with `submitted_ftp`, and `fastq_md5` with `fastq_ftp`, by position.
Missing/malformed checksums, unequal lists, and mixed submitted formats are unresolved, with no
fallback that silently changes the data product. Saved older reports without `submitted_md5`
therefore cannot validate submitted files; production requests now include that ENA column.
Every source must pass gzip integrity and ENA MD5 checks. The combined FASTA record count must
exactly equal ENA `read_count`; there is no inferred multiplier or tolerance.

Raw names include source family, index, and a URL/checksum identity hash, so retries resume only
bytes from the same source. HTTP/1.1, byte-range resume, speed limits and retry backoff remain.
Checksum-corrupt bytes are retained as `.bad.*`, and clean downloads retry. Validated raw files
survive conversion/count failures and are reused without downloading on restart. Legacy
`<run>.raw.gz` files have no reliable source identity; they are left untouched, not guessed at.
FASTA, checksum sidecars and completion records use temporary files and atomic rename; sources
are removed only after validation and publication. A new `.DONE` records script/report SHA256,
file count and read count. Successful retries clear `.FAILED` and `.UNRESOLVED`.

Existing `.DONE` outputs are not re-downloaded or re-audited by this migration. The separate
A18.8 historical-data audit and the remaining ingestion/provenance work are not claimed here.
Filtered metadata is still checked before accepting an existing completion marker.

For offline execution, set `LONGREAD_ROOT` to a scratch directory and `LONGREAD_FILEREPORT` to
an absolute saved-report path; no ENA metadata request is then made. `SEQKIT` can select the
converter, and `LONGREAD_RETRY_DELAY=0` removes outer retry sleeps in tests. Tests replace curl
with an offline resumable transport and use the installed seqkit for actual conversion.

```bash
/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python -m pytest -q tests/test_longread_fetch.py
```

The fixture README records five byte-for-byte saved reports and their SHA256 checksums. Tests
never read the live evidence tree. Derived rows and tiny two-file FASTQs test missing fields,
reordered columns, positional checksums, ID/count preservation, resume, count mismatch,
conversion failure, marker recovery, atomic publication, and migration guards.
