# FLIC validation-only ONT alignment: separate driver

Author decision 2026-09-25: the 3 FLIC runs are aligned by a separate driver, and the production
scripts are not modified. Status: written, not submitted.

| Item | Value |
|---|---|
| Dataset | `ont/Athaliana_FLIC_PRJNA1087576` |
| Runs (array index 0,1,2) | SRR28341943, SRR28341944, SRR28341945 |
| Role | `b1_validation_only`: one row per run in `evidence/DATASET_ROLES.longread_v1.tsv` (md5 a12f9f002f97a7c44b60b052d635393c, identical to `b6_prepare_20260925/inputs/` copy) |
| Reads | `evidence/ont/Athaliana_FLIC_PRJNA1087576/<run>.fa.gz` (md5 sidecars verified OK 2026-09-25) |
| Reference | `genomes/Athaliana_167_TAIR10.fa` md5 513e5ef30845ed754b00816a99abbf8e (manifest = existing Athaliana PROVENANCE.txt) |
| Mirrors | `orientation_audit.sbatch` a8be9a948f8b70eec113a3008dff7d74, `longread_align.sbatch` 73446f9bacb7ad3b235b2dfe5b478272 (repo = pronghorn) |
| Tools | minimap2 2.28-r1209, samtools 1.3.1 (`RNASEQ_bch709/bin`) |

## What differs from production

The two scripts differ from production only in the points below. The unified diffs are at the
end of this file and in `audit.diff` and `align.diff`.

1. **Role gate.** The production gate requires exactly one `c2_training_eligible` row. Here each
   run must have exactly one DATASET_ROLES row in total, and that row must be
   `$1==ont/Athaliana_FLIC_PRJNA1087576 && $3==Athaliana && $12==b1_validation_only`. Anything
   else exits 2.
2. **Run table.** These runs are not in `longread_runs_v1.tsv`, so the table is fixed in the script.
   It has 3 runs, `--array=0-2`, and any other index exits 2.
3. **Driver record.** `driver=flic_separate (author decision 2026-09-25)` is written as a line in
   `PROVENANCE.txt` and `DONE` (audit and alignment), and appended to the central
   `PROVENANCE.tsv` note. The PROVENANCE `role` line reads `b1_validation_only`.
4. **No-overwrite sentinel.** Each output dir gets a `.driver` file. If the dir already exists
   without that sentinel, the script exits 7. The align step also requires the audit dir's
   `.driver`, so it accepts only an audit written by the FLIC audit driver.
5. **Central `PROVENANCE.tsv` scope.** The align row scope is `longread_validation_evidence`
   instead of `longread_training_evidence`, including in the duplicate-row guard. The audit row
   scope is unchanged.
6. **Job names.** `flic_orient_audit` and `flic_lr_align`.

Unchanged: CPU, memory and time requests (audit 8 CPU/16G/4h, align 32 CPU/96G/1 day), log dirs,
the 10,000-read sample and seed, all minimap2 and samtools commands, the PASS→`-uf` and
FAIL/UNRESOLVED→no `-uf` rule, the audit and align keys, DONE key order, and the PROVENANCE fields.

## Submit (in order; not yet done)

```bash
D=<pronghorn dir you copy the two .sbatch files into>   # not chosen yet
A=$(sbatch --parsable $D/flic_orientation_audit.sbatch)
sbatch --dependency=aftercorr:$A $D/flic_longread_align.sbatch
```

`aftercorr` makes align task *i* wait for audit task *i* to succeed. The scripts use absolute
paths, so they can run from any directory.

## Expected outputs

- `evidence/orientation_audit_v1/Athaliana/Athaliana_FLIC_PRJNA1087576/<run>/`: `audit.json`,
  `sample.sam.gz`, `sample.json`, `.data_key`, `PROVENANCE.txt`, `DONE`
  (key/run/status/use_uf/driver/audit_md5/completed), `.driver`
- `evidence/longread_align_v1/Athaliana/Athaliana_FLIC_PRJNA1087576/<run>/`: `<run>.bam` and `.bai`,
  `<run>.paf.gz`, `flagstat.txt`, `contig_alias.tsv`, `minimap2*.err`, `samtools.err`,
  `PROVENANCE.txt`, `DONE` (key/run/uf/audit_status/bam_md5/paf_md5/driver/completed), `.driver`
- 6 new rows in `transgenic/revision/results/evidence/PROVENANCE.tsv`

Expected cost per align task, based on comparable Athaliana jobs: about 1 h and MaxRSS about 27 GB.

## Verify after completion

```bash
E=/data/gpfs/assoc/pgl/data/Transgenic/evidence; B=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
sacct -j $A,<ALIGN_JOB> --format=JobID,State,ExitCode,Elapsed,MaxRSS
for r in SRR28341943 SRR28341944 SRR28341945; do
  o=$E/longread_align_v1/Athaliana/Athaliana_FLIC_PRJNA1087576/$r
  cat $E/orientation_audit_v1/Athaliana/Athaliana_FLIC_PRJNA1087576/$r/DONE $o/DONE
  [ "$(md5sum < $o/$r.bam | cut -d' ' -f1)" = "$(awk -F= '$1=="bam_md5"{print $2}' $o/DONE)" ] && echo bam_md5 OK
  [ "$(md5sum < $o/$r.paf.gz | cut -d' ' -f1)" = "$(awk -F= '$1=="paf_md5"{print $2}' $o/DONE)" ] && echo paf_md5 OK
  $B/samtools quickcheck -v $o/$r.bam && $B/samtools view -H $o/$r.bam | grep '^@PG.*minimap2' | grep -c -- '--secondary=no.*VN:2.28-r1209\|VN:2.28-r1209.*--secondary=no'
  grep -E '^(uf|role|command|driver)' $o/PROVENANCE.txt   # uf must agree with the audit status
done
grep -c 'flic_separate' /data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision/results/evidence/PROVENANCE.tsv   # expect 6
```

Before C0 can use these runs, each primary dir also needs `c0_metadata.tsv` and its `.sha256`,
written by `b6_c0_metadata.py`. This driver does not produce them.

## Diffs vs production

### flic_orientation_audit.sbatch
```diff
--- /tmp/b6-integration/revision/scripts/evidence/orientation_audit.sbatch	2026-09-24 09:13:43.521181941 -0700
+++ flic_orientation_audit.sbatch	2026-09-25 23:23:06.834089000 -0700
@@ -1,12 +1,20 @@
 #!/usr/bin/env bash
-#SBATCH --job-name=orient_audit
+#SBATCH --job-name=flic_orient_audit
 #SBATCH --account=cpu-s1-pgl-0
 #SBATCH --partition=cpu-s1-pgl-0
 #SBATCH --cpus-per-task=8
 #SBATCH --mem=16G
 #SBATCH --time=04:00:00
 #SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/orientation_audit_v1/logs/%x_%A_%a.out
-#SBATCH --array=0-128%12
+#SBATCH --array=0-2
+#
+# FLIC SEPARATE DRIVER (author decision 2026-09-25). A copy of revision/scripts/evidence/
+# orientation_audit.sbatch (md5 a8be9a948f8b70eec113a3008dff7d74) that differs ONLY in which runs it
+# accepts: exactly SRR28341943/44/45 of ont/Athaliana_FLIC_PRJNA1087576, each of which must carry
+# exactly one DATASET_ROLES row and that row must be b1_validation_only. Everything else - sampling,
+# command, scorer, key, DONE and PROVENANCE layout - is the production script's. The production
+# scripts are NOT modified. These runs are not in longread_runs_v1.tsv, so the run table is fixed here.
+# It never writes into an output directory it did not create (.driver sentinel, exit 7).
 #
 # §3.3 item 2 — library orientation audit, ONE TASK PER RUN. This decides `-uf` for that run and
 # nothing else; §3.3 item 4 requires every run to be scored separately, so a dataset-level verdict
@@ -29,15 +37,24 @@
 SCRIPTS=$ROOT/transgenic/revision/scripts
 MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
 ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
-RUNS=$ROOT/evidence/longread_runs_v1.tsv
+DRIVER="driver=flic_separate (author decision 2026-09-25)"
+FLIC_DS=ont/Athaliana_FLIC_PRJNA1087576
+FLIC_RUNS=(SRR28341943 SRR28341944 SRR28341945)
 OUTROOT=$ROOT/evidence/orientation_audit_v1
 THREADS=${SLURM_CPUS_PER_TASK:-8}
 
-read -r SP DS RUN FA < <(awk -F'\t' -v i="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==i' "$RUNS")
-[ -n "${RUN:-}" ] || { echo "REFUSED: no run at index $SLURM_ARRAY_TASK_ID in $RUNS" >&2; exit 2; }
+case "${SLURM_ARRAY_TASK_ID:-}" in
+  0|1|2) ;;
+  *) echo "REFUSED: array index '${SLURM_ARRAY_TASK_ID:-}' is not 0-2 (the 3 FLIC runs)" >&2; exit 2;;
+esac
+SP=Athaliana; DS=$FLIC_DS; RUN=${FLIC_RUNS[$SLURM_ARRAY_TASK_ID]}; FA=evidence/$DS/$RUN.fa.gz
 FA=$ROOT/$FA
 OUT=$OUTROOT/$SP/$(basename "$DS")/$RUN
+if [ -e "$OUT" ] && [ "$(cat "$OUT/.driver" 2>/dev/null)" != "$DRIVER" ]; then
+  echo "REFUSED: $OUT exists and was not created by this driver; will not overwrite" >&2; exit 7
+fi
 mkdir -p "$OUT" "$OUTROOT/logs"
+[ -e "$OUT/.driver" ] || printf '%s\n' "$DRIVER" > "$OUT/.driver"
 DONE=$OUT/DONE
 
 # --- Gate 1: A18.3 role, resolved by RUN IDENTITY ---
@@ -47,9 +64,12 @@
 # to `excluded`. Looking the role up by the path we are walking returns `excluded` and would drop
 # exactly the runs the protocol calls training evidence. Identity is the key; the manifest
 # guarantees one canonical eligible row per run.
-n_eli=$(awk -F'\t' -v r="$RUN" '$2==r && $12=="c2_training_eligible"' "$ROLES" | wc -l)
-if [ "$n_eli" -ne 1 ]; then
-  echo "REFUSED: $RUN has $n_eli c2_training_eligible rows in $(basename "$ROLES"), expected exactly 1" >&2
+# FLIC: the run must have exactly one row in total, and that row must be b1_validation_only for
+# this dataset and species. Any other role (or a second row of any role) fails closed.
+n_val=$(awk -F'\t' -v d="$DS" -v r="$RUN" -v s="$SP" '$1==d && $2==r && $3==s && $12=="b1_validation_only"' "$ROLES" | wc -l)
+n_all=$(awk -F'\t' -v r="$RUN" '$2==r' "$ROLES" | wc -l)
+if [ "$n_val" -ne 1 ] || [ "$n_all" -ne 1 ]; then
+  echo "REFUSED: $RUN has $n_val b1_validation_only rows and $n_all rows in total in $(basename "$ROLES"), expected exactly 1 and 1" >&2
   exit 2
 fi
 
@@ -154,7 +174,7 @@
   echo "run            $RUN"
   echo "species        $SP"
   echo "dataset        $DS"
-  echo "role           c2_training_eligible   (resolved by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
+  echo "role           b1_validation_only   (resolved by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
   echo "reads          $FA"
   echo "reads_md5      $FA_MD5"
   echo "reference      $REF"
@@ -165,6 +185,7 @@
   echo "sam_reused     $REUSED"
   echo "scorer_md5     $SCORER_MD5   (A39: the marker is bound to it; the superseded scorer gated a different statistic)"
   echo "command        $CMD"
+  echo "$DRIVER"
   echo "status         $STATUS"
   echo "use_uf         $USEUF"
   echo "sense_read     $FRAC   (A39: P(read strand == annotated strand); THIS is the gate)"
@@ -185,7 +206,7 @@
   printf "%s\tlongread_orientation_audit\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
     "$(date -Is)" "$SP" "$FA" "$FA_MD5" "$REF" "$REF_MD5" "$("$BIN"/minimap2 --version)" "$CMD" \
     "$OUT/audit.json" "$(md5sum "$OUT/audit.json" | cut -d' ' -f1)" \
-    "§3.3 item 2 as corrected by A39, run $RUN: $STATUS, sense_read_fraction $FRAC (motif_annotation_agreement $MOTIF, diagnostic), use_uf=$USEUF; scorer md5 $SCORER_MD5; GFF md5 $GFF_MD5 verified against b5_species_v1.tsv" \
+    "§3.3 item 2 as corrected by A39, run $RUN: $STATUS, sense_read_fraction $FRAC (motif_annotation_agreement $MOTIF, diagnostic), use_uf=$USEUF; scorer md5 $SCORER_MD5; GFF md5 $GFF_MD5 verified against b5_species_v1.tsv; role b1_validation_only; $DRIVER" \
     >> "$PROVTSV"
 } 9>"$PROVTSV.lock"
 
@@ -194,6 +215,7 @@
   printf 'run=%s\n' "$RUN"
   printf 'status=%s\n' "$STATUS"
   printf 'use_uf=%s\n' "$USEUF"
+  printf '%s\n' "$DRIVER"
   printf 'audit_md5=%s\n' "$(md5sum "$OUT/audit.json" | cut -d' ' -f1)"
   printf 'completed=%s\n' "$(date -Is)"
 } > "$DONE.tmp"
```

### flic_longread_align.sbatch
```diff
--- /tmp/b6-integration/revision/scripts/evidence/longread_align.sbatch	2026-09-24 09:13:43.521181941 -0700
+++ flic_longread_align.sbatch	2026-09-25 23:23:37.075212800 -0700
@@ -1,12 +1,21 @@
 #!/usr/bin/env bash
-#SBATCH --job-name=lr_align
+#SBATCH --job-name=flic_lr_align
 #SBATCH --account=cpu-s1-pgl-0
 #SBATCH --partition=cpu-s1-pgl-0
 #SBATCH --cpus-per-task=32
 #SBATCH --mem=96G
 #SBATCH --time=1-00:00:00
 #SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/longread_align_v1/logs/%x_%A_%a.out
-#SBATCH --array=0-128%3
+#SBATCH --array=0-2
+#
+# FLIC SEPARATE DRIVER (author decision 2026-09-25). A copy of revision/scripts/evidence/
+# longread_align.sbatch (md5 73446f9bacb7ad3b235b2dfe5b478272) that differs ONLY in which runs it
+# accepts: exactly SRR28341943/44/45 of ont/Athaliana_FLIC_PRJNA1087576, each with exactly one
+# DATASET_ROLES row, b1_validation_only. Commands, -uf gate, key, DONE and PROVENANCE layout are the
+# production script's; the central PROVENANCE.tsv scope is longread_validation_evidence (these runs
+# are not training evidence). The audit it reads must come from flic_orientation_audit.sbatch, and it
+# never writes into an output directory it did not create (.driver sentinel, exit 7).
+# Submit with --dependency=aftercorr:<flic_orient_audit job id> so task i waits for audit task i.
 #
 # ONT long-read -> genome spliced alignment. Protocol B1 §4 (frozen command) + §3.3 + A18.3.
 #
@@ -48,18 +57,27 @@
 PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
 MANIFEST=$ROOT/transgenic/data/manifests/b5_species_v1.tsv
 ROLES=$ROOT/evidence/DATASET_ROLES.longread_v1.tsv
-RUNS=$ROOT/evidence/longread_runs_v1.tsv
+DRIVER="driver=flic_separate (author decision 2026-09-25)"
+FLIC_DS=ont/Athaliana_FLIC_PRJNA1087576
+FLIC_RUNS=(SRR28341943 SRR28341944 SRR28341945)
 AUDITROOT=$ROOT/evidence/orientation_audit_v1
 OUTROOT=$ROOT/evidence/longread_align_v1
 THREADS=${SLURM_CPUS_PER_TASK:-32}
 
-read -r SP DS RUN FA < <(awk -F'\t' -v i="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==i' "$RUNS")
-[ -n "${RUN:-}" ] || { echo "REFUSED: no run at index $SLURM_ARRAY_TASK_ID in $RUNS" >&2; exit 2; }
+case "${SLURM_ARRAY_TASK_ID:-}" in
+  0|1|2) ;;
+  *) echo "REFUSED: array index '${SLURM_ARRAY_TASK_ID:-}' is not 0-2 (the 3 FLIC runs)" >&2; exit 2;;
+esac
+SP=Athaliana; DS=$FLIC_DS; RUN=${FLIC_RUNS[$SLURM_ARRAY_TASK_ID]}; FA=evidence/$DS/$RUN.fa.gz
 FA=$ROOT/$FA
 DSBASE=$(basename "$DS")
 OUT=$OUTROOT/$SP/$DSBASE/$RUN
 AUDIT=$AUDITROOT/$SP/$DSBASE/$RUN
+if [ -e "$OUT" ] && [ "$(cat "$OUT/.driver" 2>/dev/null)" != "$DRIVER" ]; then
+  echo "REFUSED: $OUT exists and was not created by this driver; will not overwrite" >&2; exit 7
+fi
 mkdir -p "$OUT" "$OUTROOT/logs"
+[ -e "$OUT/.driver" ] || printf '%s\n' "$DRIVER" > "$OUT/.driver"
 BAM=$OUT/$RUN.bam
 PAF=$OUT/$RUN.paf.gz
 PROV=$OUT/PROVENANCE.txt
@@ -67,8 +85,11 @@
 
 # --- Gate 1: A18.3 role, by RUN IDENTITY (three Cui 2020 runs exist under two paths; the copy
 #     under training/ont/ is the `excluded` duplicate, so a path lookup drops training evidence) ---
-n_eli=$(awk -F'\t' -v r="$RUN" '$2==r && $12=="c2_training_eligible"' "$ROLES" | wc -l)
-[ "$n_eli" -eq 1 ] || { echo "REFUSED: $RUN has $n_eli c2_training_eligible rows, expected exactly 1" >&2; exit 2; }
+# FLIC: exactly one row for the run in total, and it is b1_validation_only for this dataset/species.
+n_val=$(awk -F'\t' -v d="$DS" -v r="$RUN" -v s="$SP" '$1==d && $2==r && $3==s && $12=="b1_validation_only"' "$ROLES" | wc -l)
+n_all=$(awk -F'\t' -v r="$RUN" '$2==r' "$ROLES" | wc -l)
+{ [ "$n_val" -eq 1 ] && [ "$n_all" -eq 1 ]; } \
+  || { echo "REFUSED: $RUN has $n_val b1_validation_only rows and $n_all rows in total, expected exactly 1 and 1" >&2; exit 2; }
 
 REF=$(awk -F'\t' -v s="$SP" '$1==s{print $4}' "$MANIFEST")
 GFF=$(awk -F'\t' -v s="$SP" '$1==s{print $6}' "$MANIFEST")
@@ -89,6 +110,8 @@
 # --- Gate 3: §3.3 item 2 verdict. No audit, no alignment. ---
 [ -s "$AUDIT/DONE" ] && [ -s "$AUDIT/audit.json" ] \
   || { echo "REFUSED: $RUN has no §3.3 orientation audit at $AUDIT" >&2; exit 5; }
+[ "$(cat "$AUDIT/.driver" 2>/dev/null)" = "$DRIVER" ] \
+  || { echo "REFUSED: $RUN audit at $AUDIT was not written by flic_orientation_audit.sbatch" >&2; exit 5; }
 # The audit's key names the data it measured AND the scorer that measured it (A39). This driver
 # checks the data part - it must be the objects we are about to align - and records the scorer,
 # but does not pin a scorer md5: that would couple the aligner to a version number instead of to a
@@ -200,7 +223,7 @@
   echo "run            $RUN"
   echo "species        $SP"
   echo "dataset        $DS"
-  echo "role           c2_training_eligible   (by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
+  echo "role           b1_validation_only   (by run identity in $(basename "$ROLES"), md5 $(md5sum "$ROLES" | cut -d' ' -f1))"
   echo "reads          $FA"
   echo "reads_md5      $FA_MD5"
   echo "reference      $REF"
@@ -216,6 +239,7 @@
   echo "minimap2       $("$BIN"/minimap2 --version)"
   echo "samtools       $("$BIN"/samtools --version | head -1)"
   echo "command        $CMD"
+  echo "$DRIVER"
   echo "started        $START"
   echo "finished       $(date -Is)"
   echo "wall_seconds   $SECONDS"
@@ -235,14 +259,14 @@
   [ -s "$PROVTSV" ] || printf "date\tscope\tspecies\tinput\tinput_md5\treference\treference_md5\ttool\tcommand\toutput\toutput_md5\tnote\n" > "$PROVTSV"
   # A task that was killed after appending and before writing DONE will run again. Without this
   # the central table grows a second row for the same run and key, and nothing says which is live.
-  if grep -qF "	longread_training_evidence	" "$PROVTSV" 2>/dev/null \
+  if grep -qF "	longread_validation_evidence	" "$PROVTSV" 2>/dev/null \
      && grep -qF "run $RUN; " "$PROVTSV" 2>/dev/null; then
     grep -vF "run $RUN; " "$PROVTSV" > "$PROVTSV.tmp" && mv "$PROVTSV.tmp" "$PROVTSV"
   fi
-  printf "%s\tlongread_training_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
+  printf "%s\tlongread_validation_evidence\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
     "$(date -Is)" "$SP" "$FA" "$FA_MD5" "$REF" "$REF_MD5" "$("$BIN"/minimap2 --version)" "$CMD" \
     "$BAM;$PAF" "$(md5sum "$BAM" | cut -d' ' -f1);$(md5sum "$PAF" | cut -d' ' -f1)" \
-    "run $RUN; $UFNOTE agreement $FRAC -> ${UF:-no -uf}; audit md5 $AUDIT_MD5; UMI unknown, counts are reads" \
+    "run $RUN; $UFNOTE agreement $FRAC -> ${UF:-no -uf}; audit md5 $AUDIT_MD5; UMI unknown, counts are reads; role b1_validation_only; $DRIVER" \
     >> "$PROVTSV"
 } 9>"$PROVTSV.lock" || { echo "REFUSED: could not append central provenance for $RUN" >&2; exit 4; }
 grep -qF "run $RUN; " "$PROVTSV" || { echo "REFUSED: central provenance row for $RUN is not present after the append" >&2; exit 4; }
@@ -254,6 +278,7 @@
   printf 'audit_status=%s\n' "$STATUS"
   printf 'bam_md5=%s\n' "$(md5sum "$BAM" | cut -d' ' -f1)"
   printf 'paf_md5=%s\n' "$(md5sum "$PAF" | cut -d' ' -f1)"
+  printf '%s\n' "$DRIVER"
   printf 'completed=%s\n' "$(date -Is)"
 } > "$DONE.tmp" || { echo "REFUSED: could not write the completion marker for $RUN" >&2; exit 4; }
 mv "$DONE.tmp" "$DONE" || { echo "REFUSED: could not install the completion marker for $RUN" >&2; exit 4; }
```
