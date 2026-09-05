#!/usr/bin/env bash
set -Eeuo pipefail
# The ONT datasets training_run_all.sh never reached: it stopped after S. italica on 2026-09-02 23:29
# (no "TRAINING LONGREAD ALL FINISHED" line in longread.log) and never started O. sativa or P. patens.
# ONT only: every downloaded training PacBio run so far is a _subreads file, which protocol v1.3/v1.5
# excludes (Sequel II+ CCS/FLNC level only), so the PacBio lines are deliberately left out pending a decision.
cd "${LONGREAD_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}"
S=./longread_fetch.sh
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_dRNA_6tissue_PRJNA752930 PRJNA752930
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_pool_PRJNA953663    PRJNA953663
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_sheath_PRJNA1044249 PRJNA1044249
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      indica_flagleaf_PRJNA1291274 PRJNA1291274
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Ppatens      dRNA_gametophore_PRJNA681088 PRJNA681088
echo "$(date -Is) TRAINING ONT (Osativa, Ppatens) FINISHED" >> longread.log
