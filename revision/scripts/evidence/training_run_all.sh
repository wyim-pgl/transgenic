#!/usr/bin/env bash
set -Eeuo pipefail
# Training-evidence long reads (protocol v1.5): ONT (any) + PacBio Sequel II/IIe/Revio CCS/FLNC-level only.
# Nine training species. FASTA only. Validation-only A. thaliana sets (PRJNA1087576, PRJEB77203, PRJNA911826) are NOT here.
cd "${LONGREAD_ROOT:-/data/gpfs/assoc/pgl/data/Transgenic/evidence}"
S=./longread_fetch.sh
PB='Sequel II|Sequel IIe|Revio'
# --- A. thaliana (training-eligible)
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Athaliana    cui2020_PRJNA594286     PRJNA594286
$S training/ont/Athaliana    col0_DRP009401          PRJDB14952  'DRR42473(1|2|3)'                 # Col-0 subset
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Athaliana inflorescence_PRJNA596358 PRJNA596358
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Athaliana seedlingPi_PRJNA649694  PRJNA649694
# --- V. vinifera
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Vvinifera    pinotnoir_berry_PRJNA776245 PRJNA776245
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Vvinifera    callus_PRJNA732451      PRJNA732451
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Vvinifera zhuosexiang_9tissue_PRJNA1185815 PRJNA1185815
# --- G. max (no usable Sequel II+ CCS)
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Gmax         wm82_graft_PRJNA648759  PRJNA648759
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Gmax         wm82_seed_PRJNA416810   PRJNA416810
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Gmax         scn_roots_PRJNA803218   PRJNA803218                  # genotype 09-138
# --- P. trichocarpa
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Ptrichocarpa sdx_dRNA_PRJNA517295    PRJNA517295
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Ptrichocarpa sdx_drought_PRJNA672182 PRJNA672182
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Ptrichocarpa jgi_F1hybrid_PRJNA709498 PRJNA709498    # hybrid tag
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Ptrichocarpa jgi_F1hybrid_PRJNA709499 PRJNA709499    # hybrid tag
# --- S. bicolor (no ONT); Wray Sequel II/IIe HiFi-level runs (two subreads runs skipped by MAX_READS), Baijiu HiFi
for prj in PRJNA1275171 PRJNA1275229 PRJNA1275264 PRJNA1275290 PRJNA1275324 PRJNA1275413 PRJNA1275435; do
  PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Sbicolor wray_$prj $prj
done
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Sbicolor baijiu_PRJNA1034755 PRJNA1034755
# --- B. distachyon: none available
# --- S. italica
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Sitalica     ci846_salt_PRJNA1097621 PRJNA1097621
# --- O. sativa
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_dRNA_6tissue_PRJNA752930 PRJNA752930
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_pool_PRJNA953663    PRJNA953663
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      nip_sheath_PRJNA1044249 PRJNA1044249
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Osativa      indica_flagleaf_PRJNA1291274 PRJNA1291274
PLATFORM_RE='PACBIO' MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Osativa indica_hifi_PRJNA1291274 PRJNA1291274
# --- P. patens
PLATFORM_RE='OXFORD_NANOPORE' $S training/ont/Ppatens      dRNA_gametophore_PRJNA681088 PRJNA681088
echo "$(date -Is) TRAINING LONGREAD ALL FINISHED" >> longread.log
