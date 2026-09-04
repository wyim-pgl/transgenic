#!/usr/bin/env bash
# Training-evidence long reads (protocol v1.5): ONT (any) + PacBio Sequel II/IIe/Revio CCS/FLNC-level only.
# Nine training species. FASTA only. Validation-only A. thaliana sets (PRJNA1087576, PRJEB77203, PRJNA911826) are NOT here.
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
S=./longread_fetch.sh
PB='Sequel II|Sequel IIe|Revio'
# --- A. thaliana (training-eligible)
$S training/ont/Athaliana    cui2020_PRJNA594286     PRJNA594286 'OXFORD_NANOPORE'
$S training/ont/Athaliana    col0_DRP009401          PRJDB14952  'DRR42473(1|2|3)'                 # Col-0 subset
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Athaliana inflorescence_PRJNA596358 PRJNA596358 'PACBIO'
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Athaliana seedlingPi_PRJNA649694  PRJNA649694 'PACBIO'
# --- V. vinifera
$S training/ont/Vvinifera    pinotnoir_berry_PRJNA776245 PRJNA776245 'OXFORD_NANOPORE'
$S training/ont/Vvinifera    callus_PRJNA732451      PRJNA732451 'OXFORD_NANOPORE'
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Vvinifera zhuosexiang_9tissue_PRJNA1185815 PRJNA1185815 'PACBIO'
# --- G. max (no usable Sequel II+ CCS)
$S training/ont/Gmax         wm82_graft_PRJNA648759  PRJNA648759 'OXFORD_NANOPORE'
$S training/ont/Gmax         wm82_seed_PRJNA416810   PRJNA416810 'OXFORD_NANOPORE'
$S training/ont/Gmax         scn_roots_PRJNA803218   PRJNA803218 'OXFORD_NANOPORE'                 # genotype 09-138
# --- P. trichocarpa
$S training/ont/Ptrichocarpa sdx_dRNA_PRJNA517295    PRJNA517295 'OXFORD_NANOPORE'
$S training/ont/Ptrichocarpa sdx_drought_PRJNA672182 PRJNA672182 'OXFORD_NANOPORE'
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Ptrichocarpa jgi_F1hybrid_PRJNA709498 PRJNA709498 'PACBIO'   # hybrid tag
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Ptrichocarpa jgi_F1hybrid_PRJNA709499 PRJNA709499 'PACBIO'   # hybrid tag
# --- S. bicolor (no ONT); Wray Sequel II/IIe HiFi-level runs (two subreads runs skipped by MAX_READS), Baijiu HiFi
for prj in PRJNA1275171 PRJNA1275229 PRJNA1275264 PRJNA1275290 PRJNA1275324 PRJNA1275413 PRJNA1275435; do
  MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Sbicolor wray_$prj $prj 'PACBIO'
done
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Sbicolor baijiu_PRJNA1034755 PRJNA1034755 'PACBIO'
# --- B. distachyon: none available
# --- S. italica
$S training/ont/Sitalica     ci846_salt_PRJNA1097621 PRJNA1097621 'OXFORD_NANOPORE'
# --- O. sativa
$S training/ont/Osativa      nip_dRNA_6tissue_PRJNA752930 PRJNA752930 'OXFORD_NANOPORE'
$S training/ont/Osativa      nip_pool_PRJNA953663    PRJNA953663 'OXFORD_NANOPORE'
$S training/ont/Osativa      nip_sheath_PRJNA1044249 PRJNA1044249 'OXFORD_NANOPORE'
$S training/ont/Osativa      indica_flagleaf_PRJNA1291274 PRJNA1291274 'OXFORD_NANOPORE'
MODEL_RE="$PB" MAX_READS=5000000 $S training/pacbio/Osativa indica_hifi_PRJNA1291274 PRJNA1291274 'PACBIO'
# --- P. patens
$S training/ont/Ppatens      dRNA_gametophore_PRJNA681088 PRJNA681088 'OXFORD_NANOPORE'
echo "$(date -Is) TRAINING LONGREAD ALL FINISHED" >> longread.log
