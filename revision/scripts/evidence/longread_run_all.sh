#!/usr/bin/env bash
# Run after EST lanes finish. Sequential to spare bandwidth. FASTA only on disk.
# Roles (protocol v1.5): test-species sets = validation only (any PacBio generation);
# A. thaliana FLIC and Zhong 2025 = VALIDATION-ONLY (never training); Cui 2020 = training-eligible.
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
S=./longread_fetch.sh
$S ont    Zmays_roottip_PRJNA822071      PRJNA822071  'SRR185719(05|06|07|11|14|15)'   # B73 only, validation
$S pacbio Zmays_wang2018_PRJEB22122      PRJEB22122   'maize'                           # HQ isoforms, validation (RS II/Sequel allowed for test species)
$S ont    Athaliana_FLIC_PRJNA1087576    PRJNA1087576 'OXFORD_NANOPORE'                 # VALIDATION-ONLY
$S ont    Athaliana_cui2020_PRJNA594286  PRJNA594286  'OXFORD_NANOPORE'                 # training-eligible
$S ont    Slycopersicum_heinz_PRJEB37834 PRJEB37834   'ERR4039883'                      # validation (conditional tomato)
$S pacbio Zmays_B73_ccs_PRJNA1470126   PRJNA1470126 'SRR388187(69|70|71)'             # B73 Sequel II CCS (validation)
$S pacbio Athaliana_zhang2023_PRJNA911826 PRJNA911826 'SRR227190(02|03|04|05|06|07)'   # Col-0 WT CCS-level, VALIDATION-ONLY
# maize Revio Kinnex FLNC (B73 x Mo17 hybrid; pooled-genotype stratum) -- not on ENA: SRA toolkit, FASTA only
mkdir -p pacbio/Zmays_kinnex_hybrid_PRJNA1290227 && cd pacbio/Zmays_kinnex_hybrid_PRJNA1290227
SRA=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sra/bin
for r in SRR34503567 SRR34503568 SRR34503569 SRR34503570; do
  [ -f $r.DONE ] && continue
  $SRA/prefetch --max-size 80G -O . $r >> log 2>&1 && $SRA/fasterq-dump --fasta --skip-technical -e 8 -O . $r >> log 2>&1 \
    && gzip -1 $r.fasta && mv $r.fasta.gz $r.fa.gz && md5sum $r.fa.gz > $r.md5 && rm -rf $r && touch $r.DONE
done
cd ../..
# maize Wang 2020 FLNC (Zenodo) -> validation (Sequel; allowed for test species)
mkdir -p pacbio/Zmays_wang2020_zenodo2611319 && cd pacbio/Zmays_wang2020_zenodo2611319
SEQKIT=$(ls /data/gpfs/assoc/pgl/bin/conda/conda_envs/*/bin/seqkit | head -1)
if [ ! -f flnc.DONE ]; then
  curl -sL "https://zenodo.org/api/records/2611319/files/F1maize.INTERMEDIATE.flnc.fastq.gz/content" | zcat | "$SEQKIT" fq2fa | gzip -1 > F1maize.flnc.fa.gz \
    && md5sum F1maize.flnc.fa.gz > F1maize.flnc.md5 && touch flnc.DONE
fi
if [ ! -f gff.DONE ]; then
  curl -sL -o F1maize.FINAL.gff "https://zenodo.org/api/records/2611319/files/F1maize.FINAL.gff/content" \
    && curl -sL -o F1maize.demux_FL_count.txt "https://zenodo.org/api/records/2611319/files/F1maize.FINAL.demux_FL_count.txt/content" && touch gff.DONE
fi
cd ../..
echo "$(date -Is) LONGREAD ALL FINISHED" >> longread.log
