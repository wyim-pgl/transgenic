#!/bin/bash
# Download reference annotations for AS evaluation
# - AtRTD3: Comprehensive AS annotation (Zhang et al., Genome Biology, 2022)
# - TAIR10: Standard Arabidopsis reference annotation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${REVISION_DIR}/data"

echo "=============================================="
echo "TransGenic Revision: Download Reference Data"
echo "=============================================="
echo ""

#######################################
# 1. AtRTD3 (Alternative Splicing Reference)
#######################################
ATRTD3_DIR="${DATA_DIR}/AtRTD3"
mkdir -p "${ATRTD3_DIR}"

echo "=== [1/2] Downloading AtRTD3 ==="
echo "Source: Hutton Institute"
echo "Reference: Zhang et al., Genome Biology, 2022"
echo ""

cd "${ATRTD3_DIR}"

# AtRTD3 GTF
if [[ ! -f "AtRTD3.gtf" ]]; then
    echo "Downloading AtRTD3 GTF..."
    wget -q --show-progress -c "https://ics.hutton.ac.uk/atRTD/RTD3/AtRTD3_29122021.gtf" \
        -O "AtRTD3.gtf" 2>&1 || {
        echo "Trying curl..."
        curl -L -# -o "AtRTD3.gtf" "https://ics.hutton.ac.uk/atRTD/RTD3/AtRTD3_29122021.gtf"
    }
    echo "  -> AtRTD3.gtf downloaded"
else
    echo "  AtRTD3.gtf already exists, skipping"
fi

# AtRTD3 transcript FASTA (optional)
if [[ ! -f "AtRTD3.fa" ]]; then
    echo "Downloading AtRTD3 transcript FASTA..."
    wget -q --show-progress -c "https://ics.hutton.ac.uk/atRTD/RTD3/AtRTD3_29122021.fa" \
        -O "AtRTD3.fa" 2>&1 || {
        curl -L -# -o "AtRTD3.fa" "https://ics.hutton.ac.uk/atRTD/RTD3/AtRTD3_29122021.fa" || {
            echo "  WARNING: Failed to download AtRTD3 FASTA (optional)"
        }
    }
else
    echo "  AtRTD3.fa already exists, skipping"
fi

echo ""

#######################################
# 2. TAIR10 (Standard Reference)
#######################################
TAIR10_DIR="${DATA_DIR}/TAIR10"
mkdir -p "${TAIR10_DIR}"

echo "=== [2/2] Downloading TAIR10 ==="
echo "Source: Arabidopsis.org / Ensembl Plants"
echo ""

cd "${TAIR10_DIR}"

# TAIR10 genome FASTA
if [[ ! -f "TAIR10_genome.fa" ]]; then
    echo "Downloading TAIR10 genome FASTA..."
    # Try Ensembl Plants first (more reliable)
    wget -q --show-progress -c \
        "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz" \
        -O "TAIR10_genome.fa.gz" 2>&1 || {
        echo "Ensembl failed, trying TAIR..."
        wget -q --show-progress -c \
            "https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas" \
            -O "TAIR10_genome.fa" 2>&1 || {
            curl -L -# -o "TAIR10_genome.fa" \
                "https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas"
        }
    }
    # Decompress if gzipped
    if [[ -f "TAIR10_genome.fa.gz" ]]; then
        echo "  Decompressing..."
        gunzip -f "TAIR10_genome.fa.gz"
    fi
    echo "  -> TAIR10_genome.fa downloaded"
else
    echo "  TAIR10_genome.fa already exists, skipping"
fi

# TAIR10 GTF (from Ensembl Plants - better format)
if [[ ! -f "TAIR10.gtf" ]]; then
    echo "Downloading TAIR10 GTF..."
    wget -q --show-progress -c \
        "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gtf/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gtf.gz" \
        -O "TAIR10.gtf.gz" 2>&1 || {
        echo "Ensembl GTF failed, trying GFF3 from TAIR..."
        wget -q --show-progress -c \
            "https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff" \
            -O "TAIR10.gff3" 2>&1
    }
    # Decompress if gzipped
    if [[ -f "TAIR10.gtf.gz" ]]; then
        echo "  Decompressing..."
        gunzip -f "TAIR10.gtf.gz"
        echo "  -> TAIR10.gtf downloaded"
    elif [[ -f "TAIR10.gff3" ]]; then
        echo "  Converting GFF3 to GTF..."
        # Will need gffread for conversion
        if command -v gffread &> /dev/null; then
            gffread "TAIR10.gff3" -T -o "TAIR10.gtf"
            echo "  -> TAIR10.gtf converted from GFF3"
        else
            echo "  WARNING: gffread not found, keeping GFF3 format"
            echo "  Run 'gffread TAIR10.gff3 -T -o TAIR10.gtf' after activating environment"
        fi
    fi
else
    echo "  TAIR10.gtf already exists, skipping"
fi

echo ""

#######################################
# Summary
#######################################
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo ""
echo "AtRTD3 directory: ${ATRTD3_DIR}"
ls -lh "${ATRTD3_DIR}" 2>/dev/null | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "TAIR10 directory: ${TAIR10_DIR}"
ls -lh "${TAIR10_DIR}" 2>/dev/null | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Count transcripts
echo "Transcript counts:"
if [[ -f "${ATRTD3_DIR}/AtRTD3.gtf" ]]; then
    count=$(grep -c $'\ttranscript\t' "${ATRTD3_DIR}/AtRTD3.gtf" 2>/dev/null || echo "?")
    echo "  AtRTD3: ${count} transcripts"
fi
if [[ -f "${TAIR10_DIR}/TAIR10.gtf" ]]; then
    count=$(grep -c $'\ttranscript\t' "${TAIR10_DIR}/TAIR10.gtf" 2>/dev/null || echo "?")
    echo "  TAIR10: ${count} transcripts"
fi

echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo "1. micromamba activate transgenic-revision"
echo "2. Run analysis against BOTH references:"
echo ""
echo "   # vs AtRTD3 (AS evaluation)"
echo "   python scripts/02_gffcompare_analysis.py \\"
echo "       -r data/AtRTD3/AtRTD3.gtf \\"
echo "       -p <transgenic.gtf> -o results/vs_AtRTD3"
echo ""
echo "   # vs TAIR10 (standard reference)"
echo "   python scripts/02_gffcompare_analysis.py \\"
echo "       -r data/TAIR10/TAIR10.gtf \\"
echo "       -p <transgenic.gtf> -o results/vs_TAIR10"
echo ""
