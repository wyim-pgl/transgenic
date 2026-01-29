# Reference Datasets

Reference genomes and annotations used for TransGenic evaluation.

## Summary Table

| Species | Reference | Genome | GTF | Transcripts | Genes | Source |
|---------|-----------|--------|-----|-------------|-------|--------|
| Arabidopsis | TAIR10 | 117MB | 261MB | 54,013 | 32,833 | Ensembl Plants |
| Arabidopsis | AtRTD3 | - | 402MB | 169,499 | 40,929 | Hutton Institute |
| Rice | IRGSP-1.0 | 364MB | 171MB | 45,973 | 38,993 | Ensembl Plants |
| Maize | B73 NAM-5.0 | 2.1GB | 339MB | 77,341 | 44,303 | Ensembl Plants |

## Directory Structure

```
revision/data/
├── TAIR10/
│   ├── TAIR10.gtf              # Standard Arabidopsis annotation
│   └── TAIR10_genome.fa        # Genome sequence
├── AtRTD3/
│   ├── AtRTD3.gtf              # Comprehensive AS annotation
│   └── AtRTD3_transcripts.fa   # Transcript sequences (cDNA)
├── Rice_IRGSP/
│   ├── Rice_IRGSP.gtf
│   └── Rice_IRGSP_genome.fa
└── Maize_B73/
    ├── Maize_B73.gtf
    └── Maize_B73_genome.fa
```

## Arabidopsis References

### TAIR10 (Standard Reference)
- **Source**: [Ensembl Plants Release 57](https://plants.ensembl.org/Arabidopsis_thaliana)
- **Purpose**: Standard baseline evaluation
- **Citation**: Lamesch et al., The Arabidopsis Information Resource (TAIR), NAR 2012

### AtRTD3 (Comprehensive AS Reference)
- **Source**: [Hutton Institute](https://ics.hutton.ac.uk/atRTD/RTD3/)
- **Purpose**: Alternative splicing evaluation
- **Features**:
  - 169,499 transcripts (3x more than TAIR10)
  - 78% from Iso-seq with accurate splice junctions
  - Includes low-abundance isoforms
- **Citation**: Zhang et al., Genome Biology 2022

### Why Two References?

| Reference | Purpose | Expected Recall |
|-----------|---------|-----------------|
| TAIR10 | Standard baseline | Higher (well-annotated genes) |
| AtRTD3 | Comprehensive AS | Lower (includes rare isoforms) |

**Key insight**: Lower recall against AtRTD3 reflects missing low-abundance isoforms, not prediction errors.

## Rice Reference

### IRGSP-1.0
- **Source**: [Ensembl Plants](https://plants.ensembl.org/Oryza_sativa)
- **Organism**: *Oryza sativa* japonica (Nipponbare)
- **Citation**: Kawahara et al., Rice 2013

## Maize Reference

### B73 NAM-5.0
- **Source**: [Ensembl Plants](https://plants.ensembl.org/Zea_mays)
- **Organism**: *Zea mays* (B73 inbred line)
- **Citation**: Hufford et al., Science 2021

## Download Script

```bash
# Download all references
bash revision/scripts/01_download_references.sh
```

### Manual Download URLs

**Arabidopsis TAIR10:**
```
GTF: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gtf/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gtf.gz
Genome: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
```

**AtRTD3:**
```
GTF: https://ics.hutton.ac.uk/atRTD/RTD3/atRTD3_TS_21Feb22_transfix.gtf
Transcripts: https://ics.hutton.ac.uk/atRTD/RTD3/atRTD3_29122021.fa
```

**Rice IRGSP-1.0:**
```
GTF: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gtf/oryza_sativa/Oryza_sativa.IRGSP-1.0.57.gtf.gz
Genome: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa.gz
```

**Maize B73:**
```
GTF: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gtf/zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.57.gtf.gz
Genome: https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa.gz
```
