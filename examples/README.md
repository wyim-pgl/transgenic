# Example Files

Example prediction and validation files for TransGenic.

## Files (not tracked in git)

| File | Size | Description |
|------|------|-------------|
| `TAIR10_validation_labels.gff3` | 7.2 MB | Reference annotations |
| `TAIR10_validation_prediction.gff3` | 4.7 MB | TransGenic predictions |
| `TAIR10_gffcmp.annotated.gtf` | 4.1 MB | GFFCompare annotated output |

## Download

These files are available from the project's data repository or can be generated using the revision pipeline:

```bash
# Run AS evaluation pipeline to generate comparison files
bash revision/scripts/05_run_full_analysis.sh <your_prediction.gtf>
```

## Contact

For access to example files, contact the repository maintainer.
