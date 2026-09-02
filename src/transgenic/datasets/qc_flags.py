"""Annotation-quality flag semantics shared by 62_geenuff_qc.py, 63_swissprot_sensitivity.py and build_b5.py (protocols A22, A30).

Hard flags mask a gene model from the training loss (train_weight 0) or drop the flagged transcript; every other flag is
soft and recorded only. `swissprot_caution_*` (A30: a Swiss-Prot SEQUENCE CAUTION of a structural type whose curated
sequence is absent from the current reference proteome) is hard; `swissprot_note_*` is soft."""
HARD_FLAG_PATTERNS = ("missing_start", "missing_stop", "wrong_starting_phase", "mismatched_ending_phase", "mismatched_phase",
                      "overlapping_exon", "too_short_intron", "empty_transcript", "empty_super_locus", "wrong_phase",
                      "swissprot_caution")


def is_hard(flag: str) -> bool:
    f = flag.lower()
    return any(p in f for p in HARD_FLAG_PATTERNS)
