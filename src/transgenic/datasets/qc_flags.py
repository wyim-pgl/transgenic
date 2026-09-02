"""Annotation-quality flag semantics shared by 62_geenuff_qc.py and build_b5.py (protocol A22)."""
HARD_FLAG_PATTERNS = ("missing_start", "missing_stop", "wrong_starting_phase", "mismatched_ending_phase", "mismatched_phase",
                      "overlapping_exon", "too_short_intron", "empty_transcript", "empty_super_locus", "wrong_phase")


def is_hard(flag: str) -> bool:
    f = flag.lower()
    return any(p in f for p in HARD_FLAG_PATTERNS)
