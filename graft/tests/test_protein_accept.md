# tests/test_protein_accept.py

- _paf · function · L28-L31 — def _paf(prot: str, chrom: str) -> str: # qname qlen qstart qend strand tname tlen tstart tend match blocklen mapq tags...
- _alignment · function · L34-L38 — def _alignment(prot: str, chrom: str, mid: str, score: int = 500) -> str
- run · function · L42-L53 — def run(tmp_path_factory)
- _rows · function · L56-L57 — def _rows(path: Path)
- test_every_alignment_is_judged_and_accepted · function · L60-L63 — def test_every_alignment_is_judged_and_accepted(run)
- test_ambiguous_protein_is_named · function · L66-L69 — def test_ambiguous_protein_is_named(run)
- test_ambiguous_protein_gives_no_locus_specific_support · function · L72-L78 — def test_ambiguous_protein_gives_no_locus_specific_support(run)
- test_intron_support_counts_only_unambiguous_organisms · function · L81-L86 — def test_intron_support_counts_only_unambiguous_organisms(run)
- test_summary_records_the_exclusion · function · L89-L96 — def test_summary_records_the_exclusion(run)
- test_alignments_table_still_lists_the_ambiguous_alignments · function · L99-L103 — def test_alignments_table_still_lists_the_ambiguous_alignments(run)
- test_intermediates_are_removed · function · L106-L107 — def test_intermediates_are_removed(run)
