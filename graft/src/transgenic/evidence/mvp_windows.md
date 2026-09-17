# src/transgenic/evidence/mvp_windows.py

- WindowError · class · L57-L58 — class WindowError(ValueError)
- _require · function · L61-L63 — def _require(condition, message)
- PanelLocus · class · L67-L83 — class PanelLocus
- gene_start0 · method · L77-L78 — def gene_start0(self)
- gene_end0 · method · L81-L83 — def gene_end0(self)
- sha256_file · function · L86-L91 — def sha256_file(path)
- load_panel · function · L94-L136 — def load_panel(path, expected_sha256=PANEL_SHA256, labels=MVP_LABELS)
- evaluation_window · function · L139-L160 — def evaluation_window(locus, chrom_length=None)
- prefilter_interval · function · L163-L172 — def prefilter_interval(window)
- windows_by_chromosome · function · L175-L181 — def windows_by_chromosome(loci, chrom_lengths=None)
- prefilter_by_chromosome · function · L184-L190 — def prefilter_by_chromosome(loci)
- _contained · function · L193-L197 — def _contained(span_start0, span_end0_incl, intervals)
- _overlapping · function · L200-L204 — def _overlapping(span_start0, span_end0_incl, intervals)
- junction_within_windows · function · L207-L213 — def junction_within_windows(chrom, intron_start0, intron_end0, windows)
- genes_by_chromosome · function · L216-L222 — def genes_by_chromosome(loci)
- junction_within_genes · function · L225-L229 — def junction_within_genes(chrom, intron_start0, intron_end0, genes)
- transcript_overlaps_gene · function · L232-L236 — def transcript_overlaps_gene(chrom, tx_start0, tx_end0_incl, genes)
