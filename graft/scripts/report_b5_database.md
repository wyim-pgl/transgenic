# scripts/report_b5_database.py

- classify · function · L46-L50 — def classify(reason)
- tier_margin_unguaranteed · function · L53-L86 — def tier_margin_unguaranteed(con)
- canonical_order_audit · function · L89-L109 — def canonical_order_audit(con)
- distribution · function · L112-L121 — def distribution(values)
- quantile · function · L115-L118 — def quantile(p)
- evaluation_population · function · L124-L180 — def evaluation_population(con, reference_distributions=False)
- labelled_gene_checks · function · L183-L200 — def labelled_gene_checks(con): # Gene-level assertions the validator structurally cannot make: geneList.orthogroup_id is NULL for every # tile row, so validate_split's group key degenerates to the window id and its cross-split check is inert. # window_genes holds only the genes a tile actually labelled (masked genes are dropped before the insert).
- export_population_tables · function · L202-L226 — def export_population_tables(report, out_dir)
- main · function · L229-L331 — def main()
