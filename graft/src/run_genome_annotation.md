# src/run_genome_annotation.py

- ensure_parent_dir · function · L70-L79 — def ensure_parent_dir(path)
- check_agat_installed · function · L82-L91 — def check_agat_installed()
- sort_gff · function · L93-L113 — def sort_gff(input_gff, output_gff, agat_cmd)
- main · function · L115-L702 — def main(): # ========================================================================= # STEP 0: Parse command-line arguments. # The two positional arguments (fasta_file, gff_file) are mandatory; all # others have sensible defaults. The device defaults to CUDA when a GPU is # detected, falling back to CPU otherwise. # ========================================================================
- load_checkpoint · function · L271-L282 — def load_checkpoint(path)
- save_checkpoint · function · L284-L292 — def save_checkpoint(path, completed)
- SkipSampler · class · L311-L320 — class SkipSampler(torch.utils.data.Sampler)
- __init__ · method · L312-L314 — def __init__(self, data_source, start_idx=0)
- __iter__ · method · L316-L317 — def __iter__(self)
- __len__ · method · L319-L320 — def __len__(self)
