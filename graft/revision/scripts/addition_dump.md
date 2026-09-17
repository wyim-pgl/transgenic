# revision/scripts/addition_dump.py

- load_filter · function · L9-L14 — def load_filter()
- digest · function · L17-L22 — def digest(path, algorithm="sha256")
- introns · function · L25-L28 — def introns(structure): # Script 48's chain uses flanking CDS bases; C0 uses inclusive intronic bases.
- IndexedContig · class · L31-L52 — class IndexedContig
- __init__ · method · L33-L35 — def __init__(self, handle, length, offset, bases, width)
- __len__ · method · L37-L38 — def __len__(self)
- __getitem__ · method · L40-L52 — def __getitem__(self, key)
- indexed_genome · function · L55-L62 — def indexed_genome(path, handle, aliases)
- structure_rows · function · L65-L113 — def structure_rows(pred, ref, primary, metadata, genome, species, filters)
- write_frozen_dump · function · L116-L173 — def write_frozen_dump(destination, scorer)
