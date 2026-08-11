#!/usr/bin/env python3
"""Replicate exactly which A. thaliana genes genome2GSFDataset inserts into the
training DuckDB, without needing duckdb or torch.

The training DB was built as
    Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db
i.e. genome2GSFDataset(..., mode="train", staticSize=6144, addExtra=200,
addRC=True, addRCIsoOnly=True, clean=True, maxLen=49152).

`duckdb` is replaced by a shim whose .sql() records the geneModel of every INSERT,
and `torch` by a shim providing randint. Neither affects which genes are inserted.
Every skip message preprocess.py writes to stderr is captured so each excluded gene
gets a reason.
"""
import json
import re
import sys
import types
from pathlib import Path

SRC = "/data/gpfs/assoc/pgl/data/Transgenic/transgenic/src"
CMP = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic_comparison")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

GENOME = str(CMP / "genomes" / "Athaliana_167_TAIR10.fa")
GFF3 = str(CMP / "reference_annotations" / "Athaliana_167_TAIR10.gene.clean.gff3")

inserted = []          # geneModel of every row the real code would INSERT
_INSERT_RE = re.compile(r"INSERT INTO geneList[^)]*\)\s*VALUES\s*\(nextval\('row_id'\),\s*'([^']*)'")


class _Con:
    def sql(self, q):
        if q.lstrip().upper().startswith("INSERT INTO GENELIST"):
            m = _INSERT_RE.search(q)
            if not m:
                raise RuntimeError(f"could not parse INSERT: {q[:200]}")
            inserted.append(m.group(1))
        return None

    def close(self):
        pass


duckdb_shim = types.ModuleType("duckdb")
duckdb_shim.connect = lambda *a, **k: _Con()
sys.modules["duckdb"] = duckdb_shim

torch_shim = types.ModuleType("torch")
torch_shim.randint = lambda high, size: _Int(0)
sys.modules.setdefault("torch", torch_shim)


class _Int(int):
    pass


# pandas / tqdm / numpy are imported by preprocess but play no part in which genes
# are inserted; stub whichever the interpreter lacks.
for _mod in ("pandas", "numpy"):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = types.ModuleType(_mod)
try:
    import tqdm  # noqa: F401
except ImportError:
    _t = types.ModuleType("tqdm")
    _t.tqdm = lambda it, *a, **k: it
    sys.modules["tqdm"] = _t

sys.path.insert(0, SRC)

# Import preprocess.py without executing the package __init__ files, which pull in
# torch.nn / transformers that this replication does not need. Registering stub
# package modules whose __path__ points at the real directories lets the normal
# import machinery find the real preprocess.py and utils submodules.
for _name, _sub in (("transgenic", ""), ("transgenic.datasets", "/datasets"),
                    ("transgenic.utils", "/utils")):
    _m = types.ModuleType(_name)
    _m.__path__ = [SRC + "/transgenic" + _sub]
    sys.modules[_name] = _m

import importlib  # noqa: E402

genome2GSFDataset = importlib.import_module(
    "transgenic.datasets.preprocess").genome2GSFDataset


def main():
    err_path = OUT / "preprocess_stderr.txt"
    real_stderr = sys.stderr
    with err_path.open("w") as errfh:
        sys.stderr = errfh
        try:
            genome2GSFDataset(
                genome=GENOME,
                gff3=GFF3,
                db="/dev/null",
                anoType="gff",
                mode="train",
                maxLen=49152,
                addExtra=200,
                staticSize=6144,
                addRC=True,
                addRCIsoOnly=True,
                clean=True,
                speciesPrefix="",
            )
        finally:
            sys.stderr = real_stderr

    # Genes present in the annotation, in file order
    all_genes = []
    with open(GFF3) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            all_genes.append(f[8].split(";")[0].split("=")[1])

    ins_nonrc = [g for g in inserted if not g.endswith("-rc")]
    ins_rc = [g for g in inserted if g.endswith("-rc")]
    in_db = set(ins_nonrc)
    missing = [g for g in all_genes if g not in in_db]

    # Attribute a reason to each excluded gene from the captured stderr
    reasons = {}
    oversize = re.compile(r"^Skipping (\S+) because gene length > ")
    startc = re.compile(r"^Start codon missing in \S+ of (\S+)\.\. Skipping")
    stopc = re.compile(r"^Stop codon missing in \S+ of (\S+)\.\. Skipping")
    mult3 = re.compile(r"^(\S+) not a multiple of 3\.\. Skipping")
    verr = re.compile(r"^Error validating (\S+), skipping: ")
    chrmiss = re.compile(r"^Skipping (\S+) because chromosome ")
    idxbad = re.compile(r"^Skipping (\S+) because invalid sequence index")
    seq0 = re.compile(r"^Skipping (\S+) because extracted sequence length is 0")
    for line in err_path.read_text().splitlines():
        for rx, label in (
            (oversize, "oversize_maxLen_49152"),
            (startc, "clean_start_codon_missing"),
            (stopc, "clean_stop_codon_missing"),
            (mult3, "clean_not_multiple_of_3"),
            (verr, "clean_validate_error"),
            (chrmiss, "chromosome_missing"),
            (idxbad, "invalid_index_range"),
            (seq0, "empty_sequence"),
        ):
            m = rx.match(line)
            if m:
                reasons.setdefault(m.group(1), label)
                break

    reason_counts = {}
    unexplained = []
    for g in missing:
        r = reasons.get(g)
        if r is None:
            # the final gene in file order is never flushed (no end-of-file flush)
            r = "never_flushed_last_gene_in_file" if g == all_genes[-1] else "unexplained"
            if r == "unexplained":
                unexplained.append(g)
        reason_counts[r] = reason_counts.get(r, 0) + 1

    summary = {
        "genome_fasta": GENOME,
        "annotation_gff3": GFF3,
        "params": {
            "mode": "train", "maxLen": 49152, "addExtra": 200, "staticSize": 6144,
            "addRC": True, "addRCIsoOnly": True, "clean": True,
        },
        "genes_in_annotation": len(all_genes),
        "rows_inserted_total": len(inserted),
        "rows_inserted_nonrc": len(ins_nonrc),
        "rows_inserted_rc": len(ins_rc),
        "distinct_genes_in_db": len(in_db),
        "genes_absent_from_db": len(missing),
        "absence_reasons": reason_counts,
        "unexplained_examples": unexplained[:20],
    }
    (OUT / "db_membership_summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "at_genes_in_db.txt").write_text("\n".join(sorted(in_db)) + "\n")
    (OUT / "at_genes_not_in_db.tsv").write_text(
        "gene\treason\n" + "".join(
            f"{g}\t{reasons.get(g, 'never_flushed_last_gene_in_file' if g == all_genes[-1] else 'unexplained')}\n"
            for g in missing) )
    (OUT / "at_genes_rc_in_db.txt").write_text("\n".join(sorted(ins_rc)) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
