"""Protocol A19.1 leakage filter for the OrthoDB v12 Viridiplantae partition (revision/scripts/64_orthodb_filter.py).

Headers of the partitioned file are `>taxid_version:gene<TAB>taxid_version`. The filter removes (1) every sequence
whose organism taxid is an evaluated species or a sub-taxon of one (lineage from an NCBI taxonomy table, fail closed
when a taxid has no lineage), and (2) every sequence identical to a protein of an excluded proteome (evaluated-species
reference proteomes; later the training-species test-orthogroup proteins of #14). Counts per taxid before/after and
md5s are written for the provenance record.
"""
import gzip
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "revision" / "scripts" / "64_orthodb_filter.py"


def _load(path, name):
    if not path.exists():
        pytest.fail(f"{path} does not exist yet (RED state)", pytrace=False)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def od():
    return _load(SCRIPT, "orthodb_filter")


SEQS = {
    "3702_0:000001": "MAAATHALIANA",
    "381124_0:000001": "MZEAMAYSSUBSP",      # Zea mays subsp. mays -> sub-taxon of 4577
    "4081_0:000001": "MSOLANUMLYCOP",
    "3708_0:000001": "MBRASSICANAPUS",
    "4565_0:000001": "MTRITICUMONE",
    "4565_0:000002": "MTRITICUMTWO",          # identical to an excluded-proteome protein -> removed
}
LINEAGE = {"3702": "3702;3701;980083;3700;91836", "381124": "381124;4577;4575;147370;4479", "4081": "4081;4107;4070;4069",
           "3708": "3708;3705;981071;3700;91836", "4565": "4565;4564;147389;4479"}


def _write_inputs(tmp_path):
    fa = tmp_path / "odb.fa.gz"
    with gzip.open(fa, "wt") as fh:
        for h, s in SEQS.items():
            fh.write(f">{h}\t{h.split(':')[0]}\n{s}\n")
    lin = tmp_path / "lineage.tsv"
    lin.write_text("taxid\tlineage\n" + "".join(f"{t}\t{l}\n" for t, l in LINEAGE.items()))
    prot = tmp_path / "Zmays.protein.fa"
    prot.write_text(">Zm00001d000001_T001\nMTRITICUMTWO*\n>Zm00001d000002_T001\nMSOMETHINGELSE\n")
    return fa, lin, prot


def test_header_parsing_and_lineage_exclusion(od):
    assert od.taxid_of_header(">381124_0:000001\t381124_0") == 381124
    assert od.taxid_of_header(">3702_0:000001") == 3702
    lineage = {int(t): [int(x) for x in l.split(";")] for t, l in LINEAGE.items()}
    assert od.excluded_taxids({3702, 4577, 4081}, lineage) == {3702, 381124, 4081}
    with pytest.raises(SystemExit):
        od.excluded_taxids({3702}, {}, present={3702, 999999})         # missing lineage -> fail closed


def test_filter_writes_fasta_counts_and_summary(od, tmp_path):
    fa, lin, prot = _write_inputs(tmp_path)
    out = tmp_path / "out"
    rc = od.main(["--fasta", str(fa), "--out-dir", str(out), "--lineage", str(lin), "--exclude-taxid", "3702", "4577", "4081",
                  "--exclude-proteome", f"Zmays={prot}"])
    assert rc == 0
    kept = {}
    with gzip.open(out / "odb12_Viridiplantae.filtered.fa.gz", "rt") as fh:
        name = None
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                kept[name] = ""
            else:
                kept[name] += line.strip()
    assert set(kept) == {"3708_0:000001", "4565_0:000001"}
    counts = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in (out / "counts_by_taxid.tsv").read_text().splitlines()[1:]}
    assert counts["3702"][1:3] == ["1", "0"] and counts["381124"][1:3] == ["1", "0"] and counts["4565"][1:3] == ["2", "1"]
    summary = json.loads((out / "filter_summary.json").read_text())
    assert summary["sequences_in"] == 6 and summary["sequences_out"] == 2
    assert summary["removed_by_taxid"] == 3 and summary["removed_by_exact_match"] == {"Zmays": 1}
    assert set(summary["excluded_taxids"]) == {3702, 381124, 4081}
    assert summary["input_md5"] and summary["output_md5"]


def test_lineage_xml_parsing(od):
    xml = ("<TaxaSet><Taxon><TaxId>381124</TaxId><ScientificName>Zea mays subsp. mays</ScientificName>"
           "<LineageEx><Taxon><TaxId>4479</TaxId></Taxon><Taxon><TaxId>147370</TaxId></Taxon><Taxon><TaxId>4575</TaxId></Taxon>"
           "<Taxon><TaxId>4577</TaxId></Taxon></LineageEx></Taxon>"
           "<Taxon><TaxId>3702</TaxId><LineageEx><Taxon><TaxId>3701</TaxId></Taxon></LineageEx></Taxon></TaxaSet>")
    lin = od.parse_taxonomy_xml(xml)
    assert lin[381124][0] == 381124 and 4577 in lin[381124] and lin[3702] == [3702, 3701]
