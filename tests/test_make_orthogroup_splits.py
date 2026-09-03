"""scripts/make_orthogroup_splits.py (#14): OrthoFinder orthogroups + reference GFFs -> data/splits/b5_orthogroup_split_v1.tsv.

Rules (docs/gsf_spec_v1.md §7, protocol A18/A29): one row per gene of every training annotation, gene_id = the builder key
(gsf_contract.gene_key: generated code for long/dotted ids), orthogroup_id from OrthoFinder (empty for singletons: unassigned
genes and genes without a protein), orthogroup-level 75/10/15 with a seed, strict held-out loci (A. thaliana list, names
without the .TAIR10 suffix) and their whole orthogroups forced to test; validate_split must pass.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_orthogroup_splits.py"


def _load(path, name):
    if not path.exists():
        pytest.fail(f"{path} does not exist yet (RED state)", pytrace=False)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def mk():
    return _load(SCRIPT, "make_orthogroup_splits")


def _gff(path, species_prefix, genes):
    lines = ["##gff-version 3"]
    pos = 1000
    for gid, name in genes:
        s, e = pos, pos + 599
        lines += [f"Chr1\tt\tgene\t{s}\t{e}\t.\t+\t.\tID={gid};Name={name}",
                  f"Chr1\tt\tmRNA\t{s}\t{e}\t.\t+\t.\tID={gid}.1;Parent={gid};Name={name}.1",
                  f"Chr1\tt\tCDS\t{s}\t{e}\t.\t+\t0\tID={gid}.1.cds;Parent={gid}.1"]
        pos += 2000
    path.write_text("\n".join(lines) + "\n")


def _inputs(tmp_path):
    ath = tmp_path / "Ath.gff3"
    _gff(ath, "Ath", [("AT1G01010.TAIR10", "AT1G01010"), ("AT1G01020.TAIR10", "AT1G01020"), ("AT1G01030.TAIR10", "AT1G01030"), ("AT1G01040.TAIR10", "AT1G01040")])
    osa = tmp_path / "Osa.gff3"
    _gff(osa, "Osa", [("LOC_Os01g01010.MSUv7.0", "LOC_Os01g01010"), ("LOC_Os01g01019.MSUv7.0", "LOC_Os01g01019")])
    og = tmp_path / "Orthogroups.tsv"
    og.write_text("Orthogroup\tAthaliana\tOsativa\n"
                  "OG0000000\tAT1G01010.TAIR10, AT1G01020.TAIR10\tLOC_Os01g01010.MSUv7.0\n"     # held-out gene inside -> whole group test
                  "OG0000001\tAT1G01040.TAIR10\tLOC_Os01g01019.MSUv7.0\n")
    un = tmp_path / "Orthogroups_UnassignedGenes.tsv"
    un.write_text("Orthogroup\tAthaliana\tOsativa\nOG0000002\tAT1G01030.TAIR10\t\n")                 # singleton by OrthoFinder
    held = tmp_path / "heldout.txt"
    held.write_text("AT1G01010\nAT9G99999\n")                                                        # second one is not in the GFF
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\n"
                        f"Athaliana\tA\tTAIR10\t/x.fa\t\t{ath}\t\t\nOsativa\tO\tMSU\t/y.fa\t\t{osa}\t\t\n")
    return og, un, held, manifest


def test_split_table_rows_keys_groups_and_holdout(mk, tmp_path):
    og, un, held, manifest = _inputs(tmp_path)
    out, summ = tmp_path / "split.tsv", tmp_path / "split.json"
    rc = mk.main(["--orthogroups", str(og), "--unassigned", str(un), "--species-manifest", str(manifest), "--strict-holdout", str(held),
                  "--strict-holdout-species", "Athaliana", "--seed", "123", "--source-version", "test-run", "--out", str(out), "--summary", str(summ)])
    assert rc == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "species_id\tgene_id\torthogroup_id\tsplit\tstrict_holdout\tseed\tsource_version"
    rows = [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]
    assert len(rows) == 6                                                                            # every GFF gene has a row
    by = {(r["species_id"], r["gene_id"]): r for r in rows}
    keys = sorted(k for k in by)
    assert all(not g.endswith(".TAIR10") and not g.endswith(".MSUv7.0") for _, g in keys)             # builder keys, not GFF ids
    s = json.loads(summ.read_text())
    ath_key = s["key_of"]["Athaliana"]["AT1G01010.TAIR10"]
    assert by[("Athaliana", ath_key)]["strict_holdout"] == "true" and by[("Athaliana", ath_key)]["split"] == "test"
    og0 = [r for r in rows if r["orthogroup_id"] == "OG0000000"]
    assert len(og0) == 3 and {r["split"] for r in og0} == {"test"}                                   # whole orthogroup follows the held-out gene
    single = [r for r in rows if r["orthogroup_id"] == ""]
    assert {s["key_of"]["Athaliana"]["AT1G01030.TAIR10"]} == {r["gene_id"] for r in single}          # OrthoFinder singleton -> empty orthogroup_id
    assert all(r["seed"] == "123" and r["source_version"] == "test-run" for r in rows)
    assert s["strict_holdout"]["requested"] == 2 and s["strict_holdout"]["found"] == 1 and s["strict_holdout"]["missing"] == ["AT9G99999"]
    assert s["validation_violations"] == [] and s["unmapped_orthofinder_genes"] == {}
    assert s["per_species"]["Athaliana"]["test"] == 2 and s["per_species"]["Osativa"]["test"] == 1   # OG0000000 only (held-out group)
    assert sum(s["per_species"]["Osativa"].values()) == 2


def test_gene_without_protein_becomes_singleton_and_seed_is_reproducible(mk, tmp_path):
    og, un, held, manifest = _inputs(tmp_path)
    # remove AT1G01030 from the unassigned list: it is then a GFF gene absent from OrthoFinder -> singleton row all the same
    un.write_text("Orthogroup\tAthaliana\tOsativa\n")
    outs = []
    for i in range(2):
        out, summ = tmp_path / f"s{i}.tsv", tmp_path / f"s{i}.json"
        assert mk.main(["--orthogroups", str(og), "--unassigned", str(un), "--species-manifest", str(manifest), "--strict-holdout", str(held),
                        "--strict-holdout-species", "Athaliana", "--seed", "7", "--source-version", "t", "--out", str(out), "--summary", str(summ)]) == 0
        outs.append(out.read_text())
    assert outs[0] == outs[1]
    s = json.loads((tmp_path / "s0.json").read_text())
    assert s["singletons"]["Athaliana"] == 1 and s["genes_without_orthofinder_entry"]["Athaliana"] == 1
