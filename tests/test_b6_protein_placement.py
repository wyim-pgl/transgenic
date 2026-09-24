"""WP2-prot (plan v4.1 §4): protein-placement supplement from the raw miniprot GFF + reference, joined to C0."""
import json
import shutil
from pathlib import Path

import pytest

import test_b6_evidence_world as EW
import b6_build_protein_placement as BP
from transgenic.evidence import b6_protein_placement as PP
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def ev(tmp_path_factory):
    return EW.build_evidence_world(tmp_path_factory.mktemp("prot_world"))


def _build(ev, out, convention="inside_last_block", **over):
    kw = dict(species=EW.SPECIES, gff=ev["protein_gff"], reference=ev["fasta"], protein_c0=ev["dbs"]["protein"],
              merged=ev["merged"], out=out, stop_convention=convention)
    kw.update(over)
    return PP.build(**kw)


@pytest.fixture(scope="module")
def supplement(ev):
    out = ev["tmp"] / f"protein_placement_b6_v1.{EW.SPECIES}.db"
    before = LK.sha256(ev["dbs"]["protein"])
    prov = _build(ev, out)
    assert LK.sha256(ev["dbs"]["protein"]) == before          # no c0-v1 change
    return out, prov


def rows(db):
    con = duckdb.connect(str(db), read_only=True)
    try:
        cols = [c for c, _ in PP.COLUMNS]
        return {r[cols.index("protein_id")]: dict(zip(cols, r))
                for r in con.execute(f"SELECT {', '.join(cols)} FROM protein_placement ORDER BY alignment_id").fetchall()}
    finally:
        con.close()


def test_one_row_per_accepted_primary_placement(supplement):
    got = rows(supplement[0])
    assert sorted(got) == ["39947_0:000013", "39947_0:000022", "4577_0:000002", "4577_0:000011", "4577_0:000012",
                           "4577_0:000014", "4577_0:000021"]            # no ambiguous 15, no secondary of 14


def test_alignment_id_joins_c0_uniquely_and_flags_are_inherited(ev, supplement):
    got = rows(supplement[0])
    con = duckdb.connect(str(ev["dbs"]["protein"]), read_only=True)
    c0 = dict(con.execute("SELECT read_name, alignment_id FROM evidence_alignment WHERE accepted AND "
                          "placement_kind = 'primary' ORDER BY read_name").fetchall())
    con.close()
    assert {p: r["alignment_id"] for p, r in got.items()} == c0
    assert got["4577_0:000002"]["molecule_state"] == "excluded_heldout"
    assert got["4577_0:000011"]["molecule_state"] == "train_track" and got["4577_0:000011"]["organism"] == "4577_0"


def test_complete_orfs_on_both_strands_with_gff_phase(ev, supplement):
    got = rows(supplement[0])
    for p, strand in (("4577_0:000011", "+"), ("4577_0:000012", "-"), ("4577_0:000014", "+")):
        r = got[p]
        assert r["strand"] == strand and r["orf_complete"] and r["start_codon_ok"] and r["stop_codon_ok"]
        assert r["qstart0"] == 0 and r["qend"] == r["qlen"] and r["internal_stop_n"] == 0 and r["frameshift_n"] == 0
        blocks = json.loads(r["cds_blocks"])
        assert len(blocks) == r["n_blocks"] == 2
        tx = blocks if strand == "+" else blocks[::-1]
        assert [b[2] for b in tx] == [0, 2]                    # 61-nt first exon -> second CDS phase 2
        assert sum(b[1] - b[0] for b in blocks) == 3 * r["qlen"] + 3   # the stop codon is inside the last block


def test_partial_query_is_not_orf_complete(supplement):
    r = rows(supplement[0])["39947_0:000013"]
    assert r["qstart0"] == 3 and r["start_codon_ok"] and r["stop_codon_ok"] and not r["orf_complete"]


def test_provenance_binds_every_input_and_the_measured_convention(ev, supplement):
    out, prov = supplement
    assert prov["inputs"]["miniprot_gff"]["sha256"] == LK.sha256(ev["protein_gff"])
    assert prov["inputs"]["reference"]["sha256"] == LK.sha256(ev["fasta"])
    assert prov["inputs"]["merged"]["sha256"] == LK.sha256(ev["merged"])
    assert prov["stop_convention"] == "inside_last_block"
    m = prov["stop_convention_measurement"]
    assert m["accepted_primaries"] == 7 and m.get("inside_last_block", 0) + m.get("both", 0) >= 4
    sha, p2 = PP.open_supplement(out)
    assert sha == prov["output_sha256"] == LK.sha256(out) and not Path(str(out) + ".wal").exists()


# ------------------------------------------------------------------------------ both stop conventions (unit fixtures)
def _record(blocks, strand, qlen, qstart=0):
    return {"protein": "4577_0:9", "organism": "4577_0", "chrom": "c", "strand": strand, "score": 1,
            "start1": blocks[0][0], "end1": blocks[-1][1], "cds": blocks,
            "paf": {"qlen": qlen, "qstart": qstart, "qend": qlen, "fs": 0}}


@pytest.mark.parametrize("strand", ["+", "-"])
def test_after_last_block_convention(strand):
    seq = "CC" + "ATG" + "AAACCC" + "GG" + "TTT" + "TAA" + "CC"     # exon1 ATGAAACCC | intron GG | exon2 TTT | stop TAA
    blocks = [(3, 11, 0), (14, 16, 0)]                              # 1-based: ATGAAACCC and TTT, stop after exon 2
    genome = {"c": seq if strand == "+" else EW.revcomp(seq)}
    if strand == "-":
        L = len(seq)
        blocks = sorted((L - e + 1, L - s + 1, ph) for s, e, ph in blocks)
    after = PP.placement_fields(genome, _record(blocks, strand, 4), "after_last_block")
    inside = PP.placement_fields(genome, _record(blocks, strand, 4), "inside_last_block")
    assert after["start_codon_ok"] and after["stop_codon_ok"] and after["orf_complete"] and after["internal_stop_n"] == 0
    assert not inside["stop_codon_ok"] and not inside["orf_complete"]


def test_inside_convention_counts_internal_stops_but_not_the_terminal_one():
    genome = {"c": "ATGAAATAACCCTGA"}
    f = PP.placement_fields(genome, _record([(1, 15, 0)], "+", 4), "inside_last_block")
    assert f["stop_codon_ok"] and f["internal_stop_n"] == 1
    assert PP.measure_stop_convention  # used end to end above


def test_unknown_convention_is_refused():
    with pytest.raises(PP.PlacementError):
        PP.codon_context({"c": "ATGTAA"}, "c", "+", [(0, 6)], 0, "guess")


# ------------------------------------------------------------------------------ fail-closed inputs
def test_gff_reference_merged_and_output_are_verified(ev, supplement, tmp_path):
    bad = tmp_path / "p.gff"
    shutil.copyfile(ev["protein_gff"], bad)
    with open(bad, "a") as fh:
        fh.write("# edited\n")
    with pytest.raises(PP.PlacementError, match="GFF the protein C0 recorded"):
        _build(ev, tmp_path / "a.db", gff=bad)
    ref = tmp_path / "r.fa"
    ref.write_text(Path(ev["fasta"]).read_text() + ">x\nA\n")
    with pytest.raises(PP.PlacementError, match="reference"):
        _build(ev, tmp_path / "b.db", reference=ref)
    merged = tmp_path / "m.db"
    shutil.copyfile(ev["merged"], merged)
    with pytest.raises(PP.PlacementError, match="sidecar"):
        _build(ev, tmp_path / "c.db", merged=merged)
    with pytest.raises(PP.PlacementError, match="exists"):
        _build(ev, supplement[0])


def test_cli_measure_and_build_modes(ev, tmp_path, capsys):
    assert BP.main(["--measure-stop-convention", "--gff", str(ev["protein_gff"]), "--reference", str(ev["fasta"])]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["accepted_primaries"] == 7
    assert BP.main(["--gff", str(ev["protein_gff"]), "--reference", str(ev["fasta"])]) == 1
    out = tmp_path / "cli.db"
    assert BP.main(["--species", EW.SPECIES, "--gff", str(ev["protein_gff"]), "--reference", str(ev["fasta"]),
                    "--protein-c0", str(ev["dbs"]["protein"]), "--merged", str(ev["merged"]),
                    "--stop-convention", "inside_last_block", "--out", str(out)]) == 0
    assert PP.open_supplement(out)[0] == LK.sha256(out)


# ------------------------------------------------------------------------------ review 1 M5: split terminal codons
def test_start_and_inside_stop_codons_split_across_cds_blocks_are_read_from_the_spliced_cds():
    genome = {"c": "A" + "GGGGGG" + "TGAAATAA"}                 # [0,1)='A', [7,15)='TGAAATAA' -> 'ATGAAATAA'
    ok = PP.codon_context(genome, "c", "+", [(0, 1), (7, 15)], 0, "inside_last_block")
    assert ok == (True, True, 0)
    stop_split = {"c": "ATGAAATA" + "CCCCC" + "A"}                # stop TA|A split over the last intron
    assert PP.codon_context(stop_split, "c", "+", [(0, 8), (13, 14)], 0, "inside_last_block") == (True, True, 0)
    minus = {"c": EW.revcomp("A" + "GGGGGG" + "TGAAATAA")}
    assert PP.codon_context(minus, "c", "-", [(0, 8), (14, 15)], 0, "inside_last_block") == (True, True, 0)


def test_exterior_stop_convention_still_reads_the_bases_after_the_last_block():
    genome = {"c": "AT" + "CCC" + "GAAA" + "TAG"}                # A T | intron | G AAA, then exterior TAG
    assert PP.codon_context(genome, "c", "+", [(0, 2), (5, 9)], 0, "after_last_block") == (True, True, 0)
    assert PP.codon_context(genome, "c", "+", [(0, 2), (5, 9)], 0, "inside_last_block")[1] is False
