"""b6_est_molecule_map.py: the single EST molecule-id export consumed by part B (b6_hsp_project --molecule-map)."""
import hashlib
import json

import pytest

import test_b6_c0_world as W
import b6_build_c0 as B
import b6_est_molecule_map as MM

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def est_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("map")
    world = W.build_corpus(tmp)
    W.write_est(tmp)
    args = W.cli_args(world, tmp)
    assert B.main(args) == 0
    return args[args.index("--out") + 1]


def _rows(path):
    lines = path.read_text().splitlines()
    return lines[0].split("\t"), [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]


def test_map_exports_the_c0_units_with_states_and_a_sha256_sidecar(est_db, tmp_path):
    out = tmp_path / "Athaliana.est_molecule_map.tsv"
    result = MM.export(est_db, W.SPECIES, out)
    header, rows = _rows(out)
    assert header == list(MM.COLUMNS)
    by = {r["qname"]: r for r in rows}
    assert by["e1"]["superseded"] == "true" and by["e2"]["superseded"] == "false"   # superseded version flagged
    assert by["e2"]["est_accession"] == "EA1.2" and by["e1"]["molecule_id"] == by["e2"]["molecule_id"]
    assert sum(r["superseded"] == "true" for r in rows) == result["superseded_rows"] == 1
    assert by["e2"]["molecule_id"] == hashlib.sha256(json.dumps(["e1", "e2"], separators=(",", ":")).encode()).hexdigest()
    assert by["e3"]["molecule_id"] == by["e4"]["molecule_id"]                      # A21 parts
    assert by["h1"]["molecule_state"] == "excluded_heldout" and by["e2"]["molecule_state"] == "train_track"
    assert by["v1"]["train_tiles_excluded"] == "true" and by["e2"]["train_tiles_excluded"] == "false"
    assert all(r["member_uid"] == r["qname"] for r in rows)
    # accepted = the member has an A18.5-accepted primary (finding 3): u1 unmapped, s1 chimeric, m1 ambiguous
    assert {r["qname"] for r in rows if r["accepted"] == "false"} == {"u1", "s1", "m1"}
    assert by["e1"]["accepted"] == "true" and result["rejected_rows"] == 3
    assert "rejected_rows\t3" in (tmp_path / (out.name + ".PROVENANCE.txt")).read_text()
    assert (tmp_path / (out.name + ".sha256")).read_text().split() == [result["sha256"], out.name]
    con = duckdb.connect(est_db, read_only=True)
    c0 = dict(con.execute("SELECT read_name, molecule_id FROM molecule_member").fetchall())
    con.close()
    assert all(c0[r["qname"]] == r["molecule_id"] for r in rows)


def test_part_b_loader_accepts_the_map(est_db, tmp_path):
    hp = B._load("b6_hsp_project", B._SCRIPTS / "evidence" / "b6_hsp_project.py")
    out = tmp_path / "m.tsv"
    MM.export(est_db, W.SPECIES, out)
    by_qname = hp.load_molecule_map(out)
    assert by_qname["e2"]["molecule_state"] == "train_track" and by_qname["v1"]["train_tiles_excluded"] is True
    assert by_qname["e2"]["accepted"] is True and by_qname["u1"]["accepted"] is False


def test_map_refuses_unpublished_or_tampered_db_and_clobber(est_db, tmp_path):
    import shutil

    out = tmp_path / "m.tsv"
    MM.export(est_db, W.SPECIES, out)
    with pytest.raises(FileExistsError):
        MM.export(est_db, W.SPECIES, out)
    copy = tmp_path / "c.db"
    shutil.copy(est_db, copy)
    shutil.copy(est_db + ".PROVENANCE.txt", str(copy) + ".PROVENANCE.txt")
    con = duckdb.connect(str(copy))
    con.execute("UPDATE molecule_member SET molecule_state = 'train_track'")
    con.close()
    with pytest.raises(B.C0Error, match="PROVENANCE output_sha256"):
        MM.export(copy, W.SPECIES, tmp_path / "x.tsv")
    assert MM.main(["--est-c0", est_db, "--species", "Osativa", "--out", str(tmp_path / "y.tsv")]) == 1


def test_map_export_is_byte_deterministic(est_db, tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    MM.export(est_db, W.SPECIES, a)
    MM.export(est_db, W.SPECIES, b)
    assert a.read_bytes() == b.read_bytes()
    names = [l.split("\t")[0] for l in a.read_text().splitlines()[1:]]
    assert names == sorted(names)


def test_map_field_rule_allows_spaces_in_library_only():
    row = dict(qname="q", member_uid="q", molecule_id="a" * 64, library="Lib A root", est_accession="A.1",
               molecule_state="train_track", train_tiles_excluded="false", superseded="false", accepted="true")
    assert MM.check_fields(row) is None
    for bad in (dict(library="Lib\tA"), dict(library="Lib\rA"), dict(library=""), dict(qname="q 1"),
                dict(est_accession="A\n1"), dict(superseded=""), dict(accepted="")):
        assert MM.check_fields(dict(row, **bad)) is not None


# ------------------------------------------------------------------ HIGH-1: a real multi-word /clone_lib end to end
LIB_L1 = "Arabidopsis thaliana Col-0 mixed tissue cDNA library"       # wrapped across two GenBank lines below
LIB_L2 = "Arabidopsis thaliana Ler flower buds cDNA library"


def _genbank_record(acc, lib, clone):
    lib_lines = ('                     /clone_lib="Arabidopsis thaliana Col-0 mixed tissue\n'
                 '                     cDNA library"\n') if lib == LIB_L1 else f'                     /clone_lib="{lib}"\n'
    return (f"LOCUS       {acc}                 100 bp    mRNA    linear   EST 01-JAN-2000\nVERSION     {acc}\n"
            "FEATURES             Location/Qualifiers\n     source          1..100\n"
            '                     /organism="Arabidopsis thaliana"\n' + lib_lines
            + (f'                     /clone="{clone}"\n' if clone else "") + "ORIGIN\n//\n")


def test_multiword_clone_lib_runs_metadata_to_map_to_projection(tmp_path):
    import random

    import b6_c0_metadata as CM
    from revision.scripts.evidence.est_blastx import search_command
    from test_b6_hsp_project import SAFE, raw_row

    world = W.build_corpus(tmp_path)
    align, _ = W.write_est(tmp_path)
    d = align / W.SPECIES
    for name in ("c0_metadata.tsv", "c0_metadata.tsv.sha256"):
        (d / name).unlink()
    rng = random.Random(3)
    coding = "".join(rng.choice(SAFE) for _ in range(200))            # e2: 600 nt, 3 exons of 200 (W.GA3)
    fasta, gb = [], []
    for read, _, lib, clone, kw in W.EST_READS:
        qlen = sum(e - s + 1 for s, e in kw["exons"]) or 150
        seq = coding if read == "e2" else "A" * qlen
        assert len(seq) == qlen
        fasta.append(f">{read} EST {read} univec=keep:1-{qlen}/{qlen}\n{seq}\n")
        gb.append(_genbank_record(read, LIB_L1 if lib == "L1" else LIB_L2, clone))
    screened = tmp_path / "est.univec.fa"
    screened.write_text("".join(fasta))
    (tmp_path / "est.gb").write_text("".join(gb))
    _, est_roles = W.write_roles(tmp_path)
    assert CM.main(["--source", "est", "--species", W.SPECIES, "--roles", str(est_roles), "--screened", str(screened),
                    "--genbank", str(tmp_path / "est.gb"), "--bioproject", "PRJNA1",
                    "--out", str(d / "c0_metadata.tsv")]) == 0
    meta = {l.split("\t")[0]: l.split("\t") for l in (d / "c0_metadata.tsv").read_text().splitlines()}
    assert meta["e2"][2] == LIB_L1 and meta["e7"][2] == LIB_L2        # wrapped qualifier joined with one space
    args = W.cli_args(world, tmp_path)
    assert B.main(args) == 0
    est_c0 = args[args.index("--out") + 1]

    out = tmp_path / "map.tsv"
    MM.export(est_c0, W.SPECIES, out)
    _, rows = _rows(out)
    by = {r["qname"]: r for r in rows}
    assert by["e2"]["library"] == LIB_L1 and by["e7"]["library"] == LIB_L2

    hp = B._load("b6_hsp_project", B._SCRIPTS / "evidence" / "b6_hsp_project.py")
    raw, protein = raw_row("e2", "4577_0:000001", coding, 0, 60)
    (tmp_path / "orthodb.raw13.tsv").write_text("\t".join(raw) + "\n")
    (tmp_path / "orthodb.fa").write_text(f">4577_0:000001\t4577_0\n{protein}\n")
    (tmp_path / "orthodb.commands.json").write_text(json.dumps(
        [["diamond", "makedb"], search_command("/bin/diamond", screened, "db.dmnd", "o", 1, "t")]))
    hsp = tmp_path / "hsp.tsv"
    assert hp.main(["--species", W.SPECIES, "--diamond-version", "2.1.24.178", "--est-fasta", str(screened),
                    "--alignment", str(d / f"{W.SPECIES}.est.sam"), "--molecule-map", str(out),
                    "--lineage", str(W.write_lineage(tmp_path)), "--out", str(hsp), "--qc", str(tmp_path / "qc.json"),
                    "--raw13", f"orthodb={tmp_path / 'orthodb.raw13.tsv'}",
                    "--protein-fasta", f"orthodb={tmp_path / 'orthodb.fa'}",
                    "--blastx-command", f"orthodb={tmp_path / 'orthodb.commands.json'}"]) == 0
    head, *lines = hsp.read_text().splitlines()
    got = [dict(zip(head.split("\t"), l.split("\t"))) for l in lines]
    assert got and {r["library"] for r in got} == {LIB_L1}
    assert {r["molecule_id"] for r in got} == {by["e2"]["molecule_id"]}
