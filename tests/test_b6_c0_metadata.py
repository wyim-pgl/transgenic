"""b6_c0_metadata.py: per-read c0_metadata.tsv producer (field sources documented, fail closed on missing fields)."""
import json

import pytest

import test_b6_c0_world as W
import b6_build_c0 as B
import b6_c0_metadata as CM

duckdb = pytest.importorskip("duckdb")

GB = """LOCUS       AB000001                 150 bp    mRNA    linear   EST 01-JAN-2000
VERSION     AB000001.1
DBLINK      BioProject: PRJNA111
FEATURES             Location/Qualifiers
     source          1..150
                     /organism="Arabidopsis thaliana"
                     /clone_lib="Lib A
                     root"
                     /clone="cX 5'"
ORIGIN
//
LOCUS       AB000002                 300 bp    mRNA    linear   EST 01-JAN-2000
VERSION     AB000002.1
FEATURES             Location/Qualifiers
     source          1..300
                     /organism="Arabidopsis thaliana"
                     /clone_lib="LibB"
     gene            1..300
                     /clone="not a source qualifier"
ORIGIN
//
"""


def _est(tmp_path, gb=GB, screened=None):
    (tmp_path / "est.gb").write_text(gb)
    screened = screened or (">AB000001.1 EST one univec=keep:1-150/150\n" + "A" * 150 + "\n"
                            ">AB000002.1_part1 EST two univec=split:1-120/300\n" + "C" * 120 + "\n"
                            ">AB000002.1_part2 EST two univec=split:181-300/300\n" + "G" * 120 + "\n")
    (tmp_path / "est.univec.fa").write_text(screened)
    _, est_roles = W.write_roles(tmp_path)
    return ["--source", "est", "--species", W.SPECIES, "--roles", str(est_roles), "--screened", str(tmp_path / "est.univec.fa"),
            "--genbank", str(tmp_path / "est.gb"), "--out", str(tmp_path / "c0_metadata.tsv")]


def _read(path):
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


def test_est_metadata_from_screened_fasta_and_genbank(tmp_path):
    assert CM.main(_est(tmp_path) + ["--bioproject", "PRJNA999"]) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["read"] == "*" and rows[0]["genotype_stratum"] == "reference" and rows[0]["bioproject"] == "PRJNA999"
    r1, p1, p2 = rows[1:]
    assert (r1["accession"], r1["library"], r1["clone"], r1["bioproject"], r1["post_trim_length"]) == \
        ("AB000001.1", "Lib A root", "cX 5'", "PRJNA111", "150")
    assert (p1["read"], p1["library"], p1["clone"], p1["bioproject"], p1["post_trim_length"]) == \
        ("AB000002.1_part1", "LibB", "", "", "120")
    assert p2["post_trim_length"] == "120"
    assert B.parse_accession(p2["accession"]) == ("AB000002", 1, 2)
    assert (tmp_path / "c0_metadata.tsv.sha256").read_text().split()[1] == "c0_metadata.tsv"


def test_est_metadata_fails_closed_on_missing_fields(tmp_path, capsys):
    assert CM.main(_est(tmp_path)) == 1                                   # AB000002 has no BioProject, none given
    assert "no BioProject" in capsys.readouterr().err and not (tmp_path / "c0_metadata.tsv").exists()
    assert CM.main(_est(tmp_path, gb=GB.replace('/clone_lib="LibB"\n', "")) + ["--bioproject", "P"]) == 1
    assert "no /clone_lib" in capsys.readouterr().err
    bad = ">AB000001.1 EST one univec=keep:1-150/150\n" + "A" * 149 + "\n"
    assert CM.main(_est(tmp_path, screened=bad) + ["--bioproject", "P"]) == 1
    assert "!= sequence length" in capsys.readouterr().err


def test_manifest_non_reference_is_normalised_to_the_c0_known_nonreference_token(tmp_path):
    """Finding 5: DATASET_ROLES.longread_v1.tsv spells the A18.5 'known non-reference' stratum `non_reference` (A47.3);
    the c0-v1 vocabulary is `known_nonreference`.  The producer maps it at ingestion; anything else fails closed."""
    assert CM.canonical_genotype("non_reference") == "known_nonreference"
    assert CM.canonical_genotype("known_nonreference") == "known_nonreference"
    assert CM.canonical_genotype("nonreference") is None and CM.canonical_genotype("") is None
    roles = tmp_path / "roles.tsv"
    head = "dataset\trun\tspecies\tgenotype_stratum\tinstrument\tdata_type\texpected_files\texpected_reads\trole\n"
    roles.write_text(head + "training/ont/Gmax/scn_roots_PRJNA803218\tSRR17876321\tGmax\tnon_reference\tPromethION\tRNA-Seq\t1\t1"
                     "\tc2_training_eligible\n"
                     "training/ont/Gmax/x\tSRR2\tGmax\tnon-ref\tPromethION\tRNA-Seq\t1\t1\tc2_training_eligible\n")
    row = CM.role_row(roles, "training/ont/Gmax/scn_roots_PRJNA803218", "SRR17876321")
    assert row["genotype_stratum"] == "known_nonreference" and row["run"] == "SRR17876321"
    with pytest.raises(CM.MetadataError, match="genotype_stratum 'non-ref' is not one of"):
        CM.role_row(roles, "training/ont/Gmax/x", "SRR2")
    # end to end: the sidecar carries the canonical token and the builder's CHECK accepts it
    fq = tmp_path / "SRR17876321.fastq"
    fq.write_text("@r1 x\nACGT\n+\nIIII\n")
    assert CM.main(["--source", "ont", "--species", "Gmax", "--roles", str(roles), "--dataset",
                    "training/ont/Gmax/scn_roots_PRJNA803218", "--run", "SRR17876321", "--library", "SRX14100000",
                    "--reads", str(fq), "--protocol", "cDNA", "--umi-status", "absent",
                    "--out", str(tmp_path / "c0_metadata.tsv")]) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["genotype_stratum"] == "known_nonreference"
    from transgenic.evidence import c0_schema as S
    assert rows[0]["genotype_stratum"] in S.GENOTYPE_STRATA


def _ont(tmp_path, reads, source="ont", run="SRR1", extra=(), library="SRX1000001"):
    fq = tmp_path / f"{run}.fastq"
    fq.write_text("".join(f"@{r} extra\nACGT\n+\nIIII\n" for r in reads))
    lr, _ = W.write_roles(tmp_path, extra_rows=[f"pacbio/{W.SPECIES}_x_PRJNA5\tPB1\t{W.SPECIES}\treference\tSequel\tRNA-Seq\t1\t2"
                                                "\t\t\t\tb1_validation_only\tt\t"])
    ds = W.ONT_DATASET if source == "ont" else f"pacbio/{W.SPECIES}_x_PRJNA5"
    return ["--source", source, "--species", W.SPECIES, "--roles", str(lr), "--dataset", ds, "--run", run, "--reads", str(fq),
            *(["--library", library] if library is not None else []), "--out", str(tmp_path / "c0_metadata.tsv"), *extra]


def test_pacbio_source_molecule_from_zmw_names_and_sra_names_refused(tmp_path, capsys):
    assert CM.main(_ont(tmp_path, ["m64011_190830_220126/4194370/ccs", "m64011_190830_220126/4194371/ccs/fwd"],
                        source="pacbio", run="PB1")) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["library"] == "SRX1000001" and rows[0]["bioproject"] == "PRJNA5"
    assert [r["source_molecule"] for r in rows[1:]] == ["m64011_190830_220126/4194370", "m64011_190830_220126/4194371"]
    (tmp_path / "c0_metadata.tsv").unlink()
    (tmp_path / "c0_metadata.tsv.sha256").unlink()
    (tmp_path / "c0_metadata.tsv.PROVENANCE.json").unlink()
    assert CM.main(_ont(tmp_path, ["SRR1.1"], source="pacbio", run="PB1")) == 1
    assert "source ZMW molecule" in capsys.readouterr().err


def _ont_world(world, tmp_path, protocol):
    W.write_ont(tmp_path)
    d = tmp_path / "align" / W.SPECIES / "proj1" / "SRR1"
    for p in ("c0_metadata.tsv", "c0_metadata.tsv.sha256"):
        (d / p).unlink()
    args = _ont(tmp_path, [r for r, _ in W.ONT_READS], extra=("--protocol", protocol, "--umi-status", "absent",
                                                              "--bioproject", "PRJ2"))
    args[args.index("--out") + 1] = str(d / "c0_metadata.tsv")
    assert CM.main(args) == 0
    cli = W.cli_args(world, tmp_path, source="ont", out_name=f"ont_{protocol}.db")
    return B.main(cli), cli


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build_corpus(tmp_path_factory.mktemp("meta"))


def test_ont_metadata_feeds_the_builder_and_unresolved_protocol_fails_closed(world, tmp_path, capsys):
    rc, cli = _ont_world(world, tmp_path / "ok", "cDNA")
    assert rc == 0
    con = duckdb.connect(cli[cli.index("--out") + 1], read_only=True)
    assert con.execute("SELECT DISTINCT library, unit_rule FROM molecule_member WHERE read_name = 'o1'").fetchall() == \
        [("SRX1000001", "pcr_equivalence")]                             # the --library, not the run
    con.close()
    rc, _ = _ont_world(world, tmp_path / "unresolved", "unresolved")
    assert rc == 1 and "unresolved ONT protocol/UMI status" in capsys.readouterr().err


def _pacbio_fasta(tmp_path, headers, run="SRR35145693", library="SRX29587349"):
    fa = tmp_path / f"{run}.fasta"
    fa.write_text("".join(f">{h}\nACGT\n" for h in headers))
    lr, _ = W.write_roles(tmp_path, extra_rows=[f"pacbio/{W.SPECIES}_x_PRJNA5\t{run}\t{W.SPECIES}\treference\tSequel\tRNA-Seq"
                                                "\t1\t2\t\t\t\tb1_validation_only\tt\t"])
    return ["--source", "pacbio", "--species", W.SPECIES, "--roles", str(lr), "--dataset", f"pacbio/{W.SPECIES}_x_PRJNA5",
            "--run", run, *(["--library", library] if library is not None else []), "--reads", str(fa),
            "--out", str(tmp_path / "c0_metadata.tsv")]


def test_pacbio_sra_renamed_reads_take_the_zmw_from_the_second_header_token(tmp_path):
    # Real header shape (SRR35145693): QNAME = first token; the movie/zmw is the second, with a /ccs/1 suffix.
    headers = ["SRR35145693.1 m54283U_201030_021721/11/ccs/1", "SRR35145693.2 m54283U_201030_021721/12/ccs",
               "SRR35145693.3 m54283U_201030_021721/11/ccs/2"]
    assert CM.main(_pacbio_fasta(tmp_path, headers)) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["library"] == "SRX29587349"                       # the --library (experiment), never the run
    assert [(r["read"], r["source_molecule"]) for r in rows[1:]] == [
        ("SRR35145693.1", "m54283U_201030_021721/11"), ("SRR35145693.2", "m54283U_201030_021721/12"),
        ("SRR35145693.3", "m54283U_201030_021721/11")]


def test_longread_library_is_explicit_and_never_the_run(tmp_path, capsys):
    """Fix round 6, finding 1.  A11 (protocol line 558): "runs ... do not create libraries".  The long-read producer
    takes the biological library per run (--library = the SRA/ENA experiment accession or an author key), writes it
    to the '*' row and to PROVENANCE.json, keeps the run separately, and fails closed without it, when it is
    'unknown', or when it merely repeats the run accession (before: library = run, silently)."""
    headers = ["SRR35145693.1 m54283U_201030_021721/11/ccs/1"]
    assert CM.main(_pacbio_fasta(tmp_path, headers, library=None)) == 1
    assert "--library is required" in capsys.readouterr().err
    assert not (tmp_path / "c0_metadata.tsv").exists() and not (tmp_path / "c0_metadata.tsv.PROVENANCE.json").exists()
    assert CM.main(_pacbio_fasta(tmp_path, headers, library="SRR35145693")) == 1
    assert "--library equals the run accession" in capsys.readouterr().err
    assert CM.main(_pacbio_fasta(tmp_path, headers, library="unknown")) == 1
    assert "not a library identity" in capsys.readouterr().err
    assert CM.main(_pacbio_fasta(tmp_path, headers, library=" ")) == 1
    assert "--library is required" in capsys.readouterr().err
    assert CM.main(_pacbio_fasta(tmp_path, headers, library="SRX29587349")) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["library"] == "SRX29587349" and rows[1]["read"] == "SRR35145693.1"
    prov = json.loads((tmp_path / "c0_metadata.tsv.PROVENANCE.json").read_text())
    assert prov["library"] == "SRX29587349" and prov["run"] == "SRR35145693"
    # ONT takes the same input; an author key (not an SRX) is accepted verbatim
    fq = tmp_path / "SRR1.fastq"
    fq.write_text("@r1 x\nACGT\n+\nIIII\n")
    lr, _ = W.write_roles(tmp_path)
    (tmp_path / "c0_metadata.tsv").unlink()
    for p in ("c0_metadata.tsv.sha256", "c0_metadata.tsv.PROVENANCE.json"):
        (tmp_path / p).unlink()
    args = ["--source", "ont", "--species", W.SPECIES, "--roles", str(lr), "--dataset", W.ONT_DATASET, "--run", "SRR1",
            "--reads", str(fq), "--protocol", "cDNA", "--umi-status", "absent", "--bioproject", "PRJ2",
            "--out", str(tmp_path / "c0_metadata.tsv")]
    assert CM.main(args) == 1 and "--library is required" in capsys.readouterr().err
    assert CM.main(args + ["--library", "author key: Col-0 leaf prep 2024-03"]) == 1        # free text: refused
    assert "neither an ENA/SRA experiment accession" in capsys.readouterr().err
    assert CM.main(args + ["--library", "SRX_LIB1"]) == 1                                    # not <digits>
    capsys.readouterr()
    assert CM.main(args + ["--library", "SAMN49012086:HOTXU"]) == 0                          # A49.1 sample:library_name
    assert _read(tmp_path / "c0_metadata.tsv")[0]["library"] == "SAMN49012086:HOTXU"


@pytest.mark.parametrize("header,message", [
    ("SRR35145693.1", "no second header token"),
    ("SRR35145693.1 m54283U_201030_021721/11", "is not '<movie>/<zmw>/ccs"),
    ("SRR35145693.1 m54283U_201030_021721/x/ccs/1", "is not '<movie>/<zmw>/ccs"),
    ("SRR35145693.1 m54283U_201030_021721/11/ccs/fwd", "is not '<movie>/<zmw>/ccs"),
    ("SRR35145693.1 length=1234", "is not '<movie>/<zmw>/ccs"),
])
def test_pacbio_missing_or_malformed_zmw_token_fails_closed(tmp_path, capsys, header, message):
    assert CM.main(_pacbio_fasta(tmp_path, [header])) == 1
    err = capsys.readouterr().err
    assert message in err and "A16" in err and not (tmp_path / "c0_metadata.tsv").exists()


# ------------------------------------------------------------------ EST: NCBI esummary v2.0 TSV (--est-meta)
ESUMMARY = ("accession_version\tclone_lib\tclone\tbiosample\n"
            "CF404621.1\tSAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library\tVVL001A01\tSAMN00739408\n"
            "CF404622.1\tSAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library\t\tSAMN00739408\n")


def _est_meta(tmp_path, meta=ESUMMARY, screened=None, sidecar=True):
    p = tmp_path / "est_meta.tsv"
    p.write_text(meta)
    side = tmp_path / "est_meta.tsv.sha256"
    if sidecar:
        side.write_text(f"{W.sha(p)}  est_meta.tsv\n")
    elif side.exists():
        side.unlink()                     # an earlier call in the same test may have written it
    screened = screened or (
        ">CF404621.1 PN40024 leaves cDNA clone VVL001A01 5', mRNA sequence univec=keep:1-100/100\n" + "A" * 100 + "\n"
        ">CF404622.1_part1 PN40024 leaves cDNA clone VVL001B02 5', mRNA sequence univec=split:1-60/140\n" + "C" * 60 + "\n")
    (tmp_path / "est.univec.fa").write_text(screened)
    _, est_roles = W.write_roles(tmp_path)
    return ["--source", "est", "--species", W.SPECIES, "--roles", str(est_roles), "--screened", str(tmp_path / "est.univec.fa"),
            "--est-meta", str(p), "--out", str(tmp_path / "c0_metadata.tsv")]


def test_est_meta_supplies_full_clone_lib_as_library_and_optional_clone(tmp_path):
    assert CM.main(_est_meta(tmp_path) + ["--bioproject", "PRJNA7"]) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    assert rows[0]["bioproject"] == "PRJNA7"
    lib = "SAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library"
    assert [(r["read"], r["library"], r["clone"], r["bioproject"]) for r in rows[1:]] == [
        ("CF404621.1", lib, "VVL001A01", ""), ("CF404622.1_part1", lib, "", "")]   # header clone never inferred
    prov = __import__("json").loads((tmp_path / "c0_metadata.tsv.PROVENANCE.json").read_text())
    assert prov["est_meta_sha256"] == W.sha(tmp_path / "est_meta.tsv")


@pytest.mark.parametrize('display_clone', ['AGTT_Gm-r1030-2842', 'AGTT>Gm-r1030-2842'])
def test_ncbi_fasta_clone_display_keeps_authoritative_clone(tmp_path, display_clone):
    clone = 'AGTT>Gm-r1030-2842'
    meta = ('accession_version\tclone_lib\tclone\tbiosample\n'
            'DN591027.1\tSAMN00158301 Gm-r1030\t' + clone + '\tSAMN00158301\n')
    screened = '>DN591027.1 cDNA clone ' + display_clone + " 5', mRNA sequence univec=keep:1-100/100\n" + 'A' * 100 + '\n'
    assert CM.main(_est_meta(tmp_path, meta=meta, screened=screened) + ['--bioproject', 'NCBI_EST:txid3847']) == 0
    row = _read(tmp_path / 'c0_metadata.tsv')[1]
    assert row['clone'] == clone  # A11 molecule identity must never use the display-normalized value.
    assert row['library'] == 'SAMN00158301 Gm-r1030'


@pytest.mark.parametrize('display_clone', ['AGTT_Gm-r1030-28420', 'AGTT_Gm-r1030-284X', 'AGTTGm-r1030-2842'])
def test_ncbi_clone_display_still_refuses_wrong_identity(display_clone):
    with pytest.raises(CM.MetadataError, match='disagrees'):
        CM.check_header_clone('DN591027.1', 'cDNA clone ' + display_clone + ' 5-prime', 'AGTT>Gm-r1030-2842')


def test_missing_library_fills_only_records_without_clone_lib(tmp_path, capsys):
    # author decision 2026-09-25: the Vvinifera records with no /clone_lib take the library 'NCBI-EST'
    no_lib = ESUMMARY.replace("CF404622.1\tSAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library\t",
                              "CF404622.1\t\t")
    assert CM.main(_est_meta(tmp_path, meta=no_lib) + ["--bioproject", "P", "--missing-library", "NCBI-EST"]) == 0
    rows = _read(tmp_path / "c0_metadata.tsv")
    lib = "SAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library"
    assert [(r["read"], r["library"]) for r in rows[1:]] == [("CF404621.1", lib), ("CF404622.1_part1", "NCBI-EST")]
    prov = __import__("json").loads((tmp_path / "c0_metadata.tsv.PROVENANCE.json").read_text())
    assert prov["missing_library"] == "NCBI-EST" and prov["missing_library_records"] == 1


def test_missing_library_refuses_blank_or_unknown_value(tmp_path, capsys):
    for bad in (" ", "unknown"):
        assert CM.main(_est_meta(tmp_path) + ["--bioproject", "P", "--missing-library", bad]) == 1
        assert "--missing-library" in capsys.readouterr().err


def test_est_meta_fails_closed(tmp_path, capsys):
    assert CM.main(_est_meta(tmp_path)) == 1                                       # no BioProject in esummary
    assert "--est-meta needs --bioproject" in capsys.readouterr().err
    assert CM.main(_est_meta(tmp_path, sidecar=False) + ["--bioproject", "P"]) == 1
    assert "sha256 sidecar" in capsys.readouterr().err
    tampered = _est_meta(tmp_path) + ["--bioproject", "P"]
    (tmp_path / "est_meta.tsv").write_text(ESUMMARY.replace("VVL001A01", "VVL001A09"))
    assert CM.main(tampered) == 1 and "sidecar" in capsys.readouterr().err
    assert CM.main(_est_meta(tmp_path, meta=ESUMMARY.replace("VVL001A01", "VVL009Z99")) + ["--bioproject", "P"]) == 1
    assert "disagrees with the recorded clone" in capsys.readouterr().err       # header 'cDNA clone' cross-check
    no_lib = ESUMMARY.replace("\tSAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves cDNA library\t\t", "\t\t\t")
    assert CM.main(_est_meta(tmp_path, meta=no_lib) + ["--bioproject", "P"]) == 1
    assert "no /clone_lib" in capsys.readouterr().err
    assert CM.main(_est_meta(tmp_path, meta=ESUMMARY.replace("\tbiosample", "")) + ["--bioproject", "P"]) == 1
    assert "lacks column" in capsys.readouterr().err
    both = _est_meta(tmp_path) + ["--bioproject", "P", "--genbank", str(tmp_path / "est_meta.tsv")]
    assert CM.main(both) == 1 and "exactly one of --genbank / --est-meta" in capsys.readouterr().err
