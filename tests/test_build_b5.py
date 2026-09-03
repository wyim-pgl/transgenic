"""Integration tests for the B5 builder (spec §6–§8) and the CLI wiring of #12."""
import ast
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
duckdb = pytest.importorskip("duckdb")


def _load(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def b5():
    return _load(ROOT / "src" / "transgenic" / "datasets" / "build_b5.py", "build_b5")


def _write_inputs(tmp_path, gsf):
    rng = random.Random(7)
    chrom = {"Chr1": "".join(rng.choice("ACGT") for _ in range(30000)), "2": "".join(rng.choice("ACGT") for _ in range(20000))}
    fasta = tmp_path / "g.fa"
    fasta.write_text("".join(f">{k}\n{v}\n" for k, v in chrom.items()))
    genes = {
        "g1": ("Chr1", "+", {"g1.1": [("five_prime_UTR", 1001, 1050, "."), ("CDS", 1051, 1350, "0"), ("CDS", 1501, 1700, "0"), ("three_prime_UTR", 1701, 1800, ".")],
                            "g1.2": [("CDS", 1051, 1350, "0"), ("CDS", 1601, 1700, "0")]}),
        "g2": ("Chr1", "-", {"g2.1": [("CDS", 5001, 5300, "0")]}),
        "gbig": ("2", "+", {"gbig.1": [("CDS", 100 + 10 * i, 105 + 10 * i, "0") for i in range(151)]}),  # 151 CDS -> rejected
        "glast": ("2", "-", {"glast.1": [("CDS", 3001, 3300, "0"), ("CDS", 3401, 3700, "0")]}),  # last gene at EOF
    }
    lines = []
    for gid, (chrom_name, strand, txs) in genes.items():
        allf = [f for fs in txs.values() for f in fs]
        gs, ge = min(f[1] for f in allf), max(f[2] for f in allf)
        lines.append(f"{chrom_name}\tt\tgene\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid};Name={gid}")
        for tid, fs in txs.items():
            lines.append(f"{chrom_name}\tt\tmRNA\t{gs}\t{ge}\t.\t{strand}\t.\tID={tid};Parent={gid}")
            for k, (t, s, e, ph) in enumerate(fs):
                lines.append(f"{chrom_name}\tt\t{t}\t{s}\t{e}\t.\t{strand}\t{ph}\tID={tid}.{k};Parent={tid}")
    gff = tmp_path / "g.gff3"
    gff.write_text("##gff-version 3\n" + "\n".join(lines) + "\n")
    split = tmp_path / "split.tsv"
    split.write_text("species_id\tgene_id\torthogroup_id\tsplit\tstrict_holdout\tseed\tsource_version\n"
                     "Athaliana\tg1\tOG1\ttrain\tfalse\t123\tv1\nAthaliana\tg2\tOG2\tvalid\tfalse\t123\tv1\n"
                     "Athaliana\tgbig\tOG3\ttrain\tfalse\t123\tv1\nAthaliana\tglast\tOG4\ttest\ttrue\t123\tv1\n")
    manifest = tmp_path / "species.tsv"
    manifest.write_text(f"species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\nAthaliana\tA\tTAIR10\t{fasta}\t\t{gff}\t\t\n")
    return fasta, gff, split, manifest


def test_build_validate_end_to_end(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    db = tmp_path / "b5.db"
    res = b5.build_b5_database(str(db), str(manifest), str(split), rc="isoform-only", add_extra=0, verify_md5=False)
    r = res[0]
    assert r["rows"] == 4 and r["rc_rows"] == 1  # g1(+rc), g2, glast; gbig rejected
    assert r["rejected"][0]["gene_id"] == "gbig" and "CDS" in r["rejected"][0]["reason"]
    con = duckdb.connect(str(db), read_only=True)
    rows = {row[0]: row for row in con.sql("SELECT geneModel, split, is_rc, predict, gff, gsf_token_count, fin - start, strict_holdout FROM geneList").fetchall()} if False else \
           {row[0]: row for row in con.sql("SELECT geneModel, split, is_rc, gff, gsf_token_count, fin - start, strict_holdout, sequence FROM geneList").fetchall()}
    assert "glast" in rows, "last gene of the file must be flushed"
    assert rows["g1-rc"][1] == "train" and rows["g1-rc"][2] is True and rows["g1"][2] is False
    assert all(row[5] % 6144 == 0 for row in rows.values())
    assert rows["glast"][6] is True and rows["glast"][1] == "test"
    assert all(len(row[7]) == row[5] for row in rows.values())
    assert con.sql("SELECT count(*) FROM build_manifest").fetchone()[0] == 1
    assert con.sql("SELECT count(*) FROM gene_split").fetchone()[0] == 4
    assert con.sql("SELECT count(*) FROM gene_key_map").fetchone()[0] == 4 and con.sql("SELECT gene_id_original FROM geneList WHERE geneModel='g1'").fetchone()[0] == "g1"
    con.close()
    report = b5.validate_b5_database(str(db))
    assert report["ok"], report["violations"]


def test_predict_mode_stores_null_labels(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    con = duckdb.connect(str(tmp_path / "p.db"))
    rows, _ = b5.read_split_table(str(split))
    b5.build_species(con, "Athaliana", str(fasta), str(gff), rows, "x", rc="all", mode="predict")
    assert con.sql("SELECT count(*) FROM geneList WHERE gff IS NULL").fetchone()[0] == con.sql("SELECT count(*) FROM geneList").fetchone()[0]
    assert con.sql("SELECT count(*) FROM geneList WHERE is_rc").fetchone()[0] == 0  # no RC without labels
    con.close()


def test_missing_split_fails_closed_unless_legacy(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    con = duckdb.connect(str(tmp_path / "m.db"))
    with pytest.raises(b5.gc.SplitError):
        b5.build_species(con, "Athaliana", str(fasta), str(gff), {}, "x")
    res = b5.build_species(con, "Athaliana", str(fasta), str(gff), {}, "x", allow_missing_split=True)
    assert res["rows"] >= 3 and con.sql("SELECT count(*) FROM geneList WHERE split IS NULL").fetchone()[0] == res["rows"]
    con.close()


def test_excluded_species_refused(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    manifest.write_text(manifest.read_text().replace("Athaliana", "Zmays"))
    with pytest.raises(ValueError):
        b5.build_b5_database(str(tmp_path / "z.db"), str(manifest), str(split), verify_md5=False)


def test_validate_detects_maize_rows_and_split_violations(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    db = tmp_path / "v.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="none", verify_md5=False)
    con = duckdb.connect(str(db))
    con.execute("UPDATE geneList SET split = 'valid' WHERE geneModel = 'glast'")  # strict held-out moved out of test
    con.execute("INSERT INTO geneList (rn, geneModel, species_id, gene_id, orthogroup_id, split, is_rc, strict_holdout, gff) "
                "VALUES (999, 'GRMZM2G0001', 'Zmays', 'GRMZM2G0001', 'OG9', 'train', false, false, NULL)")
    con.close()
    report = b5.validate_b5_database(str(db))
    joined = "\n".join(report["violations"]).lower()
    assert "strict" in joined and "zmays" in joined and "grmzm" in joined


def test_validate_cds_clean_filter(b5):
    seq = "ATG" + "GCT" * 5 + "TAA"
    assert b5.validate_cds(f"0|CDS1|{len(seq)}|+|A>CDS1", seq) == (True, "")
    assert b5.validate_cds(f"0|CDS1|{len(seq)-1}|+|A>CDS1", seq[:-1])[0] is False
    rc = b5.revcomp(seq)
    assert b5.validate_cds(f"0|CDS1|{len(rc)}|-|A>CDS1", rc) == (True, "")


def test_gff2gsf_cli_uses_contract(tmp_path):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gff2gsf.py"), str(gff)], capture_output=True, text=True, check=True)
    lines = dict(l.split("\t") for l in out.stdout.strip().splitlines())
    assert set(lines) == {"g1", "g2", "glast"} and "skip gbig" in out.stderr
    assert lines["g2"] == "0|CDS1|300|-|A>CDS1"  # relative to gene start, 0-based half-open
    assert lines["g1"].split(">")[1].count(";") == 1


def test_create_database_rc_mode_resolution():
    src = (ROOT / "scripts" / "create_database.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_rc_mode")
    ns = {"sys": sys, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "cd", "exec"), ns)
    resolve = ns["resolve_rc_mode"]
    A = lambda **k: types.SimpleNamespace(**{**dict(rc=None, add_rc=False, add_rc_iso_only=False), **k})
    assert resolve(A()) == "none"
    assert resolve(A(rc="isoform-only")) == "isoform-only"
    assert resolve(A(add_rc_iso_only=True)) == "isoform-only"  # no longer a silent no-op
    assert resolve(A(add_rc=True)) == "all"
    with pytest.raises(SystemExit):
        resolve(A(rc="all", add_rc=True))


def test_qc_flags_mask_and_drop_transcripts(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    flags = tmp_path / "flags.tsv"
    flags.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                     "Athaliana\tg1\tg1.2\tgeenuff_error_missing_stop_codon\t0\t0\n"   # one of two transcripts -> dropped
                     "Athaliana\tg2\t\tempty_super_locus\t0\t0\n"                      # gene-level hard -> masked
                     "Athaliana\tglast\tglast.1\tmissing_utr_5p\t0\t0\n")                # soft -> untouched
    db = tmp_path / "q.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="isoform-only", verify_md5=False, qc_flags_path=str(flags))
    con = duckdb.connect(str(db), read_only=True)
    rows = {r[0]: r for r in con.sql("SELECT geneModel, train_weight, qc_flags, n_transcripts, gff FROM geneList").fetchall()}
    assert rows["g2"][1] == 0.0 and "empty_super_locus" in rows["g2"][2]
    assert rows["g1"][1] == 1.0 and rows["g1"][3] == 1 and "transcripts_dropped" in rows["g1"][2] and ";" not in rows["g1"][4].split(">")[1]
    assert "g1-rc" not in rows  # isoform-only RC no longer applies once the second transcript is dropped
    assert rows["glast"][1] == 1.0 and rows["glast"][2] is None
    con.close()
    rep = b5.validate_b5_database(str(db))
    assert rep["ok"] and rep["rows_loss_masked"] == 1


def test_short_contig_keeps_window_multiple_by_n_padding(tmp_path, b5):
    fasta = tmp_path / "s.fa"; fasta.write_text(">ctg\n" + "ACGT" * 500 + "\n")   # 2,000-nt contig
    gff = tmp_path / "s.gff3"
    gff.write_text("ctg\tt\tgene\t101\t400\t.\t+\t.\tID=gs\nctg\tt\tmRNA\t101\t400\t.\t+\t.\tID=gs.1;Parent=gs\nctg\tt\tCDS\t101\t400\t.\t+\t0\tID=c;Parent=gs.1\n")
    con = duckdb.connect(str(tmp_path / "s.db"))
    b5.build_species(con, "Athaliana", str(fasta), str(gff), {("Athaliana", "gs"): {"split": "train"}}, "x")
    st, fn, seq, cb = con.sql("SELECT start, fin, sequence, contig_boundary FROM geneList").fetchone()
    assert fn - st == 6144 and len(seq) == 6144 and seq.endswith("N" * 100) and cb is True
    con.close()


def test_validator_catches_corrupted_token_count_and_sequence(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    db = tmp_path / "c.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="isoform-only", verify_md5=False)
    assert b5.validate_b5_database(str(db))["ok"]
    con = duckdb.connect(str(db))
    con.execute("UPDATE geneList SET gsf_token_count = gsf_token_count + 1 WHERE geneModel = 'g2'")
    con.execute("UPDATE geneList SET sequence = substr(sequence, 1, 100) WHERE geneModel = 'glast'")
    con.execute("UPDATE geneList SET gff = (SELECT gff FROM geneList WHERE geneModel = 'g1') WHERE geneModel = 'g1-rc'")
    con.close()
    v = "\n".join(b5.validate_b5_database(str(db))["violations"])
    assert "token count" in v and "sequence length" in v and "reverse complement" in v


def test_tiered_policy_build_and_validate(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    db = tmp_path / "t2.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="none", verify_md5=False, window_policy="tier6144-v2", tier_up_prob=0.0)
    con = duckdb.connect(str(db), read_only=True)
    rows = con.sql("SELECT geneModel, fin - start, window_policy, length(sequence) FROM geneList").fetchall()
    assert all(r[1] == 30720 and r[2] == "tier6144-v2" and r[3] == 30720 for r in rows)   # contigs are 20-30 kb -> N-padded
    con.close()
    assert b5.validate_b5_database(str(db))["ok"]


def test_tile_policy_v3_build(tmp_path, b5):
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    flags = tmp_path / "flags.tsv"
    flags.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\nAthaliana\tg2\t\tempty_super_locus\t0\t0\n")
    db = tmp_path / "v3.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="all", verify_md5=False, window_policy="tile6144-v3", qc_flags_path=str(flags))
    con = duckdb.connect(str(db), read_only=True)
    rows = con.sql("SELECT geneModel, split, train_weight, gff, qc_flags, window_policy, fin - start FROM geneList WHERE NOT is_rc").fetchall()
    assert rows and all(r[5] == "tile6144-v3" and r[6] in (30720, 61440, 129024) for r in rows)
    labels = {r[0]: r[3] for r in rows}
    # block splits (A29): Chr1 is one block; its split is drawn from the seeded rng. g2 (valid orthogroup) is leak-masked
    # in a train tile (N in the sequence, absent from the label) or labelled in a valid/test tile.
    chr1 = [r for r in rows if ":Chr1:" in r[0]]
    assert chr1
    for r in chr1:
        if r[1] == "train":
            assert r[3].count(b5.gc.GENE_SEP) == 0 and "leak_masked=1" in (r[4] or "") and r[2] == 1.0
        else:
            # A32: g2 is hard-flagged -> masked at gene level (N in the sequence, absent from the label); the tile keeps weight 1
            assert r[3].count(b5.gc.GENE_SEP) == 0 and r[2] == 1.0 and "hard_masked=1" in (r[4] or "")
    # contig '2' (A31): glast is strict held-out but its block is drawn like any other; in a train/valid tile glast is
    # N-masked and unlabelled (leak rule), in a test tile it is labelled. gbig rejected at gene level (151 CDS).
    c2 = [r for r in rows if r[0].startswith("Athaliana:2:")]
    assert c2 and all("CDS150" not in r[3] for r in c2)
    for r in c2:
        if r[1] == "test":
            assert r[3] != b5.gc.EMPTY_TOKEN if hasattr(b5.gc, "EMPTY_TOKEN") else True
            assert "leak_masked" not in (r[4] or "")
        else:
            assert "leak_masked=1" in (r[4] or "")
    src = (ROOT / "src" / "transgenic" / "datasets" / "build_b5.py").read_text()
    assert "forced_test=forced" not in src                                   # A31: no block forcing in the builder
    assert con.sql("SELECT count(*) FROM tile_blocks").fetchone()[0] >= 2
    assert con.sql("SELECT count(*) FROM window_genes").fetchone()[0] > 0
    assert con.sql("SELECT count(*) FROM rejected_records WHERE gene_id = 'gbig'").fetchone()[0] == 1
    con.close()
    rep = b5.validate_b5_database(str(db))
    assert rep["ok"], rep["violations"]


def test_qc_flags_accepts_several_files_and_swissprot_caution_masks(tmp_path, b5):
    """Protocol A30: a second A22-schema flag file (Swiss-Prot sequence-caution audit) is merged with the GeenuFF file;
    swissprot_caution_* is hard (row masked), swissprot_note_* is soft (recorded only)."""
    fasta, gff, split, manifest = _write_inputs(tmp_path, None)
    geenuff = tmp_path / "geenuff.tsv"
    geenuff.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                       "Athaliana\tglast\tglast.1\tmissing_utr_5p\t0\t0\n")
    swiss = tmp_path / "swiss.tsv"
    swiss.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                     "Athaliana\tg2\t\tswissprot_caution_erroneous_initiation\t0\t0\n"
                     "Athaliana\tg1\t\tswissprot_note_caution_resolved_in_reference\t0\t0\n")
    merged = b5.read_qc_flags([str(geenuff), str(swiss)])
    assert merged[("Athaliana", "g2")]["*"] == {"swissprot_caution_erroneous_initiation"}
    assert merged[("Athaliana", "glast")]["glast.1"] == {"missing_utr_5p"}
    assert b5.read_qc_flags(str(swiss)) == b5.read_qc_flags([str(swiss)])          # a single path still works
    db = tmp_path / "q2.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="none", verify_md5=False, qc_flags_path=[str(geenuff), str(swiss)])
    con = duckdb.connect(str(db), read_only=True)
    rows = {r[0]: r for r in con.sql("SELECT geneModel, train_weight, qc_flags FROM geneList").fetchall()}
    con.close()
    assert rows["g2"][1] == 0.0 and "swissprot_caution_erroneous_initiation" in rows["g2"][2]
    assert rows["g1"][1] == 1.0 and rows["g1"][2] is None                            # soft note changes nothing
    assert rows["glast"][1] == 1.0
    assert b5.validate_b5_database(str(db))["rows_loss_masked"] == 1
    # CLI accepts several files
    src = (ROOT / "scripts" / "build_b5_database.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             and n.args and getattr(n.args[0], "value", "") == "--qc-flags"]
    assert calls and any(k.arg == "nargs" for k in calls[0].keywords)


def test_qc_flags_resolve_original_gff_ids_and_names(tmp_path, b5):
    """Flag files (GeenuFF #46, Swiss-Prot A30) carry the GFF ID or Name attribute; the builder resolves them to its own
    gene keys (gene_key: a generated code when the ID is longer than 10 characters or has two dots, e.g. TAIR10 ids)."""
    rng = random.Random(3)
    seq = "".join(rng.choice("ACGT") for _ in range(12000))
    fasta = tmp_path / "k.fa"; fasta.write_text(f">Chr1\n{seq}\n")
    gff = tmp_path / "k.gff3"
    gff.write_text("##gff-version 3\n"
                   "Chr1\tt\tgene\t1001\t1600\t.\t+\t.\tID=AT1G01010.TAIR10;Name=AT1G01010\n"
                   "Chr1\tt\tmRNA\t1001\t1600\t.\t+\t.\tID=AT1G01010.1.TAIR10;Parent=AT1G01010.TAIR10;Name=AT1G01010.1\n"
                   "Chr1\tt\tCDS\t1001\t1600\t.\t+\t0\tID=c1;Parent=AT1G01010.1.TAIR10\n"
                   "Chr1\tt\tgene\t5001\t5600\t.\t-\t.\tID=AT1G01020.TAIR10;Name=AT1G01020\n"
                   "Chr1\tt\tmRNA\t5001\t5600\t.\t-\t.\tID=AT1G01020.1.TAIR10;Parent=AT1G01020.TAIR10;Name=AT1G01020.1\n"
                   "Chr1\tt\tCDS\t5001\t5600\t.\t-\t0\tID=c2;Parent=AT1G01020.1.TAIR10\n"
                   "Chr1\tt\tgene\t8001\t8600\t.\t+\t.\tID=AT1G01030.TAIR10;Name=AT1G01030\n"
                   "Chr1\tt\tmRNA\t8001\t8600\t.\t+\t.\tID=AT1G01030.1.TAIR10;Parent=AT1G01030.TAIR10;Name=AT1G01030.1\n"
                   "Chr1\tt\tCDS\t8001\t8600\t.\t+\t0\tID=c3;Parent=AT1G01030.1.TAIR10\n")
    genes = list(b5.gc.parse_gff3(gff.read_text().splitlines(), species_code=b5.gc.species_code("Athaliana")))
    keys = [g.gene_id for g in genes]
    assert all(k not in ("AT1G01010.TAIR10", "AT1G01010") for k in keys)             # generated keys (16-character ids)
    assert [g.name_original for g in genes] == ["AT1G01010", "AT1G01020", "AT1G01030"]
    split = tmp_path / "split.tsv"
    split.write_text("species_id\tgene_id\torthogroup_id\tsplit\tstrict_holdout\tseed\tsource_version\n"
                     + "".join(f"Athaliana\t{k}\tOG{i}\ttrain\tfalse\t123\tv1\n" for i, k in enumerate(keys)))
    manifest = tmp_path / "species.tsv"
    manifest.write_text(f"species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\nAthaliana\tA\tTAIR10\t{fasta}\t\t{gff}\t\t\n")
    flags = tmp_path / "flags.tsv"
    flags.write_text("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n"
                     "Athaliana\tAT1G01010\t\tswissprot_caution_erroneous_initiation\t0\t0\n"        # Name attribute
                     "Athaliana\tAT1G01020.TAIR10\t\tgeenuff_error_missing_start_codon\t0\t0\n"     # original ID attribute
                     f"Athaliana\t{keys[2]}\t\tempty_super_locus\t0\t0\n")                         # builder key itself
    db = tmp_path / "k.db"
    b5.build_b5_database(str(db), str(manifest), str(split), rc="none", verify_md5=False, qc_flags_path=str(flags))
    con = duckdb.connect(str(db), read_only=True)
    rows = {r[0]: r for r in con.sql("SELECT geneModel, train_weight, qc_flags, gene_id_original FROM geneList").fetchall()}
    keymap = {r[0]: r[1] for r in con.sql("SELECT gene_id, name_original FROM gene_key_map").fetchall()}
    con.close()
    assert set(rows) == set(keys) and keymap[keys[0]] == "AT1G01010"
    assert all(rows[k][1] == 0.0 for k in keys), rows
    assert "swissprot_caution_erroneous_initiation" in rows[keys[0]][2] and "missing_start" in rows[keys[1]][2]
    assert b5.validate_b5_database(str(db))["rows_loss_masked"] == 3


def test_validator_token_cap_follows_window_policy(b5):
    """Smoke build 2026-09-02: 410 tile6144-v3 rows were reported over the cap because the validator used the v2 cap."""
    assert b5.token_cap_for(b5.gc.WINDOW_POLICY_V3) == b5.gc.CAPS_V3["tokens"] == 8192
    assert b5.token_cap_for(b5.gc.WINDOW_POLICY) == b5.gc.CAPS["tokens"] == 2048
    assert b5.token_cap_for("tier6144-v2") == 2048
    src = (ROOT / "src" / "transgenic" / "datasets" / "build_b5.py").read_text()
    assert "CASE WHEN window_policy" in src and "token_cap_for(gc.WINDOW_POLICY_V3)" in src
