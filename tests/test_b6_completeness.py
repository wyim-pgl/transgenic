"""WP2c (plan v4.1 §6): read-end supplement, D12 CAGE peaks and per-population A16 completeness codes.

World: test_b6_c0_world's corpus, EST (no SEQ), its ONT run SRR1 and PacBio run SRR2, plus two ONT runs written here
with real SEQ, poly(A) tails and strand-switch adapters on gB's chain (exons 170001-170200 / 170301-170500):

    SRR8 (library SRX1000108; roles: b6_input)       a1, a2, a3, ad8
    SRR9 (library SRX1000109; adapter 5C approved)   b1, b2, ad9, zp1 (internal priming), mneg (minus strand)

and one CAGE run SRRC1 whose tags pile up at 0-based 170010 (one via the untemplated-G rule).  In train_track the
gB TSS cluster (>= 5 units, >= 2 libraries with >= 2 units, CAGE peak) makes a1 5C and the b1/b2 tails make its TES
cluster qualify (3C); in eval_input only SRR8 is present, so the same alignment is 5I / 3I.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import test_b6_c0_world as W
import b6_build_c0 as B
import b6_merge_evidence_db as M
import b6_build_read_ends as BRE
import b6_build_completeness as BCP
from transgenic.evidence import b6_completeness as CP
from transgenic.evidence import b6_read_ends as RE
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")
ROOT = Path(__file__).resolve().parents[1]
SSP = RE.ADAPTERS["ssp"]
TAIL = "A" * 15
DS = W.ONT_DATASET
GB = [(170001, 170200), (170301, 170500)]
# read -> (run, start1, end1, tail, ssp)
READS = {
    "a1": ("SRR8", 170001, 170535, False, False), "a2": ("SRR8", 170019, 170620, True, False),
    "a3": ("SRR8", 170025, 170600, False, False), "ad8": ("SRR8", 170061, 170650, False, True),
    "b1": ("SRR9", 170007, 170530, True, False), "b2": ("SRR9", 170013, 170542, True, False),
    "ad9": ("SRR9", 170061, 170660, False, True),
}
LIBS = {"SRR8": "SRX1000108", "SRR9": "SRX1000109"}


def _rc(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _line(name, genome, exons, *, flag=0, left="", right=""):
    chrom = genome["Chr1"]
    body = "".join(chrom[s - 1:e] for s, e in exons)
    cig, cs = [f"{len(left)}S"] if left else [], []
    for i, (s, e) in enumerate(exons):
        if i:
            gap = s - exons[i - 1][1] - 1
            cig.append(f"{gap}N")
            cs.append(f"~gt{gap}ag")
        cig.append(f"{e - s + 1}M")
        cs.append(f":{e - s + 1}")
    if right:
        cig.append(f"{len(right)}S")
    tags = [f"cs:Z:{''.join(cs)}", "AS:i:100", "ts:A:+"]
    return "\t".join([name, str(flag), "Chr1", str(exons[0][0]), "60", "".join(cig), "*", "0", "0",
                      left + body + right, "*"] + tags)


def internal_priming_end(genome):
    chrom = genome["Chr1"]
    i = chrom.find("AAAAAA", 13000)
    assert 0 < i < 37000
    return i                                                   # 0-based: the A run starts right after the read


def write_runs(tmp, genome):
    lines = {"SRR8": [], "SRR9": []}
    for name, (run, s, e, tail, ssp) in READS.items():
        exons = [(s, GB[0][1]), (GB[1][0], e)]
        lines[run].append(_line(name, genome, exons, left=(SSP + "GGG") if ssp else "", right=TAIL if tail else ""))
    z = internal_priming_end(genome)
    lines["SRR9"].append(_line("zp1", genome, [(z - 199, z)], right=TAIL))
    lines["SRR9"].append(_line("mneg", genome, [(30501, 30700)], flag=16, left="T" * 15, right=_rc(SSP + "GGG")))
    rows = []
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=yes -N 50 -p 0.5 --cs=long ref.fa reads.fq.gz"
    for run, ls in lines.items():
        d = tmp / "align" / W.SPECIES / "proj1" / run
        f = W._write_sam(d / f"{run}.sam", cl_f, ls)
        W._write_done(d, f, [("uf", "none")])
        W._write_meta(d, dict(library=LIBS[run], bioproject="PRJ2", genotype_stratum="reference", ingestion_qc_pass="true",
                              mapping_ambiguous="false", protocol="cDNA", umi_status="absent"),
                      [dict(read=l.split("\t", 1)[0]) for l in ls])
        ad = tmp / "audit" / W.SPECIES / "proj1" / run
        a = W._write_sam(ad / f"{run}.audit.sam", cl_a, ls)
        W._write_done(ad, a, [("source", "ont"), ("run", run)])
        rows.append(f"{DS}\t{run}\t{W.SPECIES}\treference\tPromethION\tRNA-Seq\t1\t{len(ls)}\t\t\t\tc2_training_eligible\tt\t")
    return rows


def write_cage(tmp, genome):
    chrom = genome["Chr1"]
    g_pos = 170009
    assert chrom[g_pos] != "G"
    recs = [("c%d" % i, 0, 170010, chrom[170010:170040], 60) for i in range(6)]
    recs.append(("cg", 0, g_pos, "G" + chrom[g_pos + 1:g_pos + 30], 60))       # untemplated G -> tag 170010
    recs += [("cq", 0, 170200, chrom[170200:170230], 5)]                        # MAPQ 5: filtered
    recs += [("f%d" % i, 0, 180000, chrom[180000:180030], 60) for i in range(3)]  # 3 tags: no peak
    body = "".join(f"{n}\t{fl}\tChr1\t{p + 1}\t{mq}\t30M\t*\t0\t0\t{s}\t*\n" for n, fl, p, s, mq in recs)
    sam = tmp / "SRRC1.cage.sam"
    sam.write_text("@SQ\tSN:Chr1\tLN:250000\n@SQ\tSN:Chr2\tLN:60000\n" + body)
    sheet = tmp / "CAGE_INPUTS.test.tsv"
    sheet.write_text("species\ttaxid\tbioproject\truns\tstrategy\ttissue\tgenotype_as_stated\tlibraries\trole\tbasis\n"
                     f"{W.SPECIES}\t3702\tPRJT\tSRRC1\tCAGE\tleaf\tCol-0\t1\tprimary\ttest\n")
    Path(str(sheet) + ".sha256").write_text(hashlib.sha256(sheet.read_bytes()).hexdigest() + "  x\n")
    return sam, sheet


def _sidecar(p):
    Path(str(p) + ".sha256").write_text(hashlib.sha256(Path(p).read_bytes()).hexdigest() + f"  {Path(p).name}\n")
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wp2c")
    w = W.build_corpus(tmp)
    W.write_est(tmp)
    W.write_ont(tmp)
    W.write_pacbio(tmp)
    roles = W.write_roles(tmp, extra_rows=[W.PACBIO_ROLE_ROW] + write_runs(tmp, w["genome"]))
    dbs = []
    for source in ("est", "ont", "pacbio"):
        args = W.cli_args(w, tmp, source=source, roles=roles)
        assert B.main(args) == 0, source
        dbs.append(args[args.index("--out") + 1])
    merged = tmp / "evidence_c0.db"
    M.merge(dbs, merged)
    ont = tmp / "ONT_PROTOCOL.test.tsv"
    ont.write_text("dataset\trun\tspecies\tprotocol\tadapter_5c_approved\n"
                   f"{DS}\tSRR1\t{W.SPECIES}\tcDNA\tfalse\n{DS}\tSRR8\t{W.SPECIES}\tcDNA\tfalse\n"
                   f"{DS}\tSRR9\t{W.SPECIES}\tcDNA\ttrue\n")
    pac = tmp / "PACBIO_LIBRARY.test.tsv"
    pac.write_text(f"dataset\trun\tspecies\tlibrary\n{W.PACBIO_DATASET}\t{W.PACBIO_RUN}\t{W.SPECIES}\t{W.PACBIO_LIBRARY}\n")
    b6roles = tmp / "DATASET_ROLES.b6_v1.tsv"
    b6roles.write_text((ROOT / "data" / "manifests" / "DATASET_ROLES.b6_v1.tsv").read_text()
                       + f"{DS}\tSRR8\t{W.SPECIES}\tDATASET_ROLES.longread_v1.tsv\tc2_training_eligible\tT-ONT8\ttrue\tfalse\t"
                         "tracks+chains\ttest\n"
                       + f"{W.PACBIO_DATASET}\t{W.PACBIO_RUN}\t{W.SPECIES}\tDATASET_ROLES.longread_v1.tsv\tb1_validation_only\t"
                         "A-CCS23\tfalse\ttrue\t-\ttest\n")
    _sidecar(b6roles)
    cage_sam, cage_sheet = write_cage(tmp, w["genome"])
    admitted = []
    for pop in CP.POPULATIONS:
        p = tmp / f"admitted.{pop}.tsv"
        chains = ["10200-10299,10500-10599"] + ([] if pop == "eval_input" else ["170200-170299"])
        p.write_text(f"# population={pop}\npopulation\tspecies\tchrom\tstrand\tintrons\n"
                     + "".join(f"{pop}\t{W.SPECIES}\tChr1\t+\t{c}\n" for c in chains))
        admitted.append(_sidecar(p))
    ends = tmp / f"read_ends_b6_v1.{W.SPECIES}.db"
    assert BRE.main(["--merged", str(merged), "--species", W.SPECIES, "--reference", str(w["fasta"]),
                     "--ont-sheet", str(ont), "--pacbio-sheet", str(pac), "--out", str(ends)]) == 0
    return dict(w, tmp=tmp, merged=merged, ont=ont, pac=pac, roles=b6roles, cage_sam=cage_sam, cage_sheet=cage_sheet,
                admitted=admitted, ends=ends)


def ends_row(world, read):
    con = duckdb.connect(str(world["ends"]), read_only=True)
    try:
        cols = list(RE.COLUMN_NAMES)
        r = con.execute(f"SELECT {', '.join(cols)} FROM read_end WHERE read_name = ?", [read]).fetchall()
        return dict(zip(cols, r[0])) if r else None
    finally:
        con.close()


RECIPE = ROOT / "data" / "manifests" / "CAGE_RECIPE.b6_v1.md"


def recipe_sha():
    return (ROOT / "data" / "manifests" / "CAGE_RECIPE.b6_v1.md.sha256").read_text().split()[0]


def _cp_args(world, out, *, cage=True, recipe_sha256=None, admitted=None, sheet=None):
    args = ["--species", W.SPECIES, "--merged", str(world["merged"]), "--read-ends", str(world["ends"]),
            "--roles", str(world["roles"]), "--reference", str(world["fasta"]), "--cage-recipe", str(RECIPE),
            "--cage-recipe-sha256", recipe_sha256 or recipe_sha(), "--cage-inputs", str(sheet or world["cage_sheet"]),
            "--admitted-chains", *[str(p) for p in (admitted or world["admitted"])], "--out", str(out)]
    if cage:
        args += ["--cage", f"SRRC1={world['cage_sam']}"]
    return args


@pytest.fixture(scope="module")
def codes(world):
    out = world["tmp"] / f"completeness_b6_v1.{W.SPECIES}.db"
    before = LK.sha256(world["merged"])
    assert BCP.main(_cp_args(world, out)) == 0
    assert LK.sha256(world["merged"]) == before                # the merged snapshot is never mutated
    return out


def _code(world, codes_db, pop, read):
    aid = ends_row(world, read)["alignment_id"]
    con = duckdb.connect(str(codes_db), read_only=True)
    try:
        r = con.execute("SELECT complete_5p_code, complete_3p_code, read_internal_code, five_basis, three_basis, "
                        "internal_priming_suspect FROM completeness WHERE population = ? AND alignment_id = ?",
                        [pop, aid]).fetchall()
        return r[0] if r else None
    finally:
        con.close()


# ------------------------------------------------------------------------------ step 1: read ends
def test_one_read_end_row_per_c0_accepted_read_alignment(world):
    con = duckdb.connect(str(world["merged"]), read_only=True)
    n = con.execute("SELECT count(*) FROM evidence_alignment WHERE accepted AND source IN ('est', 'ont', 'pacbio')").fetchone()[0]
    con.close()
    con = duckdb.connect(str(world["ends"]), read_only=True)
    assert con.execute("SELECT count(*) FROM read_end").fetchone()[0] == n
    con.close()


def test_poly_a_adapter_and_approval_fields(world):
    a2, ad8, ad9 = ends_row(world, "a2"), ends_row(world, "ad8"), ends_row(world, "ad9")
    assert a2["polyA_untemplated_len"] == 15 and a2["clip3_len"] == 15 and a2["tes0"] == 170619
    assert ad8["ssp_adapter_hit"] and not ad8["adapter_5c_enabled"] and ad8["clip5_len"] == len(SSP) + 3
    assert ad9["ssp_adapter_hit"] and ad9["adapter_5c_enabled"] and ad9["tss0"] == 170060
    assert ends_row(world, "a1")["ssp_adapter_hit"] is False and ends_row(world, "a1")["primer5_hit"] is None


def test_minus_strand_read_is_transcript_oriented(world):
    r = ends_row(world, "mneg")
    down = _rc(world["genome"]["Chr1"][30480:30500])
    assert r["transcript_strand"] == "-" and (r["tss0"], r["tes0"]) == (30699, 30500)
    assert (r["clip5_len"], r["clip3_len"], r["polyA_untemplated_len"]) == (len(SSP) + 3, 15, 15)
    assert r["ssp_adapter_hit"] and r["downstream20_A"] == down.count("A")


def test_internal_priming_window_and_missing_seq(world):
    z = ends_row(world, "zp1")
    assert z["polyA_untemplated_len"] == 15 and z["downstream_max_consecutive_A"] >= 6
    e = ends_row(world, "e2")
    assert e["seq_available"] is False and e["polyA_untemplated_len"] is None and e["clip5_seq"] is None


def test_read_end_validator_rederives_rows_and_catches_tampering(world, tmp_path, capsys):
    assert BRE.main(["--validate", str(world["ends"]), "--reference", str(world["fasta"]), "--fraction", "1.0"]) == 0
    assert json.loads(capsys.readouterr().out)["checked"] > 10
    bad = tmp_path / "bad.db"
    shutil.copyfile(world["ends"], bad)
    con = duckdb.connect(str(bad))
    con.execute("UPDATE read_end SET polyA_untemplated_len = 3 WHERE read_name = 'a2'")
    con.execute("CHECKPOINT")
    con.close()
    _sidecar(bad)
    prov = json.loads(Path(str(world["ends"]) + ".PROVENANCE.json").read_text())
    Path(str(bad) + ".PROVENANCE.json").write_text(json.dumps(dict(prov, output_sha256=LK.sha256(bad))))
    assert BRE.main(["--validate", str(bad), "--reference", str(world["fasta"]), "--fraction", "1.0"]) == 1
    assert "polyA_untemplated_len" in capsys.readouterr().err


def test_read_end_build_fails_closed_on_unbound_inputs(world, tmp_path):
    sheet = tmp_path / "ont.tsv"
    sheet.write_text("dataset\trun\tspecies\tprotocol\n" f"{DS}\tSRR1\t{W.SPECIES}\tcDNA\n")
    with pytest.raises(RE.ReadEndError, match="no row"):
        RE.build(merged=world["merged"], species=W.SPECIES, reference=world["fasta"], ont_sheet=sheet,
                 pacbio_sheet=world["pac"], out=tmp_path / "x.db")
    sam = tmp_path / "SRR8.sam"
    shutil.copyfile(world["tmp"] / "align" / W.SPECIES / "proj1" / "SRR8" / "SRR8.sam", sam)
    with open(sam, "a") as fh:
        fh.write("\n")
    with pytest.raises(RE.ReadEndError, match="frozen alignment"):
        RE.build(merged=world["merged"], species=W.SPECIES, reference=world["fasta"], ont_sheet=world["ont"],
                 pacbio_sheet=world["pac"], out=tmp_path / "y.db", alignment_paths={f"{DS}/SRR8": str(sam)})


# ------------------------------------------------------------------------------ step 2: CAGE
def test_cage_peak_uses_the_recipe_filters_and_the_untemplated_g_rule(codes):
    con = duckdb.connect(str(codes), read_only=True)
    peaks = con.execute("SELECT chrom, strand, peak0, n_tags FROM cage_peak ORDER BY ALL").fetchall()
    con.close()
    assert peaks == [("Chr1", "+", 170010, 7)]                 # 6 + the G-shifted tag; MAPQ-5 and 3-tag pile dropped


def test_recipe_sha_is_mandatory(world, tmp_path):
    assert BCP.main(_cp_args(world, tmp_path / "a.db", recipe_sha256="0" * 64)) == 1
    with pytest.raises(CP.CompletenessError, match="required"):
        CP.load_recipe(RECIPE, "")
    assert BCP.main(_cp_args(world, tmp_path / "b.db", cage=False)) == 1        # a listed CAGE run not supplied


def test_expand_runs():
    assert CP.expand_runs("SRR15601584-SRR15601586") == ["SRR15601584", "SRR15601585", "SRR15601586"]
    assert CP.expand_runs("SRR1,SRR2") == ["SRR1", "SRR2"] and CP.expand_runs("-") == []


# ------------------------------------------------------------------------------ step 3: codes per population
def test_one_alignment_has_population_specific_codes(world, codes):
    assert _code(world, codes, "train_track", "a1")[:5] == ("5C", "3C", "IC", "tss_cluster", "tes_cluster")
    assert _code(world, codes, "eval_input", "a1")[:3] == ("5I", "3I", "IX")


def test_adapter_route_needs_author_approval(world, codes):
    assert _code(world, codes, "train_track", "ad9")[0:4:3] == ("5C", "adapter")
    assert _code(world, codes, "train_track", "ad8")[0] == "5I"


def test_tail_codes_internal_priming_and_a_ccs23(world, codes):
    assert _code(world, codes, "train_track", "a2")[1:5:3] == ("3C", "polya")
    z = _code(world, codes, "train_track", "zp1")
    assert z[1] == "3I" and z[5] is True and z[2] == "IM"
    p1 = _code(world, codes, "score", "p1")
    assert p1[1] == "3I" and p1[4] == "a_ccs23_never_3prime"
    assert _code(world, codes, "train_track", "p1") is None     # b6_score never enters an input population


def test_internal_codes_ic_ip_ix(world, codes):
    assert _code(world, codes, "train_track", "e2")[2] == "IC"        # exact gA chain
    assert _code(world, codes, "train_track", "e3")[2] == "IP"        # first gA intron only
    assert _code(world, codes, "train_track", "o4")[2] == "IC"        # +-2 nt ONT junction corrected onto gB's chain
    assert _code(world, codes, "train_track", "n1")[2] == "IX"   # noncanonical donor
    assert _code(world, codes, "eval_input", "a1")[2] == "IX"         # gB chain not admitted in this population


def test_lookup_requires_a_population(world, codes):
    aid = ends_row(world, "a1")["alignment_id"]
    con = duckdb.connect(str(codes), read_only=True)
    assert CP.code_of(con, population="train_track", species=W.SPECIES, source="ont", alignment_id=aid) == ("5C", "3C", "IC")
    with pytest.raises(CP.CompletenessError, match="population"):
        CP.code_of(con, population=None, species=W.SPECIES, source="ont", alignment_id=aid)
    con.close()


def test_provenance_binds_every_input_under_the_post_c0_policy(world, codes):
    prov = json.loads(Path(str(codes) + ".PROVENANCE.json").read_text())
    assert prov["policy"] == "computed_post_c0_v1"
    inp = prov["inputs"]
    assert inp["merged"]["sha256"] == LK.sha256(world["merged"]) and inp["read_ends"]["sha256"] == LK.sha256(world["ends"])
    assert inp["cage_recipe"]["sha256"] == recipe_sha() and set(inp["admitted_chains"]) == set(CP.POPULATIONS)
    assert inp["cage_alignments"] == {"SRRC1": LK.sha256(world["cage_sam"])}


def test_each_population_needs_its_own_chain_set_and_no_cage_means_5i(world, tmp_path):
    assert BCP.main(_cp_args(world, tmp_path / "a.db", admitted=world["admitted"][:3])) == 1
    empty = tmp_path / "CAGE_INPUTS.none.tsv"
    empty.write_text("species\ttaxid\tbioproject\truns\tstrategy\ttissue\tgenotype_as_stated\tlibraries\trole\tbasis\n"
                     f"{W.SPECIES}\t3702\t-\t-\t-\t-\t-\t0\tnone\tno CAGE\n")
    _sidecar(empty)
    out = tmp_path / "nocage.db"
    assert BCP.main(_cp_args(world, out, cage=False, sheet=empty)) == 0
    assert _code(world, out, "train_track", "a1")[0] == "5I"
    assert _code(world, out, "train_track", "ad9")[0] == "5I"     # the adapter route needs a CAGE TSS within 50 nt


# ------------------------------------------------------------------------------ review 1 M6: empty admitted-chain sets
def _chain_file(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return _sidecar(p)


HEAD = "population\tspecies\tchrom\tstrand\tintrons\n"


def test_an_empty_admitted_chain_set_is_valid_when_its_population_is_declared(tmp_path):
    p = _chain_file(tmp_path, "score.tsv", "# population=score\n" + HEAD)
    pop, sha, chains = CP.read_admitted(p)
    assert (pop, dict(chains)) == ("score", {}) and sha == LK.sha256(p)


def test_admitted_chain_population_must_be_declared_and_match_every_row(tmp_path):
    with pytest.raises(CP.CompletenessError, match="declare"):
        CP.read_admitted(_chain_file(tmp_path, "a.tsv", HEAD + f"score\t{W.SPECIES}\tChr1\t+\t1-2\n"))
    with pytest.raises(CP.CompletenessError, match="population"):
        CP.read_admitted(_chain_file(tmp_path, "b.tsv", "# population=score\n" + HEAD
                                     + f"train_track\t{W.SPECIES}\tChr1\t+\t1-2\n"))
    with pytest.raises(CP.CompletenessError, match="population"):
        CP.read_admitted(_chain_file(tmp_path, "c.tsv", "# population=bogus\n" + HEAD))
