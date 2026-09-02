"""Protocol A30: Swiss-Prot reviewed Viridiplantae as a separately labelled sensitivity/audit set.

Target: revision/scripts/63_swissprot_sensitivity.py (pure Python, loaded by path). The script parses the
UniProtKB/Swiss-Prot flat file (plants taxonomic division), keeps Viridiplantae entries, records PE level,
N-terminal experimental evidence, SEQUENCE CAUTION types and gene cross-references, and writes
(1) the sensitivity-set table + FASTA and (2) per-species A22-schema flag files. A curated caution becomes a
**hard** flag only when the current reference proteome does not already carry the curated sequence.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "revision" / "scripts" / "63_swissprot_sensitivity.py"
QC_FLAGS = ROOT / "src" / "transgenic" / "datasets" / "qc_flags.py"


def _load(path, name):
    if not path.exists():
        pytest.fail(f"{path} does not exist yet (RED state)", pytrace=False)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def sp():
    return _load(SCRIPT, "swissprot_sensitivity")


@pytest.fixture(scope="module")
def qc():
    return _load(QC_FLAGS, "qc_flags_for_swissprot_test")


SEQ_A = "MEDQVGFGFRPNDEELVGHYLRNKIEGNTSRDVEVAISEVNICSYDPWNLRFQSKYKSRD"
SEQ_B = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"[:60]
SEQ_C = "MSSSSSSSKPLLTPEQVANSSASSSSSPKLPMTSRRRAAAAALQGSSTTPPPRSFSSSSLTSSSSSN"
SEQ_D = "MAAAKLSTGRLSSHSSSSAAQPKRGSRSPSTLAAAAAASQVSSLPSAALGRSFRSSGSFSSSGGGNAM"


def _fmt_seq(seq):
    chunks = [seq[i:i + 10] for i in range(0, len(seq), 10)]
    lines = []
    for i in range(0, len(chunks), 6):
        lines.append("     " + " ".join(chunks[i:i + 6]))
    return "\n".join(lines)


def _entry(entry_id, acc, taxid, oc, seq, pe="2: Evidence at transcript level;", cautions=(), drs=(), ft=""):
    cc = ""
    if cautions:
        cc = "CC   -!- SEQUENCE CAUTION:\n" + "".join(
            f"CC       Sequence={sid}; Type={ctype}; Note=Synthetic.; Evidence={{ECO:0000305}};\n" for sid, ctype in cautions)
    dr = "".join(f"DR   {d}\n" for d in drs)
    return (f"ID   {entry_id}              Reviewed;         {len(seq)} AA.\n"
            f"AC   {acc};\n"
            f"DT   01-JAN-2026, integrated into UniProtKB/Swiss-Prot.\n"
            f"DE   RecName: Full=Synthetic protein {entry_id};\n"
            f"OS   Synthetic organism.\n"
            f"{oc}"
            f"OX   NCBI_TaxID={taxid} {{ECO:0000312|Proteomes:UP000000001}};\n"
            f"{cc}{dr}"
            f"PE   {pe}\n"
            f"{ft}"
            f"SQ   SEQUENCE   {len(seq)} AA;  10000 MW;  0123456789ABCDEF CRC64;\n"
            f"{_fmt_seq(seq)}\n//\n")


OC_ATH = ("OC   Eukaryota; Viridiplantae; Streptophyta; Embryophyta; Tracheophyta;\n"
          "OC   Spermatophyta; Magnoliopsida; eudicotyledons; Brassicaceae; Arabidopsis.\n")
OC_OSA = ("OC   Eukaryota; Viridiplantae; Streptophyta; Embryophyta; Tracheophyta;\n"
          "OC   Spermatophyta; Magnoliopsida; Liliopsida; Poales; Poaceae; Oryza.\n")
OC_RED = "OC   Eukaryota; Rhodophyta; Bangiophyceae; Cyanidiales; Cyanidioschyzon.\n"

FT_NTERM = ("FT   INIT_MET        1\n"
            "FT                   /note=\"Removed\"\n"
            "FT                   /evidence=\"ECO:0000269|PubMed:22223895\"\n"
            "FT   CHAIN           2..60\n"
            "FT                   /note=\"Synthetic protein\"\n"
            "FT                   /id=\"PRO_0000000001\"\n")
FT_PREDICTED = ("FT   CHAIN           1..60\n"
                "FT                   /note=\"Synthetic protein\"\n"
                "FT                   /evidence=\"ECO:0000255\"\n")


def _dat():
    return "".join([
        # 1. Arabidopsis, PE1, experimental N-terminus, erroneous initiation, single gene xref -> hard candidate
        _entry("SYN1_ARATH", "P00001", 3702, OC_ATH, SEQ_A, pe="1: Evidence at protein level;",
               cautions=[("AAF00001.1", "Erroneous initiation")],
               drs=["TAIR; locus:2200935; AT1G01010.", "Araport; AT1G01010; -.", "EnsemblPlants; AT1G01010.1; AT1G01010.1; AT1G01010."],
               ft=FT_NTERM),
        # 2. Rice japonica, no caution, transcript-style xref -> mapped, no flag
        _entry("SYN2_ORYSJ", "P00002", 39947, OC_OSA, SEQ_B,
               drs=["EnsemblPlants; Os01t0100100-01; Os01t0100100-01; Os01g0100100.", "KEGG; osa:4326813; -."]),
        # 3. Arabidopsis, PE3, erroneous translation (not a structure error) -> soft note only
        _entry("SYN3_ARATH", "P00003", 3702, OC_ATH, SEQ_C, pe="3: Inferred from homology;",
               cautions=[("AAF00003.1", "Erroneous translation")],
               drs=["TAIR; locus:2200936; AT1G01020.", "Araport; AT1G01020; -."], ft=FT_PREDICTED),
        # 4. Rhodophyte -> not Viridiplantae, excluded
        _entry("SYN4_CYAME", "P00004", 45157, OC_RED, SEQ_D, cautions=[("BAA00004.1", "Frameshift")],
               drs=["KEGG; cme:CYME_CMA001C; -."]),
        # 5. Arabidopsis, frameshift caution but xrefs point to two genes -> ambiguous, no flag
        _entry("SYN5_ARATH", "P00005", 3702, OC_ATH, SEQ_D, cautions=[("AAF00005.1", "Frameshift")],
               drs=["TAIR; locus:2200937; AT1G01030.", "TAIR; locus:2200938; AT1G01040."]),
        # 6. Arabidopsis, erroneous termination, but the reference proteome already carries the curated sequence -> resolved note
        _entry("SYN6_ARATH", "P00006", 3702, OC_ATH, SEQ_B, cautions=[("AAF00006.1", "Erroneous termination")],
               drs=["Araport; AT1G01050; -."]),
    ])


@pytest.fixture(scope="module")
def parsed(sp, tmp_path_factory):
    d = tmp_path_factory.mktemp("sp")
    dat = d / "uniprot_sprot_plants.dat"
    dat.write_text(_dat())
    return d, dat, list(sp.parse_dat(str(dat)))


def test_parse_fields_and_viridiplantae_filter(sp, parsed):
    _, _, recs = parsed
    by_acc = {r.accession: r for r in recs}
    assert set(by_acc) == {"P00001", "P00002", "P00003", "P00004", "P00005", "P00006"}
    a = by_acc["P00001"]
    assert a.entry_name == "SYN1_ARATH" and a.taxid == 3702 and a.viridiplantae is True
    assert a.pe == 1 and a.nterm_experimental is True
    assert a.cautions == [("AAF00001.1", "Erroneous initiation")]
    assert a.sequence == SEQ_A and a.length == len(SEQ_A)
    assert "AT1G01010" in a.gene_xrefs and "AT1G01010.1" in a.gene_xrefs and "locus:2200935" not in a.gene_xrefs
    assert by_acc["P00002"].gene_xrefs >= {"Os01g0100100", "Os01t0100100-01"}
    assert by_acc["P00003"].pe == 3 and by_acc["P00003"].nterm_experimental is False
    assert by_acc["P00004"].viridiplantae is False
    kept = [r for r in recs if r.viridiplantae]
    assert {r.accession for r in kept} == {"P00001", "P00002", "P00003", "P00005", "P00006"}


def test_caution_classes(sp):
    assert sp.flag_name("Erroneous initiation") == "swissprot_caution_erroneous_initiation"
    assert sp.flag_name("Erroneous termination") == "swissprot_caution_erroneous_termination"
    assert sp.flag_name("Frameshift") == "swissprot_caution_frameshift"
    assert sp.flag_name("Erroneous gene model prediction") == "swissprot_caution_erroneous_gene_model_prediction"
    assert sp.flag_name("Erroneous translation") == "swissprot_note_erroneous_translation"
    assert sp.flag_name("Miscellaneous discrepancy") == "swissprot_note_miscellaneous_discrepancy"


def test_hard_soft_semantics_shared_with_a22(qc):
    assert qc.is_hard("swissprot_caution_erroneous_initiation")
    assert qc.is_hard("swissprot_caution_frameshift")
    assert not qc.is_hard("swissprot_note_erroneous_translation")
    assert not qc.is_hard("swissprot_note_caution_resolved_in_reference")
    assert not qc.is_hard("swissprot_note_unverified_no_proteome")


def test_species_flags_require_proteome_disagreement(sp, parsed):
    d, dat, recs = parsed
    gene_ids = {"Athaliana": {"AT1G01010", "AT1G01020", "AT1G01030", "AT1G01040", "AT1G01050"}, "Osativa": {"Os01g0100100"}}
    # reference proteome: AT1G01010.1 differs from the curated sequence (N-terminal extension), AT1G01050.1 equals it
    proteome = {"Athaliana": {"AT1G01010.1": "MAAAA" + SEQ_A, "AT1G01020.1": SEQ_C, "AT1G01050.1": SEQ_B + "*"}}
    taxids = {"Athaliana": {3702}, "Osativa": {4530, 39947}}
    rows, summary = sp.species_flags(recs, gene_ids, taxids, proteome)
    flags = {(r["species_id"], r["gene_id"], r["flag"]) for r in rows}
    assert ("Athaliana", "AT1G01010", "swissprot_caution_erroneous_initiation") in flags
    assert ("Athaliana", "AT1G01020", "swissprot_note_erroneous_translation") in flags
    assert ("Athaliana", "AT1G01050", "swissprot_note_caution_resolved_in_reference") in flags
    assert not any(g in ("AT1G01030", "AT1G01040") for _, g, _ in flags)          # ambiguous xrefs -> no flag
    assert not any(s == "Osativa" and f.startswith("swissprot_caution") for s, _, f in flags)
    assert all(r["transcript_id"] == "" for r in rows)                            # gene-level rows (A22 '*')
    assert summary["Athaliana"]["mapped"] == 3 and summary["Athaliana"]["ambiguous"] == 1
    assert summary["Athaliana"]["hard_flags"] == 1 and summary["Athaliana"]["resolved_in_reference"] == 1
    assert summary["Osativa"]["mapped"] == 1 and summary["Osativa"]["hard_flags"] == 0
    # without a proteome the caution cannot be verified against the current model -> soft note, never a hard flag
    rows2, summary2 = sp.species_flags(recs, gene_ids, taxids, {})
    assert ("Athaliana", "AT1G01010", "swissprot_note_unverified_no_proteome") in {(r["species_id"], r["gene_id"], r["flag"]) for r in rows2}
    assert summary2["Athaliana"]["hard_flags"] == 0


def test_cli_writes_table_fasta_flags_and_summary(sp, parsed, tmp_path):
    d, dat, _ = parsed
    genes = tmp_path / "genes.tsv"
    genes.write_text("species_id\tgene_id\nAthaliana\tAT1G01010\nAthaliana\tAT1G01020\nAthaliana\tAT1G01050\nOsativa\tOs01g0100100\n")
    prot = tmp_path / "Athaliana.protein.fa"
    prot.write_text(f">AT1G01010.1 pacid=1\n{'MAAAA' + SEQ_A}\n>AT1G01020.1\n{SEQ_C}\n>AT1G01050.1\n{SEQ_B}*\n")
    out = tmp_path / "out"
    rc = sp.main(["--dat", str(dat), "--out-dir", str(out), "--gene-ids", str(genes),
                  "--species-taxid", "Athaliana=3702", "--species-taxid", "Osativa=4530,39947",
                  "--proteome", f"Athaliana={prot}"])
    assert rc == 0
    table = (out / "swissprot_viridiplantae.tsv").read_text().splitlines()
    header = table[0].split("\t")
    for col in ("accession", "entry_name", "taxid", "pe", "nterm_experimental", "caution_types", "gene_xrefs", "length", "sha256"):
        assert col in header
    assert len(table) == 1 + 5                                                     # Viridiplantae entries only
    fasta = (out / "swissprot_viridiplantae.fa").read_text()
    assert fasta.count(">") == 5 and ">P00004" not in fasta and SEQ_A in fasta
    flags_ath = (out / "Athaliana.swissprot_flags.tsv").read_text().splitlines()
    assert flags_ath[0] == "species_id\tgene_id\ttranscript_id\tflag\tstart\tend"
    assert any("AT1G01010\t\tswissprot_caution_erroneous_initiation" in l for l in flags_ath)
    flags_osa = (out / "Osativa.swissprot_flags.tsv").read_text().splitlines()
    assert len(flags_osa) == 1                                                    # header only
    summary = json.loads((out / "swissprot_summary.json").read_text())
    assert summary["entries_total"] == 6 and summary["entries_viridiplantae"] == 5
    assert summary["species"]["Athaliana"]["hard_flags"] == 1
    assert summary["strata"]["pe1_nterm_experimental"] == 1 and summary["strata"]["caution_structure"] == 3
    assert "dat_sha256" in summary


# --- review findings (Kimi K3, 2026-09-02): proteome-header mapping, fail-closed sparse proteome, negative controls ---

def test_gene_map_from_gff_and_proteome_suffixes(sp, tmp_path):
    gff = tmp_path / "mixed.gff3"
    gff.write_text("##gff-version 3\n"
                   "Chr01\tphytozome\tgene\t1\t900\t.\t+\t.\tID=Glyma.01G000100;Name=Glyma.01G000100\n"
                   "Chr01\tphytozome\tmRNA\t1\t900\t.\t+\t.\tID=Glyma.01G000100.1;Parent=Glyma.01G000100\n"
                   "Chr01\tphytozome\tmRNA\t1\t800\t.\t+\t.\tID=Glyma.01G000100.2;Parent=Glyma.01G000100\n"
                   "Chr01\tphytozome\tCDS\t1\t800\t.\t+\t0\tID=Glyma.01G000100.2.CDS.1;Parent=Glyma.01G000100.2\n"
                   "Chr01\tphytozome\tgene\t1000\t1900\t.\t+\t.\tID=Pp3c1_10V3;Name=Pp3c1_10V3\n"
                   "Chr01\tphytozome\tmRNA\t1000\t1900\t.\t+\t.\tID=Pp3c1_10V3.1;Parent=Pp3c1_10V3\n"
                   "chr01\trap\tgene\t2000\t2900\t.\t+\t.\tID=Os01g0100100;Name=Os01g0100100\n"
                   "chr01\trap\tmRNA\t2000\t2900\t.\t+\t.\tID=Os01t0100100-01;Parent=Os01g0100100\n")
    genes, tx2gene = sp.read_gff_gene_map(str(gff))
    assert genes == {"Glyma.01G000100", "Pp3c1_10V3", "Os01g0100100"}
    assert tx2gene["Glyma.01G000100.2"] == "Glyma.01G000100" and tx2gene["Os01t0100100-01"] == "Os01g0100100"
    assert "Glyma.01G000100.2.CDS.1" not in tx2gene                                   # only transcript-level features
    assert sp.gene_of("Glyma.01G000100.1.p", genes, tx2gene) == "Glyma.01G000100"     # Phytozome protein header (.p)
    assert sp.gene_of("Pp3c1_10V3.1.p", genes, tx2gene) == "Pp3c1_10V3"
    assert sp.gene_of("Os01t0100100-01", genes, tx2gene) == "Os01g0100100"            # RAP transcript xref
    assert sp.gene_of("LOC_Os01g01010", genes, tx2gene) is None                       # MSU id against a RAP annotation
    assert sp.gene_of("AT1G01010.1", {"AT1G01010"}, {}) == "AT1G01010"                # fallback without a map: one suffix
    assert sp.gene_of("AT1G01010.1.p", {"AT1G01010"}, {}) == "AT1G01010"              # fallback strips .p then .1


def test_sparse_proteome_fails_closed_and_unmapped_breakdown(sp, parsed):
    _, _, recs = parsed
    gene_ids = {"Athaliana": {"AT1G01010", "AT1G01020", "AT1G01050"}, "Osativa": {"LOC_Os01g01010"}}
    taxids = {"Athaliana": {3702}, "Osativa": {4530, 39947}}
    bad = {"Athaliana": {"XYZ.1": "MAAA", "XYZ.2": "MBBB"}}                            # no header maps to a gene
    with pytest.raises(SystemExit):
        sp.species_flags(recs, gene_ids, taxids, bad)                                 # default threshold: fail closed
    rows, summary = sp.species_flags(recs, gene_ids, taxids, bad, min_proteome_mapped=0.0)
    st = summary["Athaliana"]
    assert st["proteome_records"] == 2 and st["proteome_mapped"] == 0
    assert st["hard_flags"] == 0                                                      # a gene without any reference protein is unverified, never hard
    assert ("Athaliana", "AT1G01010", "swissprot_note_unverified_no_proteome") in {(r["species_id"], r["gene_id"], r["flag"]) for r in rows}
    # a gene that has reference proteins is still verified normally in the same run
    partial = {"Athaliana": {"AT1G01010.1": "MAAAA" + SEQ_A, "AT1G01050.1": SEQ_B}}
    rows2, summary2 = sp.species_flags(recs, gene_ids, taxids, partial)
    assert summary2["Athaliana"]["proteome_mapped"] == 2 and summary2["Athaliana"]["hard_flags"] == 1
    # unmapped entries are broken down by cross-reference database (MSU annotation vs RAP/EnsemblPlants ids)
    assert summary2["Osativa"]["unmapped"] == 1 and summary2["Osativa"]["unmapped_by_db"] == {"EnsemblPlants": 1, "KEGG": 1}


def test_nterm_negative_controls_dual_taxid_gzip_truncation_and_caution_blocks(sp, tmp_path):
    import gzip
    ft_chain3 = "FT   CHAIN           3..60\nFT                   /evidence=\"ECO:0000269|PubMed:1\"\n"
    ft_init_noev = "FT   INIT_MET        1\nFT                   /note=\"Removed\"\n"
    legacy = ("CC   -!- SEQUENCE CAUTION:\n"
              "CC       Legacy free-text caution without a structured record.\n"
              "CC   -!- MISCELLANEOUS: Sequence=AAA00000.1; Type=Frameshift; is not a caution block.\n")
    text = (_entry("NEG1_ARATH", "P00011", 3702, OC_ATH, SEQ_A, pe="1: Evidence at protein level;", ft=ft_chain3)
            + _entry("NEG2_ARATH", "P00012", 3702, OC_ATH, SEQ_C, pe="1: Evidence at protein level;", ft=ft_init_noev)
            + _entry("DUAL_ORYSJ", "P00013", "39947 {ECO:0000312|Proteomes:UP1}, 4530", OC_OSA, SEQ_B))
    text = text.replace("PE   1: Evidence at protein level;\n" + ft_init_noev,
                        legacy + "PE   1: Evidence at protein level;\n" + ft_init_noev)
    truncated = text + _entry("LAST_ARATH", "P00014", 3702, OC_ATH, SEQ_D).replace("//\n", "")
    gz = tmp_path / "t.dat.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(truncated)
    recs = {r.accession: r for r in sp.parse_dat(str(gz))}
    assert set(recs) == {"P00011", "P00012", "P00013", "P00014"}
    assert recs["P00011"].nterm_experimental is False                                # CHAIN starting at 3 is not an N-terminus call
    assert recs["P00012"].nterm_experimental is False                                # INIT_MET without an experimental code
    assert recs["P00012"].cautions == [] and recs["P00012"].caution_blocks == 1      # legacy block counted, nothing parsed; MISCELLANEOUS ignored
    assert recs["P00013"].taxids == {39947, 4530} and recs["P00013"].taxid == 39947
    assert recs["P00014"].truncated is True and recs["P00014"].sequence == SEQ_D and recs["P00011"].truncated is False
    rows, summary = sp.species_flags(recs.values(), {"Osativa": set()}, {"Osativa": {4530}}, {})
    assert summary["Osativa"]["entries"] == 1                                         # dual-taxid entry matched through its second taxid
