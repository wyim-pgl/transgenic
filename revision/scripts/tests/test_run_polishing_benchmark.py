"""Tests for 33_run_polishing_benchmark.py.

None of these tests touch SSH, the GPU host, or a GPU: every function that talks to the
network is injectable (`ssh_run`, `fetch`, `push`) and every test here supplies a fake.
The functions that wrap real project code (`standardize_output`, `filter_top_beam`) are
exercised against the real `transgenic_comparison/standardize_gff.py` and
`13_beam1_filter.py`, but only against tiny synthetic fixtures written to `tmp_path`.

Fix round 1 (task-4-findings-round1.md) added: build_local_subset (I10), the C1 atomic
write, the C2 chromosome-qualified naming, the C3 acknowledgement gate, I4's enforced
host-commit pinning, I5's log-on-failure fetch, I2's found/N-A distinction, I1's
intermediate counts, I3's missing-loci manifest, I7's truncation check, and the I8/I9
tests that replace ones the reviewer showed could not fail.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_bench", Path(__file__).resolve().parents[1] / "33_run_polishing_benchmark.py")
run_bench = importlib.util.module_from_spec(spec)
sys.modules["run_bench"] = run_bench
spec.loader.exec_module(run_bench)


# ----------------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------------

def _gff_lines(rows):
    """rows: (seq, feat, start, end, attrs_str)"""
    return "".join(f"{seq}\tx\t{feat}\t{s}\t{e}\t.\t+\t.\t{attrs}\n"
                   for seq, feat, s, e, attrs in rows)


def _write_gff(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_gff_lines(rows))


def _make_gene_triples(seq, ids, start=1000, step=2000):
    """(gene, mRNA, CDS) rows for each id in `ids`, non-overlapping, in file order.

    Gene rows carry `GM=`, as real prompt_mode.py output does, so fixtures built from
    this are usable on both sides of the input/output pairing the R4-3 gates do.
    """
    rows = []
    for i, gid in enumerate(ids):
        s = start + i * step
        e = s + 500
        rows.append((seq, "gene", s, e, f"ID={gid};GM={gid}"))
        rows.append((seq, "mRNA", s, e, f"ID={gid}.1;Parent={gid}"))
        rows.append((seq, "CDS", s, e, f"ID={gid}.1.cds;Parent={gid}.1"))
    return rows


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRemote:
    """A fake ssh_run + fetch + push trio that share enough state to make the
    checksum-verification round trips in `fetch_and_verify`/`push_and_verify`
    self-consistent: `fetch()`/`push()` really register the local file, and the
    `md5sum <remote>` ssh branch reports that same file's real md5 back — exactly what
    a real host would report after a real rsync, so tests never hand-compute a hash.

    `gff_rows_by_hint`: {substring-of-remote-path: rows} — picks which fixture rows to
    write for a given remote *output* path. A fetched output path matching no hint
    raises, so an unexpected fetch fails the test instead of silently succeeding. (Input
    subset pushes are not looked up this way — the pushed file already has real content,
    written by `build_local_subset` before the push ever happens.)

    `fail_hints`: remote commands containing any of these substrings get a nonzero
    return code instead of the canned success response (simulates a remote crash for
    that one chunk, or one specific step, without touching the others).

    `host_commit`: what `git rev-parse HEAD` reports; defaults to the pinned commit
    gpu-environment.md verifies reproduces the published benchmark.
    """

    def __init__(self, gff_rows_by_hint, fail_hints=(),
                 log_text="Output written to: out.gff3\n",
                 prompt_stdout="Output written to: out.gff3\n",
                 host_commit=None):
        self.gff_rows_by_hint = gff_rows_by_hint
        self.fail_hints = tuple(fail_hints)
        self.log_text = log_text
        self.prompt_stdout = prompt_stdout
        self.host_commit = host_commit or run_bench.EXPECTED_HOST_COMMIT
        self._written = {}

    def ssh_run(self, argv):
        cmd = argv[-1]
        if cmd.startswith("md5sum "):
            remote_path = cmd[len("md5sum "):].strip()
            local_path = self._written.get(remote_path)
            if local_path is None or not local_path.exists():
                return FakeCompletedProcess(returncode=1, stderr=f"no such file: {remote_path}")
            return FakeCompletedProcess(returncode=0, stdout=f"{run_bench.md5_file(local_path)}  {remote_path}\n")
        if cmd.endswith("git rev-parse HEAD"):
            return FakeCompletedProcess(returncode=0, stdout=f"{self.host_commit}\n")
        if any(hint in cmd for hint in self.fail_hints):
            return FakeCompletedProcess(returncode=1, stderr="simulated remote failure")
        return FakeCompletedProcess(returncode=0, stdout=self.prompt_stdout)

    def fetch(self, remote_path, local_path, host="gpu"):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_path.endswith(".log"):
            local_path.write_text(self.log_text)
        else:
            rows = next((r for hint, r in self.gff_rows_by_hint.items() if hint in remote_path), None)
            if rows is None:
                raise AssertionError(f"unexpected fetch for {remote_path}")
            _write_gff(local_path, rows)
        self._written[remote_path] = local_path

    def push(self, local_path, remote_path, host="gpu"):
        # A real rsync push copies bytes to the host; here the "host" is just this
        # object's own bookkeeping, keyed the same way fetch() keys it, so the
        # subsequent md5sum branch above can find and hash the (already real) local
        # file build_local_subset wrote.
        self._written[remote_path] = local_path


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Point every path constant at a scratch tree so tests never touch real staged
    data or the real predictions directory."""
    monkeypatch.setattr(run_bench, "LOCAL_INPUTS", tmp_path / "inputs")
    monkeypatch.setattr(run_bench, "LOCAL_PREDICTIONS", tmp_path / "predictions")
    monkeypatch.setattr(run_bench, "LOCAL_CHUNKS", tmp_path / "predictions" / "_chunks")
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "predictions").mkdir(parents=True)
    return tmp_path


def _make_fake_ssh(returncode=0, stdout="Output written to: out.gff3\n", stderr=""):
    def fake_ssh_run(argv):
        return FakeCompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr)
    return fake_ssh_run


# ----------------------------------------------------------------------------
# scratch-file guard
# ----------------------------------------------------------------------------

def test_is_scratch_name_matches_smoke_and_bs_prefixes():
    assert run_bench.is_scratch_name("_smoke_gemoma_Chr1_20genes.gff3")
    assert run_bench.is_scratch_name("_bs1.gff")
    assert run_bench.is_scratch_name("/some/dir/_bs4.db")


def test_is_scratch_name_leaves_real_names_alone():
    assert not run_bench.is_scratch_name("gemoma_Athaliana.gff3")
    assert not run_bench.is_scratch_name("gemoma_Chr1_raw.gff3")


def test_assert_not_scratch_raises_for_scratch_path():
    with pytest.raises(ValueError):
        run_bench.assert_not_scratch("~/polishing_benchmark/predictions/_bs32.gff")


def test_assert_not_scratch_passes_for_real_path():
    run_bench.assert_not_scratch("~/polishing_benchmark/predictions/gemoma_Chr1_raw.gff3")


# ----------------------------------------------------------------------------
# GFF3 attribute parsing
# ----------------------------------------------------------------------------

def test_gff_attr_reads_key_anchored_on_key_name():
    assert run_bench._gff_attr("ID=g1;Parent=p1", "ID") == "g1"
    assert run_bench._gff_attr("ID=g1;Parent=p1", "Parent") == "p1"


def test_gff_attr_does_not_match_a_different_attribute_ending_in_the_same_key():
    # "GeneID=" must not satisfy a lookup for "ID"
    assert run_bench._gff_attr("GeneID=xyz", "ID") is None


def test_gff_attr_returns_none_when_key_absent():
    assert run_bench._gff_attr("ID=g1", "GM") is None


def test_first_attribute_value_matches_the_pinned_hosts_own_gm_derivation():
    """N2: preprocess.py:314 derives GM= from the *first* attribute, whatever key it
    is — GeMoMa's real gene rows lead with Name=, not ID=."""
    assert run_bench._first_attribute_value("Name=Ath_00001;ID=gene_0;transcripts=1") == "Ath_00001"
    assert run_bench._first_attribute_value("ID=gene_0;Name=Ath_00001") == "gene_0"


def test_first_attribute_value_returns_none_for_a_malformed_leading_attribute():
    assert run_bench._first_attribute_value("not_a_key_value_pair;ID=g1") is None


# ----------------------------------------------------------------------------
# stderr/stdout parsing
# ----------------------------------------------------------------------------

def test_parse_parsing_errors_found():
    text = "Output written to: out.gff\nParsing errors: 3/18 sequences skipped\n"
    perr = run_bench.parse_parsing_errors(text)
    assert perr.found is True
    assert perr.skipped == 3
    assert perr.total == 18


def test_parse_parsing_errors_not_found_means_zero_not_missing():
    perr = run_bench.parse_parsing_errors("Output written to: out.gff\n")
    assert perr.found is False
    assert perr.skipped == 0


def test_parse_output_written_present():
    assert run_bench.parse_output_written("...\nOutput written to: /x/y.gff\n") == "/x/y.gff"


def test_parse_output_written_absent():
    assert run_bench.parse_output_written("Traceback...\nRuntimeError: boom\n") is None


# ----------------------------------------------------------------------------
# GFF3 counting
# ----------------------------------------------------------------------------

def test_count_gff3_features_counts_only_requested_type(tmp_path):
    p = tmp_path / "a.gff3"
    _write_gff(p, [
        ("Chr1", "gene", 1, 100, "ID=g1"),
        ("Chr1", "mRNA", 1, 100, "ID=g1.1;Parent=g1"),
        ("Chr1", "gene", 200, 300, "ID=g2"),
    ])
    assert run_bench.count_gff3_features(p, "gene") == 2
    assert run_bench.count_gff3_features(p, "mRNA") == 1


def test_count_gff3_features_missing_file_is_zero(tmp_path):
    assert run_bench.count_gff3_features(tmp_path / "nope.gff3", "gene") == 0


def test_md5_file_matches_known_hash(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    assert run_bench.md5_file(p) == hashlib.md5(b"hello\n").hexdigest()


# ----------------------------------------------------------------------------
# loss detection
# ----------------------------------------------------------------------------

def test_check_loss_passes_within_threshold():
    run_bench.check_loss(genes_in=18, genes_out=18, max_loss_fraction=0.05, context="t")
    run_bench.check_loss(genes_in=100, genes_out=97, max_loss_fraction=0.05, context="t")


def test_check_loss_raises_above_threshold():
    with pytest.raises(run_bench.LossTooHigh):
        run_bench.check_loss(genes_in=18, genes_out=15, max_loss_fraction=0.05, context="t/Chr1")


def test_check_loss_raises_on_zero_genes_in():
    with pytest.raises(run_bench.LossTooHigh):
        run_bench.check_loss(genes_in=0, genes_out=0, max_loss_fraction=0.05, context="t")


def test_check_loss_message_names_the_context():
    with pytest.raises(run_bench.LossTooHigh, match="gemoma/Chr1"):
        run_bench.check_loss(genes_in=18, genes_out=0, max_loss_fraction=0.05, context="gemoma/Chr1")


# ----------------------------------------------------------------------------
# remote command construction
# ----------------------------------------------------------------------------

def test_build_prompt_mode_command_hardcodes_batch_size_1_and_num_beams_2():
    cmd = run_bench.build_prompt_mode_command("in.gff3", "out.gff3", "out.db", "out.log")
    assert "--batch-size 1" in cmd
    assert "--num-beams 2" in cmd
    assert "--batch-size 32" not in cmd


def test_build_prompt_mode_command_changes_cwd_to_transgenic():
    cmd = run_bench.build_prompt_mode_command("in.gff3", "out.gff3", "out.db", "out.log")
    assert f"cd {run_bench.REMOTE_TRANSGENIC_DIR}" in cmd


def test_build_prompt_mode_command_rejects_scratch_output():
    with pytest.raises(ValueError):
        run_bench.build_prompt_mode_command("in.gff3", "_bs1.gff", "out.db", "out.log")


def test_build_chunk_remote_command_paths_are_tool_and_chrom_specific():
    paths = run_bench.build_chunk_remote_command("gemoma", "ChrM")
    assert "ChrM" in paths["remote_output"]
    assert "gemoma" in paths["remote_output"]
    assert "ChrM" in paths["remote_subset"]
    assert "--batch-size 1" in paths["command"]
    # No more remote subsetting step — the command is the inference call only.
    assert "awk" not in paths["command"]


def test_build_ssh_argv_shape():
    argv = run_bench.build_ssh_argv("echo hi", host="gpu")
    assert argv == ["ssh", "gpu", "echo hi"]


def test_build_rsync_fetch_argv_shape(tmp_path):
    argv = run_bench.build_rsync_fetch_argv("~/x.gff3", tmp_path / "x.gff3", host="gpu")
    assert argv[0] == "rsync"
    assert argv[-2] == "gpu:~/x.gff3"


def test_build_rsync_push_argv_shape(tmp_path):
    local = tmp_path / "x.gff3"
    argv = run_bench.build_rsync_push_argv(local, "~/x.gff3", host="gpu")
    assert argv[0] == "rsync"
    assert argv[-1] == "gpu:~/x.gff3"
    assert argv[-2] == str(local)


# ----------------------------------------------------------------------------
# build_local_subset (I10): filters non-coding top-level loci and their descendants
# before anything is shipped to the host, so the DB the host builds reconciles with
# count_genes_for_chromosome's own definition and with Task 2/3's NONCODING_TYPES.
# ----------------------------------------------------------------------------

def test_build_local_subset_excludes_noncoding_top_level_and_descendants(tmp_path):
    inp = tmp_path / "egapx_Athaliana.gff3"
    _write_gff(inp, [
        ("ChrM", "gene", 100, 200, "ID=g1"),
        ("ChrM", "mRNA", 100, 200, "ID=g1.1;Parent=g1"),
        ("ChrM", "CDS", 100, 200, "ID=g1.1.cds;Parent=g1.1"),
        ("ChrM", "pseudogene", 300, 400, "ID=p1"),
        ("ChrM", "mRNA", 300, 400, "ID=p1.1;Parent=p1"),
        ("ChrM", "exon", 300, 400, "ID=p1.1.exon;Parent=p1.1"),
        ("ChrM", "lnc_RNA", 500, 600, "ID=l1"),
        ("ChrM", "exon", 500, 600, "ID=l1.exon;Parent=l1"),
        ("Chr1", "gene", 100, 200, "ID=g2"),  # different chromosome: filtered out too
    ])
    subset = tmp_path / "subset.gff3"
    info = run_bench.build_local_subset("egapx", "ChrM", inp, subset)
    assert info["genes"] == 1
    assert info["excluded_feature_counts"] == {"pseudogene": 1, "mRNA": 1, "exon": 2, "lnc_RNA": 1}
    kept_ids = {run_bench._gff_attr(line.split("\t")[8], "ID")
                for line in subset.read_text().splitlines() if line.strip()}
    assert kept_ids == {"g1", "g1.1", "g1.1.cds"}


def test_build_local_subset_drops_gene_shells_left_childless_by_exclusion(tmp_path):
    """N1: EGAPx nests gene(gbkey=lncRNA) -> lnc_RNA -> exon. Once the lnc_RNA is
    excluded, g1 (the gene) is a childless shell the model can never be prompted from —
    it must not be counted into genes_in, or the denominator itself guarantees loss
    before inference starts. g2 (a real protein-coding gene) must be unaffected."""
    inp = tmp_path / "egapx_Athaliana.gff3"
    _write_gff(inp, [
        ("ChrM", "gene", 100, 200, "ID=g1;gbkey=Gene;gene_biotype=lncRNA"),
        ("ChrM", "lnc_RNA", 100, 200, "ID=g1.rna;Parent=g1"),
        ("ChrM", "exon", 100, 200, "ID=g1.rna.exon;Parent=g1.rna"),
        ("ChrM", "gene", 300, 400, "ID=g2"),
        ("ChrM", "mRNA", 300, 400, "ID=g2.1;Parent=g2"),
        ("ChrM", "CDS", 300, 400, "ID=g2.1.cds;Parent=g2.1"),
    ])
    subset = tmp_path / "subset.gff3"
    info = run_bench.build_local_subset("egapx", "ChrM", inp, subset)
    assert info["genes"] == 1
    assert info["excluded_feature_counts"]["gene"] == 1
    kept_ids = {run_bench._gff_attr(line.split("\t")[8], "ID")
                for line in subset.read_text().splitlines() if line.strip()}
    assert kept_ids == {"g2", "g2.1", "g2.1.cds"}


def test_build_local_subset_drops_transcript_rows_flagged_noncoding_by_gbkey(tmp_path):
    """N5: a bare "transcript"-typed row with gbkey=misc_RNA is non-coding by Task 2/3's
    own NONCODING_GBKEYS convention, even though its type string alone ("transcript")
    is not in NONCODING_TOP_LEVEL_TYPES."""
    inp = tmp_path / "egapx_Athaliana.gff3"
    _write_gff(inp, [
        ("ChrM", "gene", 100, 200, "ID=g1"),
        ("ChrM", "transcript", 100, 200, "ID=g1.t1;Parent=g1;gbkey=misc_RNA"),
        ("ChrM", "exon", 100, 200, "ID=g1.t1.exon;Parent=g1.t1"),
        ("ChrM", "gene", 300, 400, "ID=g2"),
        ("ChrM", "mRNA", 300, 400, "ID=g2.1;Parent=g2"),
        ("ChrM", "CDS", 300, 400, "ID=g2.1.cds;Parent=g2.1"),
    ])
    subset = tmp_path / "subset.gff3"
    info = run_bench.build_local_subset("egapx", "ChrM", inp, subset)
    # g1's only transcript is excluded by gbkey, leaving g1 itself childless too.
    assert info["genes"] == 1
    assert info["excluded_feature_counts"]["transcript"] == 1
    assert info["excluded_feature_counts"]["gene"] == 1


def test_build_local_subset_only_includes_requested_chromosome(tmp_path):
    inp = tmp_path / "gemoma_Athaliana.gff3"
    _write_gff(inp, [
        ("Chr1", "gene", 1, 100, "ID=g1"),
        ("Chr1", "mRNA", 1, 100, "ID=g1.1;Parent=g1"),
        ("Chr2", "gene", 1, 100, "ID=g2"),
        ("Chr2", "mRNA", 1, 100, "ID=g2.1;Parent=g2"),
    ])
    subset = tmp_path / "subset.gff3"
    info = run_bench.build_local_subset("gemoma", "Chr1", inp, subset)
    assert info["genes"] == 1
    assert "g2" not in subset.read_text()


def test_build_local_subset_rejects_scratch_paths(tmp_path):
    with pytest.raises(ValueError):
        run_bench.build_local_subset("gemoma", "Chr1", tmp_path / "_smoke_in.gff3", tmp_path / "out.gff3")


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------

def test_merge_gff3_files_concatenates_and_counts_genes(tmp_path):
    p1 = tmp_path / "c1.gff3"
    p2 = tmp_path / "c2.gff3"
    _write_gff(p1, [("Chr1", "gene", 1, 100, "ID=g1")])
    _write_gff(p2, [("Chr2", "gene", 1, 100, "ID=g2"), ("Chr2", "gene", 200, 300, "ID=g3")])
    out = tmp_path / "merged.gff3"
    n = run_bench.merge_gff3_files([p1, p2], out)
    assert n == 3
    assert run_bench.count_gff3_features(out, "gene") == 3


def test_merge_gff3_files_raises_on_missing_chunk(tmp_path):
    p1 = tmp_path / "c1.gff3"
    _write_gff(p1, [("Chr1", "gene", 1, 100, "ID=g1")])
    with pytest.raises(FileNotFoundError):
        run_bench.merge_gff3_files([p1, tmp_path / "missing.gff3"], tmp_path / "out.gff3")


def test_merge_gff3_files_rejects_scratch_input(tmp_path):
    scratch = tmp_path / "_smoke_x.gff3"
    _write_gff(scratch, [("Chr1", "gene", 1, 100, "ID=g1")])
    with pytest.raises(ValueError):
        run_bench.merge_gff3_files([scratch], tmp_path / "out.gff3")


# ----------------------------------------------------------------------------
# standardize / top-beam filter wiring (real project code, tiny fixtures)
#
# I8: the original single test used a gene with no mRNA/CDS children, which
# standardize_gff() drops unconditionally ("Removed genes with no valid mRNAs") — so it
# passed under the correct config, a missing .fai, AND a wrong species string, and could
# never fail. These three tests use a gene WITH real children so only the specific thing
# each test claims to check can make it pass or fail.
# ----------------------------------------------------------------------------

def test_standardize_output_drops_out_of_bounds_gene_with_real_children(tmp_path):
    raw = tmp_path / "raw.gff3"
    # Chr1 is 30,427,671 bp in the real TAIR10 .fai; g2 sits well beyond it.
    _write_gff(raw, [
        ("Chr1", "gene", 100, 200, "ID=g1"),
        ("Chr1", "mRNA", 100, 200, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 100, 200, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 40_000_000, 40_000_300, "ID=g2"),
        ("Chr1", "mRNA", 40_000_000, 40_000_300, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 40_000_000, 40_000_300, "ID=g2.1.cds;Parent=g2.1"),
    ])
    out = tmp_path / "std.gff3"
    run_bench.standardize_output(raw, out, "gemoma")
    assert run_bench.count_gff3_features(out, "gene") == 1
    assert "40000000" not in out.read_text()


def test_standardize_output_raises_if_fai_missing(tmp_path):
    raw = tmp_path / "raw.gff3"
    _write_gff(raw, [
        ("Chr1", "gene", 100, 200, "ID=g1"),
        ("Chr1", "mRNA", 100, 200, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 100, 200, "ID=g1.1.cds;Parent=g1.1"),
    ])
    out = tmp_path / "std.gff3"
    empty_genome_dir = tmp_path / "no_fai_here"
    empty_genome_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        run_bench.standardize_output(raw, out, "gemoma", genome_dir=empty_genome_dir)


def test_standardize_gff_itself_silently_keeps_out_of_bounds_gene_under_wrong_species(tmp_path):
    """Documents the danger standardize_output's hardcoded "A_thaliana" avoids: called
    with an unrecognised species string, load_chrom_lengths() returns {} and
    standardize_gff() falls back to the 500,000,000 bp MAX_COORDINATE ceiling, under
    which a gene at 40,000,000 is not out of bounds at all — the same gene the test
    above correctly drops when species is right."""
    raw = tmp_path / "raw.gff3"
    _write_gff(raw, [
        ("Chr1", "gene", 40_000_000, 40_000_300, "ID=g2"),
        ("Chr1", "mRNA", 40_000_000, 40_000_300, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 40_000_000, 40_000_300, "ID=g2.1.cds;Parent=g2.1"),
    ])
    out = tmp_path / "std.gff3"
    mod = run_bench._load_module("standardize_gff_direct", run_bench.STANDARDIZE_SCRIPT)
    mod.standardize_gff(str(raw), str(out), "Z_not_a_real_species", "gemoma",
                         genome_dir=str(run_bench.LOCAL_GENOME_DIR))
    assert run_bench.count_gff3_features(out, "gene") == 1  # survived — not caught


def test_filter_top_beam_keeps_first_gene_per_gm(tmp_path):
    raw = tmp_path / "twobeam.gff3"
    _write_gff(raw, [
        ("Chr1", "gene", 100, 200, "ID=g1;GM=locusA"),
        ("Chr1", "mRNA", 100, 200, "ID=g1.1;Parent=g1"),
        ("Chr1", "gene", 100, 210, "ID=g2;GM=locusA"),  # second beam, same locus
        ("Chr1", "mRNA", 100, 210, "ID=g2.1;Parent=g2"),
    ])
    out = tmp_path / "filtered.gff3"
    run_bench.filter_top_beam(raw, out)
    assert run_bench.count_gff3_features(out, "gene") == 1
    assert "ID=g1" in out.read_text()
    assert "ID=g2" not in out.read_text()


# ----------------------------------------------------------------------------
# missing-loci manifest (I3)
# ----------------------------------------------------------------------------

def test_write_missing_loci_manifest_lists_gm_values_absent_from_output(tmp_path):
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [("Chr1", "gene", 1, 100, "ID=g1"), ("Chr1", "gene", 200, 300, "ID=g2")])
    final = tmp_path / "final.gff3"
    _write_gff(final, [("Chr1", "gene", 1, 100, "ID=g1out;GM=g1")])  # g2 never emitted
    out = tmp_path / "missing.txt"
    result = run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)
    assert result == {"total": 1, "structurally_unreachable": 0, "unexplained": 1}
    lines = out.read_text().splitlines()
    assert lines[0] == "locus\treason"
    assert lines[1] == f"g2\t{run_bench.UNEXPLAINED_MISSING_REASON}"


def test_write_missing_loci_manifest_empty_when_nothing_missing(tmp_path):
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [("Chr1", "gene", 1, 100, "ID=g1")])
    final = tmp_path / "final.gff3"
    _write_gff(final, [("Chr1", "gene", 1, 100, "ID=g1out;GM=g1")])
    out = tmp_path / "missing.txt"
    result = run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)
    assert result == {"total": 0, "structurally_unreachable": 0, "unexplained": 0}
    assert out.read_text().splitlines() == ["locus\treason"]


def test_write_missing_loci_manifest_labels_the_known_structurally_unreachable_locus(tmp_path):
    """N6 (round 3): a locus flagged by run_remote_chunk as the last gene in its chunk's
    file order (structurally unreachable per genome2GSFDataset's no-final-flush bug)
    must be labeled differently from a genuinely unexplained miss — that distinction is
    the whole point of the manifest, per the reviewer."""
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [
        ("Chr1", "gene", 1, 100, "ID=g1"),
        ("Chr1", "gene", 200, 300, "ID=g2"),  # last in file order -> structurally unreachable
    ])
    final = tmp_path / "final.gff3"
    _write_gff(final, [("Chr1", "gene", 1, 100, "ID=g1out;GM=g1")])  # only g1 emitted
    out = tmp_path / "missing.txt"
    chunk_results = [{"tool": "gemoma", "chromosome": "Chr1", "structurally_unreachable_locus": "g2"}]
    result = run_bench.write_missing_loci_manifest(chunk_results, final, out)
    assert result == {"total": 1, "structurally_unreachable": 1, "unexplained": 0}
    lines = out.read_text().splitlines()
    assert lines[1] == f"g2\t{run_bench.STRUCTURALLY_UNREACHABLE_REASON}"


def test_write_missing_loci_manifest_uses_first_attribute_not_id_for_gemoma_row_shape(tmp_path):
    """N2: the pinned host derives GM= as the *first* attribute value on the gene line
    (preprocess.py:314), not specifically ID=. GeMoMa's real gene rows lead with Name=
    (`Name=Ath_00001;ID=gene_0;...`), so GM=Ath_00001 while the old ID-keyed manifest
    looked up gene_0 and never found it — a perfect run reported 100% missing. This
    fixture reproduces GeMoMa's real attribute order exactly."""
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [
        ("Chr1", "gene", 383, 1444, "Name=Ath_00001;ID=gene_0;transcripts=1"),
        ("Chr1", "gene", 7017, 7202, "Name=Ath_00002;ID=gene_53;transcripts=1"),
    ])
    final = tmp_path / "final.gff3"
    # A genuinely complete run: the model was prompted from both loci and emitted both,
    # GM= carrying the host's real first-attribute-value derivation (Name=, not ID=).
    _write_gff(final, [
        ("Chr1", "gene", 383, 1444, "ID=A_thaliana_g1;GM=Ath_00001"),
        ("Chr1", "gene", 7017, 7202, "ID=A_thaliana_g2;GM=Ath_00002"),
    ])
    out = tmp_path / "missing.txt"
    result = run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)
    assert result["total"] == 0
    assert out.read_text().splitlines() == ["locus\treason"]


def test_write_missing_loci_manifest_raises_if_every_locus_is_missing(tmp_path):
    """The general guard N2 also asks for: a manifest reporting 100% of loci missing is
    far more likely to be a key-derivation bug than a real total loss — check_loss would
    already have stopped the run for the latter — so this refuses to write one."""
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [("Chr1", "gene", 383, 1444, "Name=Ath_00001;ID=gene_0")])
    final = tmp_path / "final.gff3"
    # Output keyed on ID (wrong, or from a mismatched run) — GM= never matches Name=.
    _write_gff(final, [("Chr1", "gene", 383, 1444, "ID=out1;GM=gene_0")])
    out = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="key-derivation mismatch"):
        run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)


def test_write_missing_loci_manifest_raises_on_missing_subset_file(tmp_path):
    """N4: a resumed run whose _chunks/ was cleaned must not silently degrade to an
    empty (misleadingly-clean) manifest — the missing input subset is an error."""
    final = tmp_path / "final.gff3"
    _write_gff(final, [("Chr1", "gene", 1, 100, "ID=g1out;GM=g1")])
    out = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)


# ----------------------------------------------------------------------------
# chunk provenance / resume
# ----------------------------------------------------------------------------

def test_chunk_is_done_false_when_no_provenance():
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_true_when_ok_and_genes_in_matches(tmp_path):
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    # N4: chunk_is_done now also requires the chunk's input subset to still be present.
    _write_gff(run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3", [("Chr1", "gene", 1, 100, "ID=g1")])
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1,
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})
    assert run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_status_failed(tmp_path):
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "failed", "genes_in": 5})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_genes_in_has_drifted(tmp_path):
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1,
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=6)


def test_chunk_is_done_false_when_reachable_genes_in_field_missing(tmp_path):
    """R4-2: a provenance record written before round 3 has no reachable_genes_in /
    structurally_unreachable_locus at all. run_tool_pipeline now sums
    r["reachable_genes_in"] directly and would crash with a KeyError on such a record
    rather than failing loudly — treat it as not-done, forcing a clean re-run instead."""
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    _write_gff(run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3", [("Chr1", "gene", 1, 100, "ID=g1")])
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1})  # pre-round-3 shape
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_a_round_3_record_lacks_a_round_4_field(tmp_path):
    """R4-2, generalised: the round-3 record shape is the one actually on disk from a
    prior run, and it satisfies the two fields that broke first while still lacking
    `missing_loci_unexplained`, which run_tool_pipeline's chunk loop now relies on. The
    check has to be the whole REQUIRED_CHUNK_PROVENANCE_KEYS contract for that to be
    caught, not a hardcoded pair of names — a test naming the pair would pass here while
    the crash it was written to prevent still happened."""
    assert "missing_loci_unexplained" in run_bench.REQUIRED_CHUNK_PROVENANCE_KEYS
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    _write_gff(run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3", [("Chr1", "gene", 1, 100, "ID=g1")])
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1,  # round-3 shape
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1"})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_output_file_missing(tmp_path):
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1,
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_output_file_truncated_after_provenance_written(tmp_path):
    """I7: a chunk whose output shrank on disk after its "ok" provenance was written
    (e.g. a killed process, a filesystem fault) must not resume as done."""
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")  # 1 gene on disk now
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 2,  # recorded 2
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_subset_file_missing(tmp_path):
    """N4: a resumed run whose _chunks/ was cleaned must not treat a chunk as done just
    because its output and provenance survived — the missing-loci manifest needs the
    subset file too, and chunk_is_done is the single place that decides "done"."""
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1,
                           "reachable_genes_in": 4, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})
    # Deliberately no gemoma_Chr1_subset.gff3 written.
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


# ----------------------------------------------------------------------------
# get_and_verify_host_commit (I4)
# ----------------------------------------------------------------------------

def test_get_and_verify_host_commit_passes_on_pinned_commit():
    fake_ssh = _make_fake_ssh(returncode=0, stdout=f"{run_bench.EXPECTED_HOST_COMMIT}\n")
    assert run_bench.get_and_verify_host_commit(ssh_run=fake_ssh) == run_bench.EXPECTED_HOST_COMMIT


def test_get_and_verify_host_commit_raises_on_drift():
    fake_ssh = _make_fake_ssh(returncode=0, stdout="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n")
    with pytest.raises(RuntimeError, match="drifted"):
        run_bench.get_and_verify_host_commit(ssh_run=fake_ssh)


def test_get_and_verify_host_commit_raises_when_unreadable():
    fake_ssh = _make_fake_ssh(returncode=255, stderr="Connection timed out")
    with pytest.raises(RuntimeError, match="could not establish"):
        run_bench.get_and_verify_host_commit(ssh_run=fake_ssh)


def test_get_host_git_commit_reports_na_with_reason_on_failure():
    fake_ssh = _make_fake_ssh(returncode=255, stderr="Connection timed out")
    commit = run_bench.get_host_git_commit(ssh_run=fake_ssh)
    assert commit.startswith("N/A")
    assert "Connection timed out" in commit


def test_get_host_git_commit_returns_sha_on_success():
    fake_ssh = _make_fake_ssh(returncode=0, stdout="5d9929ea2189e653aac5fc7e2fef234651e96ae3\n")
    assert run_bench.get_host_git_commit(ssh_run=fake_ssh) == "5d9929ea2189e653aac5fc7e2fef234651e96ae3"


# ----------------------------------------------------------------------------
# run_remote_chunk / ensure_chunk (fully faked SSH + fetch + push)
# ----------------------------------------------------------------------------

def test_run_remote_chunk_success_path(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tgene\t200\t300\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tmRNA\t200\t300\t.\t+\t.\tID=g2.1;Parent=g2\n"
    )
    # GM= on every gene row, as real prompt_mode.py output carries it (it passes
    # f"GM={gene_models[i]}" to gffString2GFF3, which appends the extra attributes to
    # every emitted feature line) — the chunk-level R4-3 gate pairs input loci to output
    # on exactly that attribute.
    remote = FakeRemote(gff_rows_by_hint={
        "Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1"), ("Chr1", "gene", 200, 300, "ID=g2;GM=g2")],
    })
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["genes_in"] == 2
    assert result["genes_out"] == 2
    assert result["host_git_commit"] == run_bench.EXPECTED_HOST_COMMIT
    assert result["output_written"] is True
    # N6 (round 3): g2, last in file order, is the one genome2GSFDataset will never
    # flush — reachable_genes_in excludes it from the denominator this chunk's own
    # check_loss gates on.
    assert result["structurally_unreachable_locus"] == "g2"
    assert result["reachable_genes_in"] == 1
    # R4-3: g1 was prompted and came back; g2 is the known structural loss, not an
    # unexplained one. So nothing is owed an explanation here.
    assert result["missing_loci_unexplained"] == 0


def test_run_remote_chunk_treats_a_single_gene_chunk_as_a_vacuous_pass(tmp_path):
    """N6: a chunk with exactly one gene has, by construction, nothing reachable at all
    (its only gene is the structurally-unreachable last one) — this must not be
    reported as a "genes_in is 0" error, since real chromosomes never hit this (they
    have dozens to thousands of genes); it only matters for a minimal fixture, and
    should read as an unmeasurable-but-not-failed chunk."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1")]})
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["genes_in"] == 1
    assert result["reachable_genes_in"] == 0
    assert result["structurally_unreachable_locus"] == "g1"


def test_run_remote_chunk_records_noncoding_features_excluded(tmp_path):
    (run_bench.LOCAL_INPUTS / "egapx_Athaliana.gff3").write_text(
        "ChrM\tx\tgene\t100\t200\t.\t+\t.\tID=g1\n"
        "ChrM\tx\tmRNA\t100\t200\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "ChrM\tx\tpseudogene\t300\t400\t.\t+\t.\tID=p1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"egapx_ChrM_raw": [("ChrM", "gene", 100, 200, "ID=g1;GM=g1")]})
    result = run_bench.run_remote_chunk("egapx", "ChrM", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["genes_in"] == 1
    assert result["noncoding_features_excluded"] == {"pseudogene": 1}


def test_run_remote_chunk_reports_na_parsing_errors_when_summary_line_absent(tmp_path):
    """I2: prompt_mode.py only prints "Parsing errors: N/M" when error_count > 0, so a
    genuinely healthy chunk's log legitimately never contains that line either. Storing
    plain 0 there was indistinguishable from a process that crashed before printing
    anything at all — this checks the found flag and the explicit N/A string instead."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1")]},
                         log_text="Output written to: out.gff3\n")
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["parsing_errors_found"] is False
    assert result["parsing_errors_skipped"] == "N/A (summary line absent)"
    assert result["parsing_errors_total"] == "N/A (summary line absent)"


def test_run_remote_chunk_records_real_parsing_errors_when_summary_line_present(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    remote = FakeRemote(
        gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1")]},
        log_text="Output written to: out.gff3\nParsing errors: 0/1 sequences skipped\n",
    )
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["parsing_errors_found"] is True
    assert result["parsing_errors_skipped"] == 0
    assert result["parsing_errors_total"] == 1


def test_run_remote_chunk_fails_on_nonzero_ssh_returncode(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    # The host-commit check and the input-subset push both succeed (their own default
    # fake branches); only the inference command itself fails, so this specifically
    # exercises a mid-pipeline remote crash, not an earlier preflight failure.
    remote = FakeRemote(gff_rows_by_hint={}, fail_hints=("examples/prompt_mode.py",))
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "1" in result["reason"]


def test_run_remote_chunk_includes_remote_log_tail_when_command_fails(tmp_path):
    """I5: all of prompt_mode.py's stdout/stderr is redirected into the remote log
    file, so the ssh command's own stderr is empty on a real crash — without fetching
    the log itself in the failure path, the only diagnostic left is "remote command
    exited 1" with nothing after the colon."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={}, fail_hints=("examples/prompt_mode.py",),
                         log_text="Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n")
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "CUDA out of memory" in result["reason"]


def test_run_remote_chunk_fails_when_host_commit_does_not_match_pinned(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1")]},
                         host_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "drifted" in result["reason"]


def test_run_remote_chunk_fails_when_output_covers_far_fewer_loci(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tgene\t200\t300\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tmRNA\t200\t300\t.\t+\t.\tID=g2.1;Parent=g2\n"
        "Chr1\tx\tgene\t400\t500\t.\t+\t.\tID=g3\n"
        "Chr1\tx\tmRNA\t400\t500\t.\t+\t.\tID=g3.1;Parent=g3\n"
        "Chr1\tx\tgene\t600\t700\t.\t+\t.\tID=g4\n"
        "Chr1\tx\tmRNA\t600\t700\t.\t+\t.\tID=g4.1;Parent=g4\n"
    )
    # This mirrors the bs32 defect: exits 0, prints "Output written to", but the file
    # covers almost none of the input.
    remote = FakeRemote(gff_rows_by_hint={"Chr1": []})
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "lost" in result["reason"]


def test_build_prompt_mode_command_deletes_the_stale_remote_db_before_running(tmp_path):
    """R4-4: `genome2GSFDataset` appends to an existing DuckDB, so a re-run against a
    leftover `.db` regenerates every locus on top of the ones already in it. Measured on
    the real GPU host at 520c64f: a second `ensure_chunk("egapx", "ChrM")` produced 22
    gene rows for 11 distinct `GM=` values. The deletion must happen before
    prompt_mode.py is invoked, or the accumulation has already happened."""
    cmd = run_bench.build_prompt_mode_command(
        "/r/in.gff3", "/r/out.gff3", "/r/chunk.db", "/r/run.log")
    assert "rm -f /r/chunk.db /r/out.gff3" in cmd
    assert cmd.index("rm -f") < cmd.index("prompt_mode.py")


def test_run_remote_chunk_fails_when_output_has_more_rows_than_distinct_loci(tmp_path):
    """R4-4: the signature of that accumulation — more gene rows than loci — passes
    every other gate here, because they all measure loss and this is the opposite."""
    ids = [f"g{i}" for i in range(1, 6)]
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(_gff_lines(_make_gene_triples("Chr1", ids)))
    doubled = _make_gene_triples("Chr1", ids[:-1]) + _make_gene_triples("Chr1", ids[:-1], start=90000)
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": doubled})
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "8 gene rows" in result["reason"] and "4 distinct" in result["reason"]
    # Nothing else here would have objected: no locus is missing and none was lost.
    run_bench.check_loss(result["reachable_genes_in"], result["genes_out"], 0.05, context="control")


def test_run_remote_chunk_gates_on_a_prompted_locus_that_never_came_back(tmp_path):
    """R4-3 at chunk level: one locus prompted and never emitted, on a chromosome large
    enough that the ratio gate cannot see it.

    24 reachable loci, 23 emitted: 1/24 = 4.17%, under the 5% default, so `check_loss`
    passes this chunk — which is the point. A ratio is blind to single-locus losses on
    anything but a tiny chromosome (on EGAPx Chr1's 6,194 filtered loci it tolerates
    ~309), so the exact per-locus comparison is the only thing standing between a real
    generation failure and a merged result that looks complete.
    """
    ids = [f"g{i}" for i in range(1, 26)]  # g25 is last in file order -> N6 unreachable
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(_gff_lines(_make_gene_triples("Chr1", ids)))
    # Emitted: g2..g24. g1 is a genuine, unexplained miss; g25 was never reachable.
    remote = FakeRemote(gff_rows_by_hint={
        "gemoma_Chr1_raw": _make_gene_triples("Chr1", ids[1:-1]),
    })
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert result["missing_loci_unexplained"] == 1
    assert result["missing_loci_unexplained_examples"] == ["g1"]
    assert "g1" in result["reason"]
    # The ratio really did pass: 23 of 24 reachable survived, and check_loss raises
    # nothing at that rate. Without this assertion the test could not distinguish "the
    # new gate caught it" from "check_loss caught it first".
    run_bench.check_loss(result["reachable_genes_in"], result["genes_out"], 0.05, context="control")


def test_run_remote_chunk_records_zero_unexplained_misses_rather_than_omitting_them(tmp_path):
    """A number that was computed must be recorded as a number. `missing_loci_unexplained`
    stays None only when the chunk failed before its output could be compared — a clean
    chunk records 0, which is what chunk_is_done's required-key contract relies on."""
    ids = [f"g{i}" for i in range(1, 6)]
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(_gff_lines(_make_gene_triples("Chr1", ids)))
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": _make_gene_triples("Chr1", ids[:-1])})
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["missing_loci_unexplained"] == 0
    assert result["missing_loci_unexplained_examples"] == []


def test_ensure_chunk_skips_remote_call_when_already_done(tmp_path):
    # R4-1: ensure_chunk's own fingerprint is now build_local_subset's post-filter
    # count, so this fixture needs an mRNA child (otherwise g1 is a childless shell,
    # excluded, and the fresh recount would be 0, not 1 — a mismatch, not a resume).
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    # N4: chunk_is_done also requires the chunk's input subset to still be present;
    # ensure_chunk itself rebuilds it fresh before checking, so this pre-seed only
    # matters if that rebuild is ever skipped.
    _write_gff(run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3", [("Chr1", "gene", 1, 100, "ID=g1")])
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 1, "genes_out": 1,
                           "reachable_genes_in": 0, "structurally_unreachable_locus": "g1",
                           "missing_loci_unexplained": 0})

    def poison(*args, **kwargs):
        raise AssertionError("should not have called ssh/fetch/push for an already-done chunk")

    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=poison, fetch=poison, push=poison)
    assert result["resumed"] is True
    assert result["status"] == "ok"


def test_ensure_chunk_runs_when_not_done(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1;GM=g1")]})
    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["resumed"] is False
    assert result["status"] == "ok"
    assert run_bench.chunk_provenance_path("gemoma", "Chr1").exists()


def test_ensure_chunk_resumes_correctly_when_input_has_excluded_rows(tmp_path):
    """R4-1: an EGAPx-shaped input (a real gene alongside a childless gene(gbkey=lncRNA)
    shell like the one N1 excludes) previously broke resume completely — the
    fingerprint ensure_chunk recomputed on every invocation was a *raw* gene count,
    which disagreed with the *filtered* genes_in build_local_subset actually recorded
    in provenance, for any chromosome with an excluded row (always true for EGAPx,
    never for GeMoMa/BRAKER3). Every invocation looked like the input had changed, so a
    completed EGAPx chunk was silently re-run from scratch forever."""
    (run_bench.LOCAL_INPUTS / "egapx_Athaliana.gff3").write_text(
        "ChrM\tx\tgene\t100\t200\t.\t+\t.\tID=g1\n"
        "ChrM\tx\tmRNA\t100\t200\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "ChrM\tx\tgene\t300\t400\t.\t+\t.\tID=g2\n"  # childless shell (N1)
        "ChrM\tx\tlnc_RNA\t300\t400\t.\t+\t.\tID=g2.rna;Parent=g2\n"
        "ChrM\tx\texon\t300\t400\t.\t+\t.\tID=g2.rna.exon;Parent=g2.rna\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"egapx_ChrM_raw": [("ChrM", "gene", 100, 200, "ID=g1;GM=g1")]})
    first = run_bench.ensure_chunk("egapx", "ChrM", max_loss_fraction=0.05,
                                    ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert first["resumed"] is False
    assert first["status"] == "ok"
    assert first["genes_in"] == 1  # g2 correctly excluded as a childless shell

    def poison(*a, **k):
        raise AssertionError("R4-1 regression: a resumable EGAPx-shaped chunk was re-run")

    second = run_bench.ensure_chunk("egapx", "ChrM", max_loss_fraction=0.05,
                                     ssh_run=poison, fetch=poison, push=poison)
    assert second["resumed"] is True
    assert second["status"] == "ok"


# ----------------------------------------------------------------------------
# fetch_and_verify / push_and_verify checksum enforcement
# ----------------------------------------------------------------------------

def test_fetch_and_verify_success(tmp_path):
    local = tmp_path / "out.gff3"

    def fake_fetch(remote_path, local_path, host="gpu"):
        local_path.write_text("hello\n")

    good_md5 = hashlib.md5(b"hello\n").hexdigest()
    fake_ssh = _make_fake_ssh(returncode=0, stdout=f"{good_md5}  remote/out.gff3\n")
    md5 = run_bench.fetch_and_verify("remote/out.gff3", local, ssh_run=fake_ssh, fetch=fake_fetch)
    assert md5 == good_md5


def test_fetch_and_verify_raises_on_checksum_mismatch(tmp_path):
    local = tmp_path / "out.gff3"

    def fake_fetch(remote_path, local_path, host="gpu"):
        local_path.write_text("hello\n")

    fake_ssh = _make_fake_ssh(returncode=0, stdout="deadbeefdeadbeefdeadbeefdeadbeef  remote/out.gff3\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run_bench.fetch_and_verify("remote/out.gff3", local, ssh_run=fake_ssh, fetch=fake_fetch)


def test_fetch_and_verify_raises_when_local_file_never_appears(tmp_path):
    local = tmp_path / "out.gff3"
    fake_ssh = _make_fake_ssh(returncode=0, stdout="deadbeef  remote/out.gff3\n")
    with pytest.raises(RuntimeError, match="missing"):
        run_bench.fetch_and_verify("remote/out.gff3", local, ssh_run=fake_ssh, fetch=lambda *a, **k: None)


def test_push_and_verify_success(tmp_path):
    local = tmp_path / "subset.gff3"
    local.write_text("hello\n")
    good_md5 = hashlib.md5(b"hello\n").hexdigest()
    fake_ssh = _make_fake_ssh(returncode=0, stdout=f"{good_md5}  remote/subset.gff3\n")
    md5 = run_bench.push_and_verify(local, "remote/subset.gff3", ssh_run=fake_ssh, push=lambda *a, **k: None)
    assert md5 == good_md5


def test_push_and_verify_raises_on_checksum_mismatch(tmp_path):
    local = tmp_path / "subset.gff3"
    local.write_text("hello\n")
    fake_ssh = _make_fake_ssh(returncode=0, stdout="deadbeefdeadbeefdeadbeefdeadbeef  remote/subset.gff3\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run_bench.push_and_verify(local, "remote/subset.gff3", ssh_run=fake_ssh, push=lambda *a, **k: None)


# ----------------------------------------------------------------------------
# preflight checksums (genome/input — separate from the host-commit check above)
# ----------------------------------------------------------------------------

def test_verify_host_genome_passes_on_match():
    fake_ssh = _make_fake_ssh(returncode=0, stdout=f"{run_bench.EXPECTED_GENOME_MD5}  x\n")
    assert run_bench.verify_host_genome(ssh_run=fake_ssh) == run_bench.EXPECTED_GENOME_MD5


def test_verify_host_genome_raises_on_drift():
    fake_ssh = _make_fake_ssh(returncode=0, stdout="0000000000000000000000000000000  x\n")
    with pytest.raises(RuntimeError, match="drifted"):
        run_bench.verify_host_genome(ssh_run=fake_ssh)


def test_verify_host_input_passes_on_match(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    local_md5 = run_bench.md5_file(run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3")
    fake_ssh = _make_fake_ssh(returncode=0, stdout=f"{local_md5}  x\n")
    assert run_bench.verify_host_input("gemoma", ssh_run=fake_ssh) == local_md5


def test_verify_host_input_raises_on_mismatch(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    fake_ssh = _make_fake_ssh(returncode=0, stdout="0000000000000000000000000000000  x\n")
    with pytest.raises(RuntimeError, match="does not match"):
        run_bench.verify_host_input("gemoma", ssh_run=fake_ssh)


# ----------------------------------------------------------------------------
# output_stub (C2)
# ----------------------------------------------------------------------------

def test_output_stub_uses_tool_name_alone_for_the_full_genome():
    assert run_bench.output_stub("gemoma", run_bench.CHROMOSOMES) == "gemoma"


def test_output_stub_is_chromosome_qualified_for_a_subset():
    assert run_bench.output_stub("gemoma", ("ChrM",)) == "gemoma_ChrM"


def test_output_stub_is_order_insensitive_about_full_genome_membership():
    reversed_full = tuple(reversed(run_bench.CHROMOSOMES))
    assert run_bench.output_stub("gemoma", reversed_full) == "gemoma"


# ----------------------------------------------------------------------------
# full tool pipeline (fakes for SSH/fetch/push, real standardize/top-beam-filter)
# ----------------------------------------------------------------------------

def test_run_tool_pipeline_end_to_end_and_then_resumes(tmp_path, monkeypatch):
    # Treat this 2-chromosome fixture as "the full genome" so the canonical
    # (non-chromosome-qualified) output name is exercised here; C2's own naming
    # behavior for a genuine subset is covered separately below.
    monkeypatch.setattr(run_bench, "CHROMOSOMES", ("Chr1", "ChrM"))
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g1.1.cds;Parent=g1.1\n"
        "ChrM\tx\tgene\t1000\t2000\t.\t+\t.\tID=g2\n"
        "ChrM\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g2.1;Parent=g2\n"
        "ChrM\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g2.1.cds;Parent=g2.1\n"
    )

    remote = FakeRemote(gff_rows_by_hint={
        "gemoma_Chr1_raw": [
            ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
            ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
            ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ],
        "gemoma_ChrM_raw": [
            ("ChrM", "gene", 1000, 2000, "ID=g2;GM=g2"),
            ("ChrM", "mRNA", 1000, 2000, "ID=g2.1;Parent=g2"),
            ("ChrM", "CDS", 1000, 2000, "ID=g2.1.cds;Parent=g2.1"),
        ],
    })

    summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
        ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
    )
    assert summary["status"] == "ok"
    assert summary["genes_in_total"] == 2
    assert summary["genes_raw_merged"] == 2  # I1: first loss channel
    assert summary["genes_standardized"] == 2  # I1: second loss channel
    assert summary["genes_out_total"] == 2
    assert summary["missing_loci_count"] == 0
    assert summary["missing_loci_structurally_unreachable"] == 0
    assert summary["missing_loci_unexplained"] == 0
    # Each chunk here has exactly one gene, which is therefore its own last-in-file-order
    # (structurally unreachable) locus — a vacuous 0/0, not an error (see N6).
    assert summary["reachable_genes_in_total"] == 0
    final = Path(summary["final_path"])
    assert final.name == "gemoma_completed.gff3"
    assert final.exists()
    assert run_bench.count_gff3_features(final, "gene") == 2
    assert (run_bench.LOCAL_PREDICTIONS / "gemoma_provenance.json").exists()
    assert Path(summary["missing_loci_path"]).read_text().splitlines() == ["locus\treason"]

    # Re-run: every chunk is already "ok", so a poisoned ssh/fetch/push must never fire.
    def poison(*args, **kwargs):
        raise AssertionError("should not re-run a completed chunk")

    summary2 = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
        ssh_run=poison, fetch=poison, push=poison, skip_preflight=True,
    )
    assert summary2["status"] == "ok"
    assert all(c["resumed"] for c in summary2["chunks"])


def test_run_tool_pipeline_stops_on_a_failing_chunk_without_rerunning_the_good_one(tmp_path):
    (run_bench.LOCAL_INPUTS / "braker3_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "ChrM\tx\tgene\t1000\t2000\t.\t+\t.\tID=g2\n"
        "ChrM\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g2.1;Parent=g2\n"
    )
    # Chr1 succeeds, ChrM "crashes" (nonzero exit) — fail_hints matches the ChrM
    # remote command specifically, leaving Chr1 unaffected. Each fixture needs a full
    # gene/mRNA/CDS hierarchy, not a bare gene row — a locus with no CDS-bearing mRNA is
    # exactly what standardize_gff() drops ("Removed genes with no valid mRNAs"), so a
    # bare-gene fixture would make the standardize step the reason genes_out is 0, not
    # the resume logic under test.
    first_pass = FakeRemote(
        gff_rows_by_hint={"braker3_Chr1_raw": [
            ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
            ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
            ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ]},
        fail_hints=("braker3_ChrM_raw",),
    )

    with pytest.raises(run_bench.LossTooHigh):
        run_bench.run_tool_pipeline(
            "braker3", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
            ssh_run=first_pass.ssh_run, fetch=first_pass.fetch, push=first_pass.push, skip_preflight=True,
        )
    assert run_bench.chunk_is_done("braker3", "Chr1", expected_genes_in=1)
    assert not run_bench.chunk_is_done("braker3", "ChrM", expected_genes_in=1)

    # Resume: Chr1 is already "ok" so ensure_chunk must not touch it again; this fake
    # only knows how to serve ChrM, so a re-fetch of Chr1 would raise "unexpected fetch".
    second_pass = FakeRemote(gff_rows_by_hint={"braker3_ChrM_raw": [
        ("ChrM", "gene", 1000, 2000, "ID=g2;GM=g2"),
        ("ChrM", "mRNA", 1000, 2000, "ID=g2.1;Parent=g2"),
        ("ChrM", "CDS", 1000, 2000, "ID=g2.1.cds;Parent=g2.1"),
    ]})

    summary = run_bench.run_tool_pipeline(
        "braker3", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
        ssh_run=second_pass.ssh_run, fetch=second_pass.fetch, push=second_pass.push, skip_preflight=True,
    )
    assert summary["status"] == "ok"
    chunk_by_chrom = {c["chromosome"]: c for c in summary["chunks"]}
    assert chunk_by_chrom["Chr1"]["resumed"] is True
    assert chunk_by_chrom["ChrM"]["resumed"] is False


def test_run_tool_pipeline_rejects_unknown_tool():
    with pytest.raises(ValueError):
        run_bench.run_tool_pipeline("not_a_real_tool", skip_preflight=True,
                                     ssh_run=lambda a: FakeCompletedProcess(), fetch=lambda *a, **k: None)


def test_run_tool_pipeline_reports_a_clean_run_despite_the_structurally_unreachable_locus(tmp_path):
    """N6 end to end: a realistic chunk (3 genes) whose last locus, g3, is correctly
    never in the (fake, but here realistic) raw output — exactly what genome2GSFDataset
    does in reality — must read as a clean, complete run (0% loss), not "1 of 3 lost",
    and the manifest must label g3 with the structurally-unreachable reason rather than
    lumping it in with an unexplained miss."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g1.1.cds;Parent=g1.1\n"
        "Chr1\tx\tgene\t3000\t4000\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tmRNA\t3000\t4000\t.\t+\t.\tID=g2.1;Parent=g2\n"
        "Chr1\tx\tCDS\t3000\t4000\t.\t+\t0\tID=g2.1.cds;Parent=g2.1\n"
        "Chr1\tx\tgene\t5000\t6000\t.\t+\t.\tID=g3\n"
        "Chr1\tx\tmRNA\t5000\t6000\t.\t+\t.\tID=g3.1;Parent=g3\n"
        "Chr1\tx\tCDS\t5000\t6000\t.\t+\t0\tID=g3.1.cds;Parent=g3.1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 3000, 4000, "ID=g2;GM=g2"),
        ("Chr1", "mRNA", 3000, 4000, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 3000, 4000, "ID=g2.1.cds;Parent=g2.1"),
        # g3 deliberately absent, matching what genome2GSFDataset actually does to the
        # last gene of every file it processes.
    ]})
    summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05,
        ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
    )
    assert summary["status"] == "ok"
    assert summary["genes_in_total"] == 3
    assert summary["reachable_genes_in_total"] == 2
    assert summary["genes_out_total"] == 2
    assert summary["missing_loci_count"] == 1
    assert summary["missing_loci_structurally_unreachable"] == 1
    assert summary["missing_loci_unexplained"] == 0
    manifest_lines = Path(summary["missing_loci_path"]).read_text().splitlines()
    assert manifest_lines[1] == f"g3\t{run_bench.STRUCTURALLY_UNREACHABLE_REASON}"


def test_run_tool_pipeline_gates_on_a_miss_only_standardization_causes(tmp_path):
    """R4-3 at tool level, and the reason it is not redundant with the chunk-level gate.

    Every reachable locus IS generated here, so the chunk-level check passes: this loss
    happens later, when standardize_gff() drops g1 for a 2 bp CDS ("Removed mRNAs with
    CDS length < 3 bp", then "Removed genes with no valid mRNAs"). Only the manifest,
    computed from the final scored file, can see it.

    The ratio cannot: 23 of 24 reachable loci survive, 1/24 = 4.17%, under the 5%
    default. So without this gate the run would install a `_completed.gff3` that is
    quietly missing a locus, and Task 5 would score it as damage the model never did.
    """
    ids = [f"g{i}" for i in range(1, 26)]  # g25 is last in file order -> N6 unreachable
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(_gff_lines(_make_gene_triples("Chr1", ids)))

    raw_rows = _make_gene_triples("Chr1", ids[:-1])  # g1..g24: every reachable locus
    # Truncate g1's CDS to 2 bp so standardize_gff drops that gene and only that gene.
    raw_rows = [(seq, feat, s, s + 1, attrs) if feat == "CDS" and "g1.1.cds" in attrs
                else (seq, feat, s, e, attrs)
                for seq, feat, s, e, attrs in raw_rows]
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": raw_rows})

    with pytest.raises(run_bench.LossTooHigh, match="R4-3"):
        run_bench.run_tool_pipeline(
            "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05,
            ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
        )
    stub = run_bench.output_stub("gemoma", ("Chr1",))
    final_path = run_bench.LOCAL_PREDICTIONS / f"{stub}_completed.gff3"
    assert not final_path.exists()
    manifest_text = (run_bench.LOCAL_PREDICTIONS / f"{stub}_missing_loci.txt").read_text()
    assert f"g1\t{run_bench.UNEXPLAINED_MISSING_REASON}" in manifest_text
    assert f"g25\t{run_bench.STRUCTURALLY_UNREACHABLE_REASON}" in manifest_text
    # The chunk really did pass its own gate — this loss is reachable only here.
    chunk_prov = run_bench.load_json(run_bench.chunk_provenance_path("gemoma", "Chr1"))
    assert chunk_prov["status"] == "ok"
    assert chunk_prov["missing_loci_unexplained"] == 0


def test_run_tool_pipeline_records_an_acknowledged_unexplained_miss_instead_of_stopping(tmp_path):
    """The escape hatch, and the fact that using it leaves a trace.

    A run that stops at hour 18 with no way forward except editing source is its own
    failure mode, so the threshold is raisable — but only with the same acknowledgement
    C3 already requires for --max-loss-fraction, and the manifest still names the locus
    and still calls it unexplained. Nothing is silently reclassified as acceptable."""
    ids = [f"g{i}" for i in range(1, 26)]
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(_gff_lines(_make_gene_triples("Chr1", ids)))
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": _make_gene_triples("Chr1", ids[1:-1])})

    with pytest.raises(ValueError, match="acknowledge"):
        run_bench.run_tool_pipeline(
            "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05, max_unexplained_missing_loci=1,
            ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
        )

    summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05, max_unexplained_missing_loci=1,
        acknowledge_high_loss_threshold=True,
        ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
        invocation_args={"max_unexplained_missing_loci": 1, "acknowledge_high_loss_threshold": True},
    )
    assert summary["status"] == "ok"
    assert summary["missing_loci_unexplained"] == 1
    assert summary["invocation_args"]["max_unexplained_missing_loci"] == 1
    manifest_text = Path(summary["missing_loci_path"]).read_text()
    assert f"g1\t{run_bench.UNEXPLAINED_MISSING_REASON}" in manifest_text


def test_run_tool_pipeline_standardizes_away_inverted_coordinates(tmp_path):
    """I9: the original end-to-end fixture was clean enough that deleting the
    standardize_output() call and running filter_top_beam directly on the raw merge
    still passed every assertion. g2's raw CDS row here comes back with start/end
    inverted (the defect gpu-environment.md documented in raw prompt_mode.py output);
    only standardize_gff()'s coordinate-swap fixes it."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tgene\t5000\t5300\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tmRNA\t5000\t5300\t.\t+\t.\tID=g2.1;Parent=g2\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 5000, 5300, "ID=g2;GM=g2"),
        ("Chr1", "mRNA", 5000, 5300, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 5300, 5000, "ID=g2.1.cds;Parent=g2.1"),  # inverted: start > end
    ]})
    summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05,
        ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
    )
    final_text = Path(summary["final_path"]).read_text()
    checked_any = False
    for line in final_text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        checked_any = True
        assert int(cols[4]) >= int(cols[3]), f"un-standardized inverted row survived: {line}"
    assert checked_any


def test_run_tool_pipeline_atomic_write_leaves_no_completed_file_when_final_gate_fails(tmp_path):
    """C1: the handoff file must never be written before the gate that validates it.

    Three loci, so the N6 structurally-unreachable exclusion (always the *last* gene in
    file order — g3 here) lands on a gene distinct from the one this test wants
    standardize_gff to drop for a genuinely different, content-based reason (g2's 2 bp
    CDS — "Removed mRNAs with CDS length < 3 bp", then "Removed genes with no valid
    mRNAs"). g3 is correctly never in the fake's raw output at all (a real run could
    never generate it either), so the per-chunk gate sees 2 reachable / 2 produced (g1,
    g2) and passes; only the final gate, after g2 is dropped by standardize_gff, catches
    the real loss standardize_gff (not N6) causes."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tgene\t3000\t4000\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tmRNA\t3000\t4000\t.\t+\t.\tID=g2.1;Parent=g2\n"
        "Chr1\tx\tgene\t5000\t6000\t.\t+\t.\tID=g3\n"
        "Chr1\tx\tmRNA\t5000\t6000\t.\t+\t.\tID=g3.1;Parent=g3\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 3000, 4000, "ID=g2;GM=g2"),
        ("Chr1", "mRNA", 3000, 4000, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 3000, 3001, "ID=g2.1.cds;Parent=g2.1"),  # 2 bp: non-translatable
        # g3 deliberately absent: it's last in file order, so N6 says a real run could
        # never have produced it either.
    ]})
    with pytest.raises(run_bench.LossTooHigh):
        run_bench.run_tool_pipeline(
            "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05,
            ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
        )
    stub = run_bench.output_stub("gemoma", ("Chr1",))
    final_path = run_bench.LOCAL_PREDICTIONS / f"{stub}_completed.gff3"
    partial_path = run_bench.LOCAL_PREDICTIONS / f"{stub}_completed.gff3.partial"
    assert not final_path.exists()
    assert partial_path.exists()  # left behind as a diagnostic artifact, per C1's design


def test_run_tool_pipeline_installs_nothing_when_the_manifest_step_fails(tmp_path):
    """N3: the manifest (and provenance) must be built before the canonical file is
    installed, not after — a failure in *that* step (not just check_loss) must also
    leave no canonical {tool}_completed.gff3. Provokes the manifest's own 100%-missing
    guard (a real key-derivation regression, not the ordinary check_loss path) to prove
    the reordering actually protects this window and not just the one C1 already
    covered."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tName=Ath_00001;ID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g1.1.cds;Parent=g1.1\n"
    )
    # check_loss passes cleanly (1 gene in, 1 out) — but GM= doesn't match the input's
    # first-attribute value (Ath_00001), simulating a hypothetical regression in how
    # prompt_mode stamps GM=, which the manifest's guard must catch on its own.
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1out;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1out.1;Parent=g1out"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1out.1.cds;Parent=g1out.1"),
    ]})
    with pytest.raises(RuntimeError, match="key-derivation mismatch"):
        run_bench.run_tool_pipeline(
            "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.05,
            ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
        )
    stub = run_bench.output_stub("gemoma", ("Chr1",))
    final_path = run_bench.LOCAL_PREDICTIONS / f"{stub}_completed.gff3"
    assert not final_path.exists()
    assert not (run_bench.LOCAL_PREDICTIONS / f"{stub}_provenance.json").exists()


def test_run_tool_pipeline_with_chromosome_subset_does_not_overwrite_full_genome_result(tmp_path, monkeypatch):
    """C2: a --chromosomes ChrM-shaped invocation run after a completed full-genome run
    must not silently overwrite the canonical {tool}_completed.gff3."""
    monkeypatch.setattr(run_bench, "CHROMOSOMES", ("Chr1", "ChrM"))
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
        "Chr1\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g1.1.cds;Parent=g1.1\n"
        "ChrM\tx\tgene\t1000\t2000\t.\t+\t.\tID=g2\n"
        "ChrM\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g2.1;Parent=g2\n"
        "ChrM\tx\tCDS\t1000\t2000\t.\t+\t0\tID=g2.1.cds;Parent=g2.1\n"
    )
    full_remote = FakeRemote(gff_rows_by_hint={
        "gemoma_Chr1_raw": [
            ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
            ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
            ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ],
        "gemoma_ChrM_raw": [
            ("ChrM", "gene", 1000, 2000, "ID=g2;GM=g2"),
            ("ChrM", "mRNA", 1000, 2000, "ID=g2.1;Parent=g2"),
            ("ChrM", "CDS", 1000, 2000, "ID=g2.1.cds;Parent=g2.1"),
        ],
    })
    full_summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
        ssh_run=full_remote.ssh_run, fetch=full_remote.fetch, push=full_remote.push, skip_preflight=True,
    )
    full_final = Path(full_summary["final_path"])
    assert full_final.name == "gemoma_completed.gff3"
    assert run_bench.count_gff3_features(full_final, "gene") == 2
    full_final_bytes = full_final.read_bytes()

    def poison(*a, **k):
        raise AssertionError("ChrM chunk already succeeded and must not be re-run")

    subset_summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("ChrM",), max_loss_fraction=0.05,
        ssh_run=poison, fetch=poison, push=poison, skip_preflight=True,
    )
    subset_final = Path(subset_summary["final_path"])
    assert subset_final.name == "gemoma_ChrM_completed.gff3"
    assert subset_final != full_final
    assert full_final.exists()
    assert full_final.read_bytes() == full_final_bytes
    assert run_bench.count_gff3_features(full_final, "gene") == 2


def test_run_tool_pipeline_refuses_high_loss_threshold_without_acknowledgement():
    with pytest.raises(ValueError, match="acknowledge_high_loss_threshold"):
        run_bench.run_tool_pipeline("gemoma", max_loss_fraction=0.5, skip_preflight=True,
                                     ssh_run=lambda a: FakeCompletedProcess(), fetch=lambda *a, **k: None)


def test_run_tool_pipeline_records_invocation_args_and_high_loss_acknowledgement(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tmRNA\t1000\t2000\t.\t+\t.\tID=g1.1;Parent=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
    ]})
    summary = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1",), max_loss_fraction=0.5,
        acknowledge_high_loss_threshold=True,
        invocation_args={"max_loss_fraction": 0.5, "acknowledge_high_loss_threshold": True, "force": False},
        ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push, skip_preflight=True,
    )
    assert summary["invocation_args"]["max_loss_fraction"] == 0.5
    assert summary["invocation_args"]["acknowledge_high_loss_threshold"] is True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def test_main_returns_1_and_still_attempts_other_tools_when_one_fails(monkeypatch):
    attempted = []

    def fake_run_tool_pipeline(tool, **kwargs):
        attempted.append(tool)
        if tool == "braker3":
            raise run_bench.LossTooHigh("simulated failure")
        return {"tool": tool, "status": "ok", "chunks": []}

    monkeypatch.setattr(run_bench, "run_tool_pipeline", fake_run_tool_pipeline)
    rc = run_bench.main(["--tool", "all", "--skip-preflight"])
    assert rc == 1
    assert set(attempted) == {"gemoma", "braker3", "egapx"}


def test_main_returns_0_when_all_tools_succeed(monkeypatch):
    monkeypatch.setattr(run_bench, "run_tool_pipeline",
                         lambda tool, **kwargs: {"tool": tool, "status": "ok", "chunks": []})
    rc = run_bench.main(["--tool", "gemoma", "--skip-preflight"])
    assert rc == 0


def test_main_passes_acknowledge_flag_and_invocation_args_through(monkeypatch):
    captured = {}

    def fake_run_tool_pipeline(tool, **kwargs):
        captured[tool] = kwargs
        return {"tool": tool, "status": "ok", "chunks": []}

    monkeypatch.setattr(run_bench, "run_tool_pipeline", fake_run_tool_pipeline)
    rc = run_bench.main(["--tool", "gemoma", "--skip-preflight",
                          "--max-loss-fraction", "0.5", "--acknowledge-high-loss-threshold"])
    assert rc == 0
    assert captured["gemoma"]["acknowledge_high_loss_threshold"] is True
    assert captured["gemoma"]["invocation_args"]["max_loss_fraction"] == 0.5


def test_main_defaults_unexplained_missing_loci_to_zero_and_passes_an_override_through(monkeypatch):
    """R4-3: the CLI must not be able to loosen the exact per-locus gate by omission,
    and an override must reach both the pipeline and the recorded invocation_args."""
    captured = {}

    def fake_run_tool_pipeline(tool, **kwargs):
        captured[tool] = kwargs
        return {"tool": tool, "status": "ok", "chunks": []}

    monkeypatch.setattr(run_bench, "run_tool_pipeline", fake_run_tool_pipeline)

    run_bench.main(["--tool", "gemoma", "--skip-preflight"])
    assert captured["gemoma"]["max_unexplained_missing_loci"] == 0

    run_bench.main(["--tool", "braker3", "--skip-preflight",
                     "--max-unexplained-missing-loci", "3", "--acknowledge-high-loss-threshold"])
    assert captured["braker3"]["max_unexplained_missing_loci"] == 3
    assert captured["braker3"]["invocation_args"]["max_unexplained_missing_loci"] == 3
