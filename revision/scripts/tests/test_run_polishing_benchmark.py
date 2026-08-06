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


def test_count_genes_for_chromosome_filters_by_seqid(tmp_path):
    p = tmp_path / "a.gff3"
    _write_gff(p, [
        ("Chr1", "gene", 1, 100, "ID=g1"),
        ("Chr2", "gene", 1, 100, "ID=g2"),
        ("Chr1", "gene", 200, 300, "ID=g3"),
    ])
    assert run_bench.count_genes_for_chromosome(p, "Chr1") == 2
    assert run_bench.count_genes_for_chromosome(p, "Chr2") == 1
    assert run_bench.count_genes_for_chromosome(p, "ChrM") == 0


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


def test_build_local_subset_only_includes_requested_chromosome(tmp_path):
    inp = tmp_path / "gemoma_Athaliana.gff3"
    _write_gff(inp, [
        ("Chr1", "gene", 1, 100, "ID=g1"),
        ("Chr2", "gene", 1, 100, "ID=g2"),
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
    n = run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)
    assert n == 1
    assert out.read_text().strip() == "g2"


def test_write_missing_loci_manifest_empty_when_nothing_missing(tmp_path):
    subset1 = run_bench.LOCAL_CHUNKS / "gemoma_Chr1_subset.gff3"
    _write_gff(subset1, [("Chr1", "gene", 1, 100, "ID=g1")])
    final = tmp_path / "final.gff3"
    _write_gff(final, [("Chr1", "gene", 1, 100, "ID=g1out;GM=g1")])
    out = tmp_path / "missing.txt"
    n = run_bench.write_missing_loci_manifest([{"tool": "gemoma", "chromosome": "Chr1"}], final, out)
    assert n == 0
    assert out.read_text() == ""


# ----------------------------------------------------------------------------
# chunk provenance / resume
# ----------------------------------------------------------------------------

def test_chunk_is_done_false_when_no_provenance():
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_true_when_ok_and_genes_in_matches(tmp_path):
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1})
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
                          {"status": "ok", "genes_in": 5, "genes_out": 1})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=6)


def test_chunk_is_done_false_when_output_file_missing(tmp_path):
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 1})
    assert not run_bench.chunk_is_done("gemoma", "Chr1", expected_genes_in=5)


def test_chunk_is_done_false_when_output_file_truncated_after_provenance_written(tmp_path):
    """I7: a chunk whose output shrank on disk after its "ok" provenance was written
    (e.g. a killed process, a filesystem fault) must not resume as done."""
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")  # 1 gene on disk now
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 5, "genes_out": 2})  # recorded 2
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
        "Chr1\tx\tgene\t200\t300\t.\t+\t.\tID=g2\n"
    )
    remote = FakeRemote(gff_rows_by_hint={
        "Chr1": [("Chr1", "gene", 1, 100, "ID=g1"), ("Chr1", "gene", 200, 300, "ID=g2")],
    })
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "ok"
    assert result["genes_in"] == 2
    assert result["genes_out"] == 2
    assert result["host_git_commit"] == run_bench.EXPECTED_HOST_COMMIT
    assert result["output_written"] is True


def test_run_remote_chunk_records_noncoding_features_excluded(tmp_path):
    (run_bench.LOCAL_INPUTS / "egapx_Athaliana.gff3").write_text(
        "ChrM\tx\tgene\t100\t200\t.\t+\t.\tID=g1\n"
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
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]},
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
    )
    remote = FakeRemote(
        gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]},
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
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]},
                         host_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "drifted" in result["reason"]


def test_run_remote_chunk_fails_when_output_covers_far_fewer_loci(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tgene\t200\t300\t.\t+\t.\tID=g2\n"
        "Chr1\tx\tgene\t400\t500\t.\t+\t.\tID=g3\n"
        "Chr1\tx\tgene\t600\t700\t.\t+\t.\tID=g4\n"
    )
    # This mirrors the bs32 defect: exits 0, prints "Output written to", but the file
    # covers almost none of the input.
    remote = FakeRemote(gff_rows_by_hint={"Chr1": []})
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["status"] == "failed"
    assert "lost" in result["reason"]


def test_ensure_chunk_skips_remote_call_when_already_done(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    out = run_bench.chunk_output_path("gemoma", "Chr1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    run_bench.write_json(run_bench.chunk_provenance_path("gemoma", "Chr1"),
                          {"status": "ok", "genes_in": 1, "genes_out": 1})

    def poison(*args, **kwargs):
        raise AssertionError("should not have called ssh/fetch/push for an already-done chunk")

    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=poison, fetch=poison, push=poison)
    assert result["resumed"] is True
    assert result["status"] == "ok"


def test_ensure_chunk_runs_when_not_done(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]})
    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=remote.ssh_run, fetch=remote.fetch, push=remote.push)
    assert result["resumed"] is False
    assert result["status"] == "ok"
    assert run_bench.chunk_provenance_path("gemoma", "Chr1").exists()


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
    final = Path(summary["final_path"])
    assert final.name == "gemoma_completed.gff3"
    assert final.exists()
    assert run_bench.count_gff3_features(final, "gene") == 2
    assert (run_bench.LOCAL_PREDICTIONS / "gemoma_provenance.json").exists()
    assert Path(summary["missing_loci_path"]).read_text() == ""

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
        "ChrM\tx\tgene\t1000\t2000\t.\t+\t.\tID=g2\n"
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


def test_run_tool_pipeline_standardizes_away_inverted_coordinates(tmp_path):
    """I9: the original end-to-end fixture was clean enough that deleting the
    standardize_output() call and running filter_top_beam directly on the raw merge
    still passed every assertion. g2's raw CDS row here comes back with start/end
    inverted (the defect gpu-environment.md documented in raw prompt_mode.py output);
    only standardize_gff()'s coordinate-swap fixes it."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tgene\t5000\t5300\t.\t+\t.\tID=g2\n"
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
    Both loci come back with a full gene/mRNA/CDS row, so the *per-chunk* gate (2/2,
    checked against the raw fetched output) passes; g2's 2 bp CDS is then legitimately
    dropped by standardize_gff() as non-translatable ("Removed mRNAs with CDS length <
    3 bp", then "Removed genes with no valid mRNAs") — a loss the chunk-level check
    cannot see, which is exactly what should trip the *final* gate."""
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tgene\t3000\t4000\t.\t+\t.\tID=g2\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"gemoma_Chr1_raw": [
        ("Chr1", "gene", 1000, 2000, "ID=g1;GM=g1"),
        ("Chr1", "mRNA", 1000, 2000, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 1000, 2000, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 3000, 4000, "ID=g2;GM=g2"),
        ("Chr1", "mRNA", 3000, 4000, "ID=g2.1;Parent=g2"),
        ("Chr1", "CDS", 3000, 3001, "ID=g2.1.cds;Parent=g2.1"),  # 2 bp: non-translatable
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
