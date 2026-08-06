"""Tests for 33_run_polishing_benchmark.py.

None of these tests touch SSH, the GPU host, or a GPU: every function that talks to the
network is injectable (`ssh_run`, `fetch`) and every test here supplies a fake. The two
functions that wrap real project code (`standardize_output`, `filter_top_beam`) are
exercised against the real `transgenic_comparison/standardize_gff.py` and
`13_beam1_filter.py`, but only against tiny synthetic fixtures written to `tmp_path`.
"""

import importlib.util
import json
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
    path.write_text(_gff_lines(rows))


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRemote:
    """A fake ssh_run + fetch pair that share enough state to make the
    checksum-verification round trip in `fetch_and_verify` self-consistent: `fetch()`
    really writes the local file, and the `md5sum <remote>` ssh branch reports that same
    file's real md5 back — exactly what a real host would report after a real rsync, so
    tests never need to hand-compute a matching hash.

    `gff_rows_by_hint`: {substring-of-remote-path: rows} — picks which fixture rows to
    write for a given remote output path. A remote path (or log) matching no hint raises,
    so an unexpected fetch call fails the test instead of silently succeeding.

    `fail_hints`: remote commands containing any of these substrings get a nonzero
    return code instead of the canned success response (simulates a remote crash for
    that one chunk without touching the others).
    """

    def __init__(self, gff_rows_by_hint, fail_hints=(),
                 log_text="Output written to: out.gff3\n",
                 prompt_stdout="Output written to: out.gff3\n"):
        self.gff_rows_by_hint = gff_rows_by_hint
        self.fail_hints = tuple(fail_hints)
        self.log_text = log_text
        self.prompt_stdout = prompt_stdout
        self._written = {}

    def ssh_run(self, argv):
        cmd = argv[-1]
        if cmd.startswith("md5sum "):
            remote_path = cmd[len("md5sum "):].strip()
            local_path = self._written.get(remote_path)
            if local_path is None or not local_path.exists():
                return FakeCompletedProcess(returncode=1, stderr=f"no such file: {remote_path}")
            return FakeCompletedProcess(returncode=0, stdout=f"{run_bench.md5_file(local_path)}  {remote_path}\n")
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
    import hashlib
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


def test_build_chromosome_subset_command_uses_awk_on_column_one():
    cmd = run_bench.build_chromosome_subset_command("in.gff3", "Chr1", "sub.gff3")
    assert "awk" in cmd
    assert "$1==c" in cmd
    assert "Chr1" in cmd


def test_build_chunk_remote_command_paths_are_tool_and_chrom_specific():
    paths = run_bench.build_chunk_remote_command("gemoma", "ChrM")
    assert paths["remote_input"].endswith("gemoma_Athaliana.gff3")
    assert "ChrM" in paths["remote_output"]
    assert "gemoma" in paths["remote_output"]
    assert "--batch-size 1" in paths["command"]


def test_build_ssh_argv_shape():
    argv = run_bench.build_ssh_argv("echo hi", host="gpu")
    assert argv == ["ssh", "gpu", "echo hi"]


def test_build_rsync_fetch_argv_shape(tmp_path):
    argv = run_bench.build_rsync_fetch_argv("~/x.gff3", tmp_path / "x.gff3", host="gpu")
    assert argv[0] == "rsync"
    assert argv[-2] == "gpu:~/x.gff3"


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
# ----------------------------------------------------------------------------

def test_standardize_output_drops_out_of_bounds_coordinates(tmp_path):
    raw = tmp_path / "raw.gff3"
    # Chr1 is 30,427,671 bp in the real TAIR10 .fai; the second gene is nonsense.
    _write_gff(raw, [
        ("Chr1", "gene", 100, 200, "ID=g1"),
        ("Chr1", "mRNA", 100, 200, "ID=g1.1;Parent=g1"),
        ("Chr1", "CDS", 100, 200, "ID=g1.1.cds;Parent=g1.1"),
        ("Chr1", "gene", 999_000_000, 999_000_100, "ID=g2"),
    ])
    out = tmp_path / "std.gff3"
    run_bench.standardize_output(raw, out, "gemoma")
    assert out.exists()
    text = out.read_text()
    assert "999000000" not in text
    assert run_bench.count_gff3_features(out, "gene") == 1


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


# ----------------------------------------------------------------------------
# run_remote_chunk / ensure_chunk (fully faked SSH + fetch)
# ----------------------------------------------------------------------------

def _make_fake_ssh(returncode=0, stdout="Output written to: out.gff3\n", stderr=""):
    def fake_ssh_run(argv):
        return FakeCompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr)
    return fake_ssh_run


def test_run_remote_chunk_success_path(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Chr1\tx\tgene\t200\t300\t.\t+\t.\tID=g2\n"
    )
    remote = FakeRemote(gff_rows_by_hint={
        "Chr1": [("Chr1", "gene", 1, 100, "ID=g1"), ("Chr1", "gene", 200, 300, "ID=g2")],
    })
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch)
    assert result["status"] == "ok"
    assert result["genes_in"] == 2
    assert result["genes_out"] == 2
    assert result["parsing_errors_skipped"] == 0
    assert result["output_written"] is True


def test_run_remote_chunk_records_parsing_errors_from_log(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    remote = FakeRemote(
        gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]},
        log_text="Output written to: out.gff3\nParsing errors: 0/1 sequences skipped\n",
    )
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=remote.ssh_run, fetch=remote.fetch)
    assert result["status"] == "ok"
    assert result["parsing_errors_skipped"] == 0
    assert result["parsing_errors_total"] == 1


def test_run_remote_chunk_fails_on_nonzero_ssh_returncode(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    fake_ssh = _make_fake_ssh(returncode=1, stderr="CUDA out of memory")
    result = run_bench.run_remote_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                         ssh_run=fake_ssh, fetch=lambda *a, **k: None)
    assert result["status"] == "failed"
    assert "1" in result["reason"]


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
                                         ssh_run=remote.ssh_run, fetch=remote.fetch)
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

    def poison_ssh_run(argv):
        raise AssertionError("should not have called ssh for an already-done chunk")

    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=poison_ssh_run, fetch=poison_ssh_run)
    assert result["resumed"] is True
    assert result["status"] == "ok"


def test_ensure_chunk_runs_when_not_done(tmp_path):
    (run_bench.LOCAL_INPUTS / "gemoma_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
    )
    remote = FakeRemote(gff_rows_by_hint={"Chr1": [("Chr1", "gene", 1, 100, "ID=g1")]})
    result = run_bench.ensure_chunk("gemoma", "Chr1", max_loss_fraction=0.05,
                                     ssh_run=remote.ssh_run, fetch=remote.fetch)
    assert result["resumed"] is False
    assert result["status"] == "ok"
    assert run_bench.chunk_provenance_path("gemoma", "Chr1").exists()


# ----------------------------------------------------------------------------
# fetch_and_verify checksum enforcement
# ----------------------------------------------------------------------------

def test_fetch_and_verify_success(tmp_path):
    local = tmp_path / "out.gff3"

    def fake_fetch(remote_path, local_path, host="gpu"):
        local_path.write_text("hello\n")

    import hashlib
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


# ----------------------------------------------------------------------------
# preflight checksums
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


def test_get_host_git_commit_reports_na_with_reason_on_failure():
    fake_ssh = _make_fake_ssh(returncode=255, stderr="Connection timed out")
    commit = run_bench.get_host_git_commit(ssh_run=fake_ssh)
    assert commit.startswith("N/A")
    assert "Connection timed out" in commit


def test_get_host_git_commit_returns_sha_on_success():
    fake_ssh = _make_fake_ssh(returncode=0, stdout="5d9929ea2189e653aac5fc7e2fef234651e96ae3\n")
    assert run_bench.get_host_git_commit(ssh_run=fake_ssh) == "5d9929ea2189e653aac5fc7e2fef234651e96ae3"


# ----------------------------------------------------------------------------
# full tool pipeline (fakes for SSH/fetch, real standardize/top-beam-filter)
# ----------------------------------------------------------------------------

def test_run_tool_pipeline_end_to_end_and_then_resumes(tmp_path):
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
        ssh_run=remote.ssh_run, fetch=remote.fetch, skip_preflight=True,
    )
    assert summary["status"] == "ok"
    assert summary["genes_in_total"] == 2
    assert summary["genes_out_total"] == 2
    final = Path(summary["final_path"])
    assert final.exists()
    assert run_bench.count_gff3_features(final, "gene") == 2
    assert (run_bench.LOCAL_PREDICTIONS / "gemoma_provenance.json").exists()

    # Re-run: every chunk is already "ok", so a poisoned ssh/fetch must never fire.
    def poison(*args, **kwargs):
        raise AssertionError("should not re-run a completed chunk")

    summary2 = run_bench.run_tool_pipeline(
        "gemoma", chromosomes=("Chr1", "ChrM"), max_loss_fraction=0.05,
        ssh_run=poison, fetch=poison, skip_preflight=True,
    )
    assert summary2["status"] == "ok"
    assert all(c["resumed"] for c in summary2["chunks"])


def test_run_tool_pipeline_stops_on_a_failing_chunk_without_rerunning_the_good_one(tmp_path):
    (run_bench.LOCAL_INPUTS / "braker3_Athaliana.gff3").write_text(
        "Chr1\tx\tgene\t1000\t2000\t.\t+\t.\tID=g1\n"
        "ChrM\tx\tgene\t1000\t2000\t.\t+\t.\tID=g2\n"
    )
    # Chr1 succeeds, ChrM "crashes" (nonzero exit) — fail_hints matches the ChrM
    # remote command specifically, leaving Chr1 unaffected.
    # Each fixture needs a full gene/mRNA/CDS hierarchy, not a bare gene row — a locus
    # with no CDS-bearing mRNA is exactly what standardize_gff() is supposed to drop
    # ("Removed genes with no valid mRNAs"), so a bare-gene fixture would make the
    # standardize step itself the reason genes_out is 0, not the resume logic under test.
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
            ssh_run=first_pass.ssh_run, fetch=first_pass.fetch, skip_preflight=True,
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
        ssh_run=second_pass.ssh_run, fetch=second_pass.fetch, skip_preflight=True,
    )
    assert summary["status"] == "ok"
    chunk_by_chrom = {c["chromosome"]: c for c in summary["chunks"]}
    assert chunk_by_chrom["Chr1"]["resumed"] is True
    assert chunk_by_chrom["ChrM"]["resumed"] is False


def test_run_tool_pipeline_rejects_unknown_tool():
    with pytest.raises(ValueError):
        run_bench.run_tool_pipeline("not_a_real_tool", skip_preflight=True,
                                     ssh_run=lambda a: FakeCompletedProcess(), fetch=lambda *a, **k: None)


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
