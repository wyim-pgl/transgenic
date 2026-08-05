import importlib.util, sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stage", Path(__file__).resolve().parents[1] / "30_stage_external_annotations.py")
stage_mod = importlib.util.module_from_spec(spec)
sys.modules["stage"] = stage_mod
spec.loader.exec_module(stage_mod)


def test_normalise_strips_species_prefix():
    assert stage_mod.normalise_seqid("Ath_Chr1") == "Chr1"


def test_normalise_leaves_tair_names_alone():
    assert stage_mod.normalise_seqid("Chr1") == "Chr1"
    assert stage_mod.normalise_seqid("ChrM") == "ChrM"


def test_normalise_maps_bare_numbers():
    assert stage_mod.normalise_seqid("1") == "Chr1"


def test_stage_drops_unplaced_scaffolds(tmp_path):
    src = tmp_path / "in.gff3"
    src.write_text(
        "Ath_Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "Ath_Chr1\tx\tmRNA\t1\t100\t.\t+\t.\tID=m1;Parent=g1\n"
        "Ath_Chr1\tx\tCDS\t1\t100\t.\t+\t0\tID=c1;Parent=m1\n"
        "scaffold_7\tx\tgene\t1\t100\t.\t+\t.\tID=g2\n"
    )
    dst = tmp_path / "out.gff3"
    meta = stage_mod.stage("test", src, dst)
    text = dst.read_text()
    assert "Chr1\t" in text
    assert "Ath_Chr1" not in text
    assert "scaffold_7" not in text
    assert meta["genes"] == 1
    assert meta["dropped_seqids"] == ["scaffold_7"]


def _gff_lines(seqid_gene_counts: dict) -> str:
    """Build minimal GFF3 text with `n` distinct gene lines on each given seqid."""
    lines = []
    i = 0
    for seqid, n in seqid_gene_counts.items():
        for _ in range(n):
            i += 1
            lines.append(f"{seqid}\tx\tgene\t1\t100\t.\t+\t.\tID=g{i}\n")
    return "".join(lines)


# -- validate(): the fail-loud guards main() relies on ---------------------


def test_validate_passes_for_healthy_meta():
    meta = {
        "tool": "healthy",
        "source": "x",
        "genes": 20001,
        "transcripts": 1,
        "seqids": sorted(stage_mod.TAIR_SEQIDS),
        "dropped_seqids": [],
    }
    stage_mod.validate("healthy", meta)  # must not raise


def test_validate_rejects_unrecognised_seqids(tmp_path):
    src = tmp_path / "in.gff3"
    src.write_text(
        "scaffold_1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "scaffold_2\tx\tgene\t1\t100\t.\t+\t.\tID=g2\n"
    )
    dst = tmp_path / "out.gff3"
    meta = stage_mod.stage("badseq", src, dst)
    assert meta["seqids"] == []
    with pytest.raises(AssertionError, match="do not exactly match TAIR10"):
        stage_mod.validate("badseq", meta)


def test_validate_rejects_missing_chromosome(tmp_path):
    present = sorted(stage_mod.TAIR_SEQIDS - {"ChrM"})
    src = tmp_path / "in.gff3"
    src.write_text(_gff_lines({s: 1 for s in present}))
    dst = tmp_path / "out.gff3"
    meta = stage_mod.stage("missingchr", src, dst)
    assert "ChrM" not in meta["seqids"]
    with pytest.raises(AssertionError, match=r"missing=\['ChrM'\]"):
        stage_mod.validate("missingchr", meta)


def test_validate_rejects_too_few_genes(tmp_path):
    src = tmp_path / "in.gff3"
    src.write_text(_gff_lines({s: 1 for s in stage_mod.TAIR_SEQIDS}))
    dst = tmp_path / "out.gff3"
    meta = stage_mod.stage("toosmall", src, dst)
    assert set(meta["seqids"]) == stage_mod.TAIR_SEQIDS
    assert not meta["dropped_seqids"]
    with pytest.raises(AssertionError, match="only 7 genes staged"):
        stage_mod.validate("toosmall", meta)


def test_validate_rejects_nonempty_dropped_seqids():
    meta = {
        "tool": "haddrops",
        "source": "x",
        "genes": 30000,
        "transcripts": 1,
        "seqids": sorted(stage_mod.TAIR_SEQIDS),
        "dropped_seqids": ["scaffold_9"],
    }
    with pytest.raises(AssertionError, match="dropped during"):
        stage_mod.validate("haddrops", meta)


# -- main(): abort before writing anything when a source is missing --------


def test_main_aborts_before_writing_when_a_source_is_missing(tmp_path, monkeypatch, capsys):
    present_src = tmp_path / "present.gff3"
    present_src.write_text("Chr1\tx\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    missing_src = tmp_path / "does_not_exist.gff3"
    out = tmp_path / "out"

    monkeypatch.setattr(stage_mod, "SOURCES", {
        "present": present_src,
        "missing": missing_src,
    })
    monkeypatch.setattr(sys, "argv", ["prog", "--out", str(out)])

    rc = stage_mod.main()

    assert rc == 1
    assert not (out / "present_Athaliana.gff3").exists()
    assert not (out / "provenance.json").exists()
    captured = capsys.readouterr()
    assert "missing" in captured.err
    assert str(missing_src) in captured.err
