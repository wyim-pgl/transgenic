import importlib.util, sys
from pathlib import Path

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
