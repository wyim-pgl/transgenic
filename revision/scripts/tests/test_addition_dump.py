"""Protect frozen addition selection, coordinate conversion and neutral ambiguities."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import addition_dump as dump

spec = importlib.util.spec_from_file_location("score48", SCRIPTS / "48_score_zmays_additions.py")
score48 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score48)


def test_collapse_primary_chain_and_novelty():
    primary = ((1, 3), (10, 12))
    alternate = ((1, 3), (20, 22))
    new = ((1, 3), (30, 32))
    boundary = ((2, 3), (10, 12))
    pred = {"g": {"p": primary, "a": alternate, "dup": alternate, "n": new, "b": boundary}}
    ref = {"g": {"p": primary, "a": alternate}}
    metadata = {t: ("1", "+") for t in pred["g"]}
    rows = dump.structure_rows(pred, ref, {"g": "p"}, metadata, {"1": "A" * 100}, "sp", dump.load_filter())
    assert len(rows) == score48.score(pred, ref, {"g": "p"}, ["g"], "fixture")["added_transcripts"] == 3
    by_cds = {r["cds_intervals"]: r for r in rows}
    assert by_cds[alternate]["prediction_transcript_ids"] == ["a", "dup"]
    assert by_cds[alternate]["novelty"] == "reference-alt"
    assert by_cds[new]["introns"] == ((4, 29),)
    assert by_cds[new]["novelty"] == "junction-novel"
    assert by_cds[boundary]["novelty"] is None
    assert by_cds[boundary]["classification_ambiguity"]
    reversed_pred = {"g": dict(reversed(list(pred["g"].items())))}
    assert rows == dump.structure_rows(reversed_pred, ref, {"g": "p"}, metadata, {"1": "A" * 100}, "sp", dump.load_filter())


@pytest.mark.parametrize("strand,sequence", [("+", "ATGTAA"), ("-", "TTACAT")])
def test_frozen_filter_strand_symmetry(strand, sequence):
    rows = dump.structure_rows({"g": {"a": ((1, 6),)}}, {"g": {"p": ((1, 3),)}},
                               {"g": "p"}, {"a": ("1", strand)}, {"1": sequence}, "sp", dump.load_filter())
    assert rows[0]["filter_pass"]
    assert rows[0]["novelty"] is None  # no invented monoexon category


def test_first_record_kept_even_if_later_record_disagrees(tmp_path):
    path = tmp_path / "prediction.gff3"
    lines = []
    for i, end in [(1, 9), (2, 18)]:
        lines += [f"1\tx\tgene\t1\t{end}\t.\t+\t.\tID=g{i};GM=locus",
                  f"1\tx\tmRNA\t1\t{end}\t.\t+\t.\tID=t{i};Parent=g{i}",
                  f"1\tx\tCDS\t1\t{end}\t.\t+\t0\tParent=t{i}"]
    path.write_text("\n".join(lines) + "\n")
    pred, audit = score48.read_prediction(path)
    assert pred == {"locus": {"t1": ((1, 9),)}}
    assert audit["duplicate_records_disagreeing_with_kept_record"] == 1


def test_missing_sequence_is_error():
    with pytest.raises(ValueError, match="missing genome"):
        dump.structure_rows({"g": {"a": ((1, 6),)}}, {}, {}, {"a": ("missing", "+")}, {}, "sp", dump.load_filter())


def test_never_overwrite_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        dump.write_frozen_dump(tmp_path, score48)


def test_indexed_fasta_matches_string_slices(tmp_path):
    path = tmp_path / "genome.fa"
    path.write_bytes(b">1\nATGT\nAACC\nGG\n")
    Path(str(path) + ".fai").write_text("1\t10\t3\t4\t5\n")
    with path.open("rb") as handle:
        genome = dump.indexed_genome(path, handle, {"1": "Chr1"})
        for start in range(11):
            for end in range(start, 11):
                assert genome["Chr1"][start:end] == "ATGTAACCGG"[start:end]
