"""The staging gate of 65_make_training_input_manifest.sh (issue #5 packaging path).

The gate this file exercises used to compare BASENAMES only, so a held-out test genome whose
content sat under an allowed training-species filename passed it. Four test genomes were found in
training_input/genomes/ on the first pass, which is why the gate exists at all; comparing names is
not enough once the bundle is assembled by hand.

Everything here runs against a miniature bundle built in a tmp_path: nine one-line FASTAs and a
manifest carrying their real md5s. The real bundle is 28.9 GB and must never be a test dependency.
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "65_make_training_input_manifest.sh"

SPECIES = ("Athaliana", "Bdistachyon", "Gmax", "Osativa", "Ppatens",
           "Ptrichocarpa", "Sbicolor", "Sitalica", "Vvinifera")

HEADER = "species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\n"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_bundle(root: Path, species=SPECIES) -> Path:
    """A miniature training_input/ that the gate should accept."""
    (root / "genomes").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    rows = []
    for sp in species:
        fa = root / "genomes" / f"{sp}_test.fa"
        fa.write_text(f">chr1 {sp}\nACGT{sp.upper()}\n")
        rows.append(f"{sp}\t{sp}\tvtest\t{fa}\t{md5(fa)}\t-\t-\tsynthetic\n")
    (root / "manifests" / "b5_species_v1.tsv").write_text(HEADER + "".join(rows))
    shutil.copy(SCRIPT, root / "make_manifest.sh")
    return root


def run(root: Path):
    return subprocess.run(["bash", str(root / "make_manifest.sh")],
                          capture_output=True, text=True)


def test_valid_bundle_is_accepted_and_writes_a_manifest(tmp_path):
    root = build_bundle(tmp_path / "training_input")
    r = run(root)
    assert r.returncode == 0, r.stderr
    manifest = root / "MANIFEST.tsv"
    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    assert lines[0].split("\t")[0] == "category"
    assert sum(1 for l in lines if l.startswith("genomes\t")) == 9
    assert lines[-1].startswith("TOTAL\t")


def test_test_species_content_under_a_training_filename_is_refused(tmp_path):
    """The bypass the basename gate allowed: right name, wrong content."""
    root = build_bundle(tmp_path / "training_input")
    # Zmays is a held-out TEST species. Give its content the name of a training genome.
    (root / "genomes" / "Athaliana_test.fa").write_text(">chr1 Zmays\nACGTZMAYS\n")
    r = run(root)
    assert r.returncode == 2
    assert "md5 mismatch" in r.stderr
    assert not (root / "MANIFEST.tsv").exists()


def test_absent_species_manifest_is_refused(tmp_path):
    """`if [ -f "$MANI" ]` used to skip the entire gate in exactly this case."""
    root = build_bundle(tmp_path / "training_input")
    (root / "manifests" / "b5_species_v1.tsv").unlink()
    r = run(root)
    assert r.returncode == 2
    assert "required species manifest is absent" in r.stderr
    assert not (root / "MANIFEST.tsv").exists()


def test_a_missing_training_genome_is_refused(tmp_path):
    """Presence of all nine, not merely absence of anything forbidden."""
    root = build_bundle(tmp_path / "training_input")
    (root / "genomes" / "Sitalica_test.fa").unlink()
    r = run(root)
    assert r.returncode == 2
    assert "missing training genome Sitalica_test.fa" in r.stderr


def test_an_extra_genome_is_refused(tmp_path):
    root = build_bundle(tmp_path / "training_input")
    (root / "genomes" / "Zmays_493_APGv4.fa").write_text(">chr1 Zmays\nACGT\n")
    r = run(root)
    assert r.returncode == 2
    assert "is not a training genome" in r.stderr


def test_a_manifest_naming_other_than_nine_species_is_refused(tmp_path):
    root = build_bundle(tmp_path / "training_input", species=SPECIES[:8])
    r = run(root)
    assert r.returncode == 2
    assert "exactly nine" in r.stderr


def test_awk_header_error_is_not_masked_by_the_row_count_check(tmp_path):
    """awk's `exit N` still runs END, so an END that calls exit would overwrite the code set in a
    rule. A bad header must report as a bad header, not as 'not nine species'."""
    root = build_bundle(tmp_path / "training_input")
    mani = root / "manifests" / "b5_species_v1.tsv"
    mani.write_text(mani.read_text().replace("species_id\t", "WRONG\t", 1))
    r = run(root)
    assert r.returncode == 2
    assert "unexpected header" in r.stderr
    assert "exactly nine" not in r.stderr


def test_awk_programs_avoid_interval_expressions():
    """`/x{32}/` matches nothing under POSIX awk and older mawk, silently.

    The gate used /^[[:xdigit:]]{32}$/ to validate an md5. mawk 1.3.4 on pronghorn matches it;
    the awk on pgl-gpu does not, so every md5 read as invalid and the gate refused a correct
    manifest — found 2026-09-05 when the A40 pre-flight ran the suite on the GPU host. The failure
    direction was safe, but ACCESS staging would have been blocked on that machine with a message
    accusing the manifest.

    A test cannot easily run two awks, so it pins the rule instead: no interval expressions in the
    awk programs of this script. Use length() plus a character class.
    """
    import re
    # Comments explain the trap and quote the bad pattern, so they must not be scanned.
    code = "\n".join(l for l in SCRIPT.read_text().splitlines() if not l.lstrip().startswith("#"))
    offenders = [m.group(0) for m in
                 re.finditer(r"/\^?\[?\[:[a-z]+:\]\]?[^/\n]*\{\d+(,\d*)?\}[^/\n]*/", code)]
    assert not offenders, f"awk interval expression is not portable: {offenders}"
