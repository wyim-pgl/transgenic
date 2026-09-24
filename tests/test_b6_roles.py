"""WP2-roles (plan v4.1 §3): ``b6_roles.load_roles`` verifies the sidecar and fails closed on every malformed row."""
import hashlib
from pathlib import Path

import pytest

from transgenic.evidence import b6_roles as R

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "manifests" / "DATASET_ROLES.b6_v1.tsv"


def _write(tmp_path, text, *, sidecar=True, name="roles.tsv"):
    p = tmp_path / name
    p.write_text(text)
    if sidecar:
        Path(str(p) + ".sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}\n")
    return p


def _real_lines():
    return REAL.read_text().splitlines(keepends=True)


def _mutate(tmp_path, fn):
    lines = _real_lines()
    head, rows = lines[0], [l.rstrip("\n").split("\t") for l in lines[1:]]
    rows = fn(rows)
    return _write(tmp_path, head + "".join("\t".join(r) + "\n" for r in rows))


def test_real_manifest_loads_28_rows_with_verified_sidecar():
    roles = R.load_roles(REAL)
    assert len(roles.rows) == 28
    assert roles.sha256 == hashlib.sha256(REAL.read_bytes()).hexdigest()
    a_ont2 = [r for r in roles.rows if r.set_label == "A-ONT2"]
    assert sorted(r.run for r in a_ont2) == ["SRR10611193", "SRR10611194", "SRR10611195"]
    assert all(r.b6_input and not r.b6_score and r.b6_input_scope == "tracks+chains" for r in a_ont2)
    hq = [r for r in roles.rows if r.set_label == "M-HQ18"]
    assert hq and all(r.b6_input_scope == "tracks_only" for r in hq)
    for label in R.SCORING_SETS:
        assert all(r.b6_score and not r.b6_input for r in roles.rows if r.set_label == label)


def test_lookup_is_by_dataset_and_run_and_refuses_unknown():
    roles = R.load_roles(REAL)
    row = roles.get("est/Athaliana", "est.fa.gz")
    assert row.set_label == "A-EST" and roles.is_input("est/Athaliana", "est.fa.gz")
    assert roles.is_score("ont/Athaliana_FLIC_PRJNA1087576", "SRR28341943")
    with pytest.raises(R.RolesError):
        roles.get("training/ont/Athaliana/proj1", "SRR1")      # never resolved by source class or species
    assert {r.run for r in roles.inputs("Zmays")} == {"est.fa.gz", "ERR2093901", "ERR2093902", "ERR2093903",
                                                      "ERR2093904", "ERR2093905"}


def test_missing_or_wrong_sidecar_fails(tmp_path):
    p = _write(tmp_path, REAL.read_text(), sidecar=False)
    with pytest.raises(R.RolesError, match="sidecar"):
        R.load_roles(p)
    Path(str(p) + ".sha256").write_text("0" * 64 + "  roles.tsv\n")
    with pytest.raises(R.RolesError, match="sha256"):
        R.load_roles(p)


def test_row_both_input_and_score_fails(tmp_path):
    def fn(rows):
        rows[-1][7] = "true"          # A-EST b6_score
        return rows
    with pytest.raises(R.RolesError, match="both"):
        R.load_roles(_mutate(tmp_path, fn))


def test_row_neither_input_nor_score_fails(tmp_path):
    def fn(rows):
        rows[-1][6], rows[-1][8] = "false", "-"
        return rows
    with pytest.raises(R.RolesError, match="neither"):
        R.load_roles(_mutate(tmp_path, fn))


def test_scoring_set_not_marked_score_fails(tmp_path):
    def fn(rows):
        rows[0][6], rows[0][7], rows[0][8] = "true", "false", "tracks+chains"     # A-ONT1 as input
        return rows
    with pytest.raises(R.RolesError, match="A-ONT1"):
        R.load_roles(_mutate(tmp_path, fn))


def test_bad_input_scope_fails(tmp_path):
    def fn(rows):
        rows[-1][8] = "chains_only"
        return rows
    with pytest.raises(R.RolesError, match="scope"):
        R.load_roles(_mutate(tmp_path, fn))


def test_score_row_with_input_scope_fails(tmp_path):
    def fn(rows):
        rows[0][8] = "tracks_only"
        return rows
    with pytest.raises(R.RolesError, match="scope"):
        R.load_roles(_mutate(tmp_path, fn))


def test_m_hq18_not_tracks_only_fails(tmp_path):
    def fn(rows):
        for r in rows:
            if r[5] == "M-HQ18":
                r[8] = "tracks+chains"
        return rows
    with pytest.raises(R.RolesError, match="M-HQ18"):
        R.load_roles(_mutate(tmp_path, fn))


def test_a_ont2_missing_a_run_fails(tmp_path):
    def fn(rows):
        return [r for r in rows if r[1] != "SRR10611194"]
    with pytest.raises(R.RolesError, match="A-ONT2"):
        R.load_roles(_mutate(tmp_path, fn))


def test_duplicate_dataset_run_and_bad_header_fail(tmp_path):
    def dup(rows):
        return rows + [rows[-1]]
    with pytest.raises(R.RolesError, match="duplicate"):
        R.load_roles(_mutate(tmp_path, dup))
    text = REAL.read_text().replace("b6_input_scope", "scope", 1)
    with pytest.raises(R.RolesError, match="header"):
        R.load_roles(_write(tmp_path, text, name="bad.tsv"))


def test_boolean_fields_are_strict(tmp_path):
    def fn(rows):
        rows[-1][6] = "yes"
        return rows
    with pytest.raises(R.RolesError, match="boolean"):
        R.load_roles(_mutate(tmp_path, fn))
