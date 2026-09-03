"""Contract tests for the dataset-role manifest builder (protocol A18.3, A30)."""
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "make_dataset_roles.py"
dr = types.ModuleType("make_dataset_roles"); dr.__file__ = str(SCRIPT)
sys.modules["make_dataset_roles"] = dr
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), dr.__dict__)


def _row(**over):
    r = dict(dataset="protein/orthodb_v12", run="odb12_Viridiplantae.filtered.fa.gz",
             species="cross_species", genotype_stratum="n/a", instrument="n/a",
             data_type="protein_resource", expected_files="1", expected_reads="",
             source_checksum="", source_checksum_authority="none",
             local_fa_md5="453cb32b02e0799950d7d5f4de5f62ac", role="c2_training_eligible",
             basis="", note="")
    r.update(over)
    return r


def test_sensitivity_set_is_an_accepted_role():
    """A30 (PROTOCOL_B1_frozen_v1.md, frozen text): the Swiss-Prot resource is recorded
    'with the role sensitivity_set (a fourth role value; builders still fail closed on
    unknown roles)'. Before this test the tuple held three values and that row was rejected."""
    assert "sensitivity_set" in dr.ROLES
    # validate() also enforces manifest-wide A18.3 invariants (e.g. Cui 2020 must be present and
    # training-eligible), so a single synthetic row can never yield an empty list. Assert on the
    # role check itself, which is what this contract is about.
    v = dr.validate([_row(role="sensitivity_set")])
    assert not [x for x in v if "invalid role" in x], v


def test_unknown_role_still_fails_closed():
    """The fourth value must not turn the check into a pass-through."""
    v = dr.validate([_row(role="training_maybe")])
    assert any("invalid role" in x for x in v), v


def test_the_three_original_roles_are_unchanged():
    for role in ("b1_validation_only", "c2_training_eligible", "excluded"):
        assert role in dr.ROLES
