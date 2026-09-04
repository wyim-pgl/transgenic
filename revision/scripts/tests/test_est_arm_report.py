"""A37 paired reporting (66_est_arm_report.py).

The parsing here is small but easy to get wrong in a way that would silently change a reported
mapping rate: samtools writes the percentage inside the same line it writes the count, newer
versions add a separate "primary mapped (" line that must not be mistaken for it, and
supplementary records have to leave both numerator and denominator or the two arms stop being
comparable to the 2026-09-03 table.
"""
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "66_est_arm_report.py"
mod = types.ModuleType("est_arm_report"); mod.__file__ = str(SCRIPT)
sys.modules["est_arm_report"] = mod
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), mod.__dict__)

# samtools 1.3.1, the version the alignment actually runs (Athaliana primary, job 6148523_0)
FLAGSTAT_1_3 = """1050677 + 0 in total (QC-passed reads + QC-failed reads)
0 + 0 secondary
3852 + 0 supplementary
0 + 0 duplicates
978663 + 0 mapped (93.15% : N/A)
0 + 0 paired in sequencing
"""

# A newer samtools adds "primary mapped"; it must not be read as the mapped count.
FLAGSTAT_NEWER = """1050677 + 0 in total (QC-passed reads + QC-failed reads)
0 + 0 secondary
3852 + 0 supplementary
0 + 0 duplicates
978663 + 0 mapped (93.15% : N/A)
1046825 + 0 primary
974811 + 0 primary mapped (93.12% : N/A)
"""


def test_reads_samtools_1_3_counts(tmp_path):
    f = tmp_path / "flagstat.txt"; f.write_text(FLAGSTAT_1_3)
    assert mod.read_flagstat(f) == (1050677, 3852, 978663)


def test_primary_mapped_line_is_not_mistaken_for_mapped(tmp_path):
    f = tmp_path / "flagstat.txt"; f.write_text(FLAGSTAT_NEWER)
    assert mod.read_flagstat(f) == (1050677, 3852, 978663)


def test_supplementary_leaves_both_numerator_and_denominator(tmp_path):
    """93.12 %, not 93.15 %: the 2026-09-03 table was primary-alignment based."""
    d = tmp_path / "evidence" / "est_align" / "Athaliana"
    d.mkdir(parents=True)
    (d / "flagstat.txt").write_text(FLAGSTAT_1_3)
    (d / "PROVENANCE.txt").write_text(
        "min_len        100   (protocol A37, v1.27)\nobserved_min   100\n"
        "est_raw_count  1529700\nwall_seconds   100\n")
    (d / "DONE").write_text("est_md5=x\n")
    v = mod.arm_stats(str(tmp_path), "Athaliana", "est_align")
    assert v["records"] == 1050677 - 3852 == 1046825
    assert v["mapped"] == 978663 - 3852 == 974811
    assert round(v["rate"], 2) == 93.12
    assert v["min_len"] == "100" and v["observed_min"] == "100"


def test_an_arm_without_a_DONE_marker_is_pending_not_zero(tmp_path):
    """A report that quietly omits what has not finished looks like a finished report."""
    d = tmp_path / "evidence" / "est_align" / "Gmax"
    d.mkdir(parents=True)
    (d / "flagstat.txt").write_text(FLAGSTAT_1_3)
    (d / "PROVENANCE.txt").write_text("min_len 100\n")
    assert mod.arm_stats(str(tmp_path), "Gmax", "est_align") is None


def test_unreadable_flagstat_raises_rather_than_reporting_a_rate(tmp_path):
    f = tmp_path / "flagstat.txt"; f.write_text("nothing useful here\n")
    with pytest.raises(ValueError):
        mod.read_flagstat(f)
