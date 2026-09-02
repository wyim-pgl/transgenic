"""Contract tests for docs/gsf_spec_v1.md (issue #11). Written before the implementation.

The target module is src/transgenic/utils/gsf_contract.py (pure Python, no torch). It is loaded by
path because importing the `transgenic` package pulls in torch, which the test environment lacks.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "transgenic" / "utils" / "gsf_contract.py"


def _load():
    if not TARGET.exists():
        pytest.fail(f"{TARGET} does not exist yet (RED state; implemented by #12/#13/#14)", pytrace=False)
    mod = types.ModuleType("gsf_contract")
    mod.__file__ = str(TARGET)
    sys.modules["gsf_contract"] = mod
    exec(compile(TARGET.read_text(), str(TARGET), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="session")
def gsf():
    return _load()


# ---- fixtures: real coordinate styles -------------------------------------------------------
TAIR10_SINGLE_CDS_MINUS = """\
ChrM\tphytozomev10\tgene\t273\t734\t.\t-\t.\tID=ATMG00010.TAIR10;Name=ATMG00010
ChrM\tphytozomev10\tmRNA\t273\t734\t.\t-\t.\tID=ATMG00010.1.TAIR10;Parent=ATMG00010.TAIR10;Name=ATMG00010.1
ChrM\tphytozomev10\texon\t273\t734\t.\t-\t.\tID=e1;Parent=ATMG00010.1.TAIR10
ChrM\tphytozomev10\tCDS\t273\t734\t.\t-\t0\tID=ATMG00010.1.TAIR10.CDS.1;Parent=ATMG00010.1.TAIR10
"""

# last gene of the TAIR10 file (minus strand, CDS + 5'UTR), trailing record with no gene after it
TAIR10_LAST_GENE_MINUS = """\
Chr5\tphytozomev10\tgene\t26970312\t26970641\t.\t-\t.\tID=AT5G67640.TAIR10;Name=AT5G67640
Chr5\tphytozomev10\tmRNA\t26970312\t26970641\t.\t-\t.\tID=AT5G67640.1.TAIR10;Parent=AT5G67640.TAIR10
Chr5\tphytozomev10\tCDS\t26970312\t26970360\t.\t-\t0\tID=AT5G67640.1.TAIR10.CDS.2;Parent=AT5G67640.1.TAIR10
Chr5\tphytozomev10\tCDS\t26970444\t26970548\t.\t-\t0\tID=AT5G67640.1.TAIR10.CDS.1;Parent=AT5G67640.1.TAIR10
Chr5\tphytozomev10\tfive_prime_UTR\t26970549\t26970641\t.\t-\t.\tID=AT5G67640.1.TAIR10.five_prime_UTR.1;Parent=AT5G67640.1.TAIR10
"""

REFGEN_V4_PLUS = """\
1\tphytozomev12\tgene\t44289\t49837\t.\t+\t.\tID=Zm00001d027230.RefGen_V4;Name=Zm00001d027230
1\tphytozomev12\tmRNA\t44289\t49837\t.\t+\t.\tID=Zm00001d027230_T001.RefGen_V4;Parent=Zm00001d027230.RefGen_V4
1\tphytozomev12\tfive_prime_UTR\t44289\t44350\t.\t+\t.\tID=u5;Parent=Zm00001d027230_T001.RefGen_V4
1\tphytozomev12\tCDS\t44351\t44947\t.\t+\t0\tID=c1;Parent=Zm00001d027230_T001.RefGen_V4
1\tphytozomev12\tCDS\t45666\t45803\t.\t+\t0\tID=c2;Parent=Zm00001d027230_T001.RefGen_V4
1\tphytozomev12\tCDS\t45888\t46133\t.\t+\t0\tID=c3;Parent=Zm00001d027230_T001.RefGen_V4
1\tphytozomev12\tthree_prime_UTR\t46134\t46342\t.\t+\t.\tID=u3;Parent=Zm00001d027230_T001.RefGen_V4
"""


def gff(chrom, gene_id, strand, tx):
    """Build GFF3 text. tx = {tx_id: [(type, start1, end1, phase), ...]}."""
    allf = [f for feats in tx.values() for f in feats]
    gs, ge = min(f[1] for f in allf), max(f[2] for f in allf)
    out = [f"{chrom}\tt\tgene\t{gs}\t{ge}\t.\t{strand}\t.\tID={gene_id}"]
    for tid, feats in tx.items():
        out.append(f"{chrom}\tt\tmRNA\t{gs}\t{ge}\t.\t{strand}\t.\tID={tid};Parent={gene_id}")
        for k, (typ, s, e, ph) in enumerate(feats):
            out.append(f"{chrom}\tt\t{typ}\t{s}\t{e}\t.\t{strand}\t{ph}\tID={tid}.{k};Parent={tid}")
    return "\n".join(out) + "\n"


@pytest.fixture
def make_gff():
    return gff
