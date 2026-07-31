"""Shared typography and export settings for every manuscript figure.

Import this before creating any figure:

    import figstyle          # from Figures/
    figstyle.apply()

Two things it fixes that matter for submission:

**Arial.** Journals ask for Arial and the cluster has neither Arial nor Helvetica
installed, so matplotlib silently falls back to DejaVu Sans — a different face with
different metrics, which only shows up once a reviewer compares figures. Liberation Sans
is metric-compatible with Arial (same advance widths, same layout) and openly licensed,
so it is bundled here and registered at import. `font.sans-serif` still lists Arial
first: on a machine that has the real thing, that wins.

**Type 42 fonts in the PDF.** matplotlib defaults to Type 3, which several publishers
reject outright and which cannot be edited in Illustrator. Type 42 embeds the TrueType
outlines and keeps the text selectable and editable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

# Bundled with the shared tooling, not with the repository: the TTFs are 1.6 MB and
# are not source. Installed 2026-07-30, Liberation Sans 2.1.5.
FONT_DIRS = (
    Path("/data/gpfs/assoc/pgl/tools/fonts"),
    Path.home() / ".fonts",
)

# Arial first so a machine that actually has it uses it; Liberation Sans is the
# metric-compatible stand-in; DejaVu Sans is matplotlib's own last resort.
SANS_STACK = ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"]
# Figure 1 sets DNA sequence and GSF tokens in a fixed pitch on purpose; keep that
# in the same type family so the page does not mix two unrelated designs.
MONO_STACK = ["Liberation Mono", "Courier New", "DejaVu Sans Mono"]

_applied = False


def register_fonts() -> list[str]:
    """Add the bundled TTFs to matplotlib's font manager. Returns the names found."""
    found = []
    for d in FONT_DIRS:
        if not d.is_dir():
            continue
        for path in fm.findSystemFonts(str(d), fontext="ttf"):
            try:
                fm.fontManager.addfont(path)
                found.append(fm.FontProperties(fname=path).get_name())
            except Exception:
                pass
    return sorted(set(found))


def resolved_family() -> str:
    """Which face matplotlib will actually use — call it to verify, not to assume."""
    return fm.FontProperties(family=SANS_STACK).get_name()


def apply(base_size: float = 8.0) -> None:
    """Set typography and export defaults. Safe to call more than once."""
    global _applied
    if not _applied:
        register_fonts()
        _applied = True
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS_STACK,
        "font.monospace": MONO_STACK,
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "legend.frameon": False,
        # Type 42 = embedded TrueType: accepted by publishers, editable in Illustrator.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


if __name__ == "__main__":
    names = register_fonts()
    apply()
    print("registered:", names or "(none found)")
    print("resolved family:", resolved_family())
    print("resolved mono:  ", fm.FontProperties(family=MONO_STACK).get_name())
    if resolved_family() not in ("Arial", "Liberation Sans"):
        raise SystemExit("FAIL: neither Arial nor Liberation Sans resolved")
    print("pdf.fonttype:", matplotlib.rcParams["pdf.fonttype"])
