#!/usr/bin/env python3
"""Check that every drawn Panel-C track really shows what its colour claims.

The figures assert three things per locus. This asserts the same three things
against the source GFF3/GTF, for exactly the transcripts the figure draws:

  1. no TAIR10 track carries the highlighted feature,
  2. the orange TransGenic track does carry it,
  3. the dark-green AtRTD3 track has a CDS intron-chain identical to that track.

Also reports, per drawn row, whether the highlighted interval is exonic - which
is what a reader sees as "box" versus "line" - so the rendering can be checked
without pixel-peeping.

Exit status is non-zero if any assertion fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

# panelC_examples -> fig4_forensics -> results -> revision -> <repo root>
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "Figures"))
from make_figure4_panelC import (  # noqa: E402
    LOCI, FIGURE4, DGREEN, GREY, ORANGE, STRONG, analyse, build_rows, chain, headline, overlaps,
)


def is_exonic(rec: dict, feat: tuple) -> bool:
    """True if the transcript covers the interval with exon sequence (drawn as a box)."""
    segs = rec["cds"] + rec["utr"] + rec["exon"]
    return any(s <= feat[0] and e >= feat[1] for s, e in segs)


def touches(rec: dict, feat: tuple) -> bool:
    segs = rec["cds"] + rec["utr"] + rec["exon"]
    return any(overlaps(feat, s) for s in segs)


def main(loci: list) -> int:
    failures = []
    for locus in loci:
        a = analyse(locus)
        rows, _ = build_rows(a)
        feat = a["feats"][0]
        kind, _ = headline(a)
        # "exonic at the feature" is the expected reading for exon/retention modes;
        # for junction modes the novel track must instead be intronic there.
        wants_exon = a["mode"] in ("exon", "retained", "unspliced_utr")
        print(f"\n### {locus}  [{a['mode']}]  {kind}  {feat[0]:,}-{feat[1]:,}")
        print(f"{'row':>3} {'colour':<8} {'label':<16} {'exonic':>7} {'overlaps':>9}")
        for i, (label, rec, color, _) in enumerate(rows, 1):
            name = {GREY: "TAIR10", ORANGE: "novel", DGREEN: "match"}.get(color, "other")
            print(f"{i:>3} {name:<8} {label:<16} "
                  f"{str(is_exonic(rec, feat)):>7} {str(touches(rec, feat)):>9}")

        novel_rec = a["pred"][a["novel_tx"]] if a["novel_tx"] else None
        match_rec = a["art"][a["novel_match"]] if a["novel_match"] else None

        # In combination mode the shaded junctions are by definition present in some
        # TAIR10 isoform - only their combination is new - so checks 1 and 2 do not
        # apply. What must hold there is check 3 plus "chain absent from TAIR10".
        if a["mode"] == "reproduced":
            # Panel-A claim: every predicted chain equals a TAIR10 chain, and at least
            # two distinct reproduced chains are also documented by AtRTD3.
            tair_set = {chain(r["cds"]) for r in a["tair"].values()}
            art_set = {chain(r["cds"]) for r in a["art"].values() if r["cds"]}
            pred_set = {chain(r["cds"]) for r in a["pred"].values()}
            extra = pred_set - tair_set
            if extra:
                failures.append(f"{locus}: prediction has {len(extra)} chain(s) absent from TAIR10")
            if len(pred_set & tair_set & art_set) < 2:
                failures.append(f"{locus}: fewer than 2 reproduced chains shared with AtRTD3")
        elif a["mode"] == "combination":
            if chain(novel_rec["cds"]) in {chain(r["cds"]) for r in a["tair"].values()}:
                failures.append(f"{locus}: novel chain is present in TAIR10")
        else:
            # 1. TAIR10 must not carry the feature
            for label, rec, color, _ in rows:
                if color != GREY:
                    continue
                has = (feat in chain(rec["cds"]) if a["mode"] == "junction"
                       else is_exonic(rec, feat))
                if has:
                    failures.append(f"{locus}: TAIR10 row {label} carries the feature")

            # 2. the novel track must carry it (exonic, or using the junction)
            ok = (is_exonic(novel_rec, feat) if wants_exon
                  else feat in chain(novel_rec["cds"]))
            if not ok:
                failures.append(f"{locus}: novel track does not carry the feature")

        # 3. exact-chain-match claim (novel panels only)
        if novel_rec is not None and match_rec is not None and \
                chain(match_rec["cds"]) != chain(novel_rec["cds"]):
            failures.append(f"{locus}: {a['novel_match']} chain != novel prediction chain")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print(f"All checks passed for {len(loci)} loci "
          "(TAIR10 lacks the feature, novel carries it, chain match exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or sorted(set(LOCI) | set(FIGURE4))))
