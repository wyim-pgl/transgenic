#!/usr/bin/env python3
"""Re-score the 13-species benchmark with GFFCompare v0.12.6, the release used elsewhere.

Table S2 and Figure 5 were produced with GFFCompare **v0.12.10**; every isoform-level
analysis in the paper (Table S4, Figure 6) uses **v0.12.6**. The two releases do not agree
on transcript-level matching, and the gap is large enough to look like a contradiction in
print: scoring the same top-beam *A. thaliana* completion-mode prediction against the same
reference gives transcript-level 79.3/93.7 under v0.12.10 and 62.0/73.3 under v0.12.6 —
the latter matching the 61.9/74.4 the isoform pipeline reports.

Rather than ask a reader to hold two incompatible scales in mind, this re-scores the whole
benchmark on v0.12.6 so Figure 5 and Figure 6 share one. Every tool and mode is re-scored,
not just TransGenic's, so the comparison between tools stays internally consistent.

The completion-mode TransGenic predictions are filtered to the top-ranked beam first, for
the reason given in `27_rescore_prompted_topbeam.py`.

Usage:
    python 29_rescore_benchmark_v0126.py [--out gffcompare_summary_v0126.csv] [--jobs 4]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
STD = CMP / "standardized_results"
REF = CMP / "reference_annotations"
GFFCOMPARE = Path("/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare")

REFERENCES = {
    "A_thaliana": "Athaliana_167_TAIR10.gene.clean.gff3",
    "B_distachyon": "Bdistachyon_314_v3.1.gene_exons.clean.gff3",
    "B_rapa": "BrapaO_302V_711_v1.1.gene.gff3",
    "G_max": "Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
    "L_sativa": "Lsativa_467_v5.gene_exons.gff3",
    "O_sativa": "Osativa_323_v7.0.gene_exons.exon.gff3",
    "P_patens": "Ppatens_318_v3.3.gene_exons.clean.gff3",
    "P_trichocarpa": "Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
    "S_bicolor": "Sbicolor_730_v5.1.gene_exons.clean.gff3",
    "S_italica": "Sitalica_312_v2.2.gene_exons.clean.gff3",
    "S_lycopersicum": "Slycopersicum_796_ITAG5.0.gene.gff3",
    "V_vinifera": "Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3",
    "Z_mays": "Zmays_493_RefGen_V4.gene_exons.exon.gff3",
}

# These export both beam hypotheses per locus and must be filtered before scoring.
TOP_BEAM_TOOLS = {"transgenic400Mprompt"}


def top_beam(src: Path, dst: Path) -> None:
    gm_re = re.compile(r"GM=([^;\s]+)")
    seen: set[str] = set()
    keep: set[str] = set()
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            m = gm_re.search(f[8])
            if not m:
                continue
            if f[2] == "gene":
                gm = m.group(1)
                if gm in seen:
                    continue
                seen.add(gm)
                gid = re.search(r"ID=([^;]+)", f[8])
                if gid:
                    keep.add(gid.group(1))
                out.write(line)
                continue
            parent = re.search(r"Parent=([^;]+)", f[8])
            ident = re.search(r"ID=([^;]+)", f[8])
            if not parent:
                continue
            if parent.group(1).split(".t")[0] in keep or parent.group(1) in keep:
                if ident:
                    keep.add(ident.group(1))
                out.write(line)


def seqids(path: Path, limit: int = 200000) -> set[str]:
    """Sequence names used by a GFF3/GTF, sampled from the first `limit` feature rows."""
    out: set[str] = set()
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i > limit:
                break
            if line.startswith("#"):
                continue
            f = line.split("\t", 1)
            if f[0]:
                out.add(f[0])
    return out


def harmonise_seqids(query: Path, ref: Path, dst: Path) -> Path:
    """Make the query's sequence names match the reference's, or fail loudly.

    S. lycopersicum ships its de novo predictions on `Chr0…Chr12` and its reference on
    `0…12`; GFFCompare silently scores every level at 0.0 when the names do not meet,
    which reads as a result rather than as a failure. Rather than special-case that one
    species, try the two transformations that can differ here and verify the overlap.
    """
    q, r = seqids(query), seqids(ref)
    if q & r:
        return query
    strip = {s[3:] for s in q if s.lower().startswith("chr")}
    add = {"Chr" + s for s in q}
    if strip & r:
        fn = lambda s: s[3:] if s.lower().startswith("chr") else s
    elif add & r:
        fn = lambda s: "Chr" + s
    else:
        raise SystemExit(
            f"sequence names in {query.name} and {ref.name} do not overlap and no simple "
            f"Chr-prefix transformation reconciles them: {sorted(q)[:4]} vs {sorted(r)[:4]}"
        )
    with query.open() as src, dst.open("w") as out:
        for line in src:
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.split("\t")
            if len(f) > 1:
                f[0] = fn(f[0])
                out.write("\t".join(f))
    return dst


def parse_stats(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    pat = re.compile(r"^\s*([A-Za-z ]+?) level:\s*([\d.-]+)\s*\|\s*([\d.-]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1).strip() + "_Sn"] = float(m.group(2))
            out[m.group(1).strip() + "_Pr"] = float(m.group(3))
    return out


def score(job: tuple[str, str, Path]) -> dict | None:
    species, tool, tmp = job
    src = STD / f"{species}_{tool}.gff3"
    ref = REF / REFERENCES[species]
    if not src.exists() or not ref.exists():
        print(f"  skip {species}/{tool}: missing input", file=sys.stderr)
        return None
    query = src
    if tool in TOP_BEAM_TOOLS:
        query = tmp / f"{species}_{tool}_top.gff3"
        top_beam(src, query)
    query = harmonise_seqids(query, ref, tmp / f"{species}_{tool}_seqfix.gff3")
    prefix = tmp / f"{species}_{tool}"
    try:
        subprocess.run([str(GFFCOMPARE), "-r", str(ref), "-o", str(prefix), str(query)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL {species}/{tool}: {e.stderr.decode()[:120]}", file=sys.stderr)
        return None
    st = parse_stats(prefix.with_suffix(".stats"))
    print(f"  {species:<16} {tool:<28} base {st.get('Base_Sn')}/{st.get('Base_Pr')}  "
          f"tx {st.get('Transcript_Sn')}/{st.get('Transcript_Pr')}", file=sys.stderr)
    return {
        "Species": species, "Tool": tool,
        "Base_Sensitivity": st.get("Base_Sn"), "Base_Precision": st.get("Base_Pr"),
        "Exon_Sensitivity": st.get("Exon_Sn"), "Exon_Precision": st.get("Exon_Pr"),
        "Transcript_Sensitivity": st.get("Transcript_Sn"),
        "Transcript_Precision": st.get("Transcript_Pr"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CMP / "gffcompare_summary_v0126.csv")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    assert GFFCOMPARE.exists(), f"gffcompare v0.12.6 not found at {GFFCOMPARE}"
    existing = list(csv.DictReader((CMP / "gffcompare_summary.csv").open()))
    jobs_src = [(r["Species"], r["Tool"]) for r in existing]

    tmp = Path(tempfile.mkdtemp(prefix="v0126_", dir=str(CMP)))
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = [r for r in ex.map(score, [(s, t, tmp) for s, t in jobs_src]) if r]

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written: {args.out} ({len(rows)}/{len(jobs_src)} rows)", file=sys.stderr)
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
