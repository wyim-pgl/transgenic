#!/usr/bin/env python3
"""A19.1 leakage filter for the OrthoDB v12 Viridiplantae partition (protocol A19; issue #44).

Input: the partitioned FASTA (`partitioned_odb12/Viridiplantae.fa.gz`; headers `>taxid_version:gene<TAB>taxid_version`).
Removes
  1. every sequence whose organism taxid is an excluded (evaluated) species or one of its sub-taxa (subspecies, cultivar,
     strain), decided from NCBI taxonomy lineages — `--lineage lineage.tsv` (taxid<TAB>lineage ids, root-most last) or
     `--fetch-lineage` (E-utilities, cached to the same TSV). A taxid of the partition without a lineage aborts the run
     (fail closed) unless `--allow-missing-lineage`.
  2. every sequence identical to a protein of an excluded proteome (`--exclude-proteome SP=<fasta>`: the evaluated species'
     reference proteomes; after #14 also the training-species test-orthogroup proteins).
Outputs (`--out-dir`): `odb12_Viridiplantae.filtered.fa.gz`, `counts_by_taxid.tsv` (before/after per taxid),
`filter_summary.json` (input/output md5, excluded taxids, removals per rule and per proteome).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

_TAXID_RE = re.compile(r"^>?(\d+)_")


def taxid_of_header(header: str) -> int:
    m = _TAXID_RE.match(header.strip())
    if not m:
        raise ValueError(f"cannot read a taxid from header {header!r}")
    return int(m.group(1))


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].rstrip("\n"), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def read_lineage(path: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    with open(path) as fh:
        header = fh.readline()
        if not header.startswith("taxid"):
            fh.seek(0)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and c[0].isdigit():
                out[int(c[0])] = [int(x) for x in c[1].split(";") if x]
    return out


def parse_taxonomy_xml(xml: str) -> Dict[int, List[int]]:
    """NCBI taxonomy efetch XML -> {taxid: [taxid, parent, grandparent, ...]} (the taxon itself first, root-most last)."""
    import xml.etree.ElementTree as ET
    out: Dict[int, List[int]] = {}
    root = ET.fromstring(xml)
    for taxon in root.findall("Taxon"):                      # top-level taxa only; LineageEx holds nested Taxon elements
        tid_el = taxon.find("TaxId")
        if tid_el is None or not (tid_el.text or "").strip().isdigit():
            continue
        tid = int(tid_el.text.strip())
        parents = [int(t.text.strip()) for t in taxon.findall("LineageEx/Taxon/TaxId") if (t.text or "").strip().isdigit()]
        out[tid] = [tid] + list(reversed(parents))
    return out


def fetch_lineage(taxids: Iterable[int], cache_path: str, batch: int = 100, pause: float = 0.4) -> Dict[int, List[int]]:
    known = read_lineage(cache_path) if os.path.exists(cache_path) else {}
    todo = sorted(set(taxids) - set(known))
    for i in range(0, len(todo), batch):
        ids = ",".join(str(t) for t in todo[i:i + batch])
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode({"db": "taxonomy", "id": ids, "retmode": "xml"})
        with urllib.request.urlopen(url, timeout=120) as resp:
            known.update(parse_taxonomy_xml(resp.read().decode()))
        time.sleep(pause)
    with open(cache_path, "w") as fh:
        fh.write("taxid\tlineage\n")
        for t in sorted(known):
            fh.write(f"{t}\t{';'.join(str(x) for x in known[t])}\n")
    return known


def excluded_taxids(exclude: Set[int], lineage: Dict[int, List[int]], present: Optional[Set[int]] = None,
                    allow_missing: bool = False) -> Set[int]:
    """Taxids to drop: every present taxid whose lineage contains an excluded species. Fails closed on a missing lineage."""
    present = set(present) if present is not None else set(lineage)
    missing = sorted(t for t in present if t not in lineage)
    if missing and not allow_missing:
        raise SystemExit(f"no lineage for {len(missing)} taxid(s) of the partition (e.g. {missing[:5]}); fetch lineages or pass --allow-missing-lineage")
    return {t for t in present if t in lineage and (set(lineage[t]) & exclude)}


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_kv(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected SPECIES=PATH, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="partitioned OrthoDB FASTA (.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude-taxid", type=int, nargs="+", default=[3702, 4577, 4081], help="evaluated species (A19.1)")
    ap.add_argument("--lineage", help="taxid<TAB>lineage TSV (cache for --fetch-lineage)")
    ap.add_argument("--fetch-lineage", action="store_true", help="fetch missing lineages from NCBI E-utilities into --lineage")
    ap.add_argument("--allow-missing-lineage", action="store_true")
    ap.add_argument("--exclude-proteome", action="append", default=[], help="SPECIES=<protein FASTA>: drop identical sequences (repeatable)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    present: Counter = Counter()
    for name, _ in read_fasta(a.fasta):
        present[taxid_of_header(name)] += 1
    lineage = read_lineage(a.lineage) if a.lineage and os.path.exists(a.lineage) else {}
    if a.fetch_lineage:
        if not a.lineage:
            raise SystemExit("--fetch-lineage needs --lineage <cache path>")
        lineage = fetch_lineage(present.keys(), a.lineage)
    drop_taxids = excluded_taxids(set(a.exclude_taxid), lineage, set(present), a.allow_missing_lineage)
    exclude_seqs: Dict[str, Set[str]] = {}
    for sp, path in _parse_kv(a.exclude_proteome).items():
        exclude_seqs[sp] = {s.rstrip("*").upper() for _, s in read_fasta(path)}
    out_fa = os.path.join(a.out_dir, "odb12_Viridiplantae.filtered.fa.gz")
    kept: Counter = Counter()
    removed_taxid = 0
    removed_exact: Counter = Counter()
    with gzip.open(out_fa, "wt") as out:
        for name, seq in read_fasta(a.fasta):
            tid = taxid_of_header(name)
            if tid in drop_taxids:
                removed_taxid += 1
                continue
            s = seq.rstrip("*").upper()
            hit = next((sp for sp, ss in exclude_seqs.items() if s in ss), None)
            if hit:
                removed_exact[hit] += 1
                continue
            kept[tid] += 1
            out.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")
    with open(os.path.join(a.out_dir, "counts_by_taxid.tsv"), "w") as fh:
        fh.write("taxid\tsequences_in\tsequences_out\texcluded_by_lineage\n")
        for tid, n in sorted(present.items()):
            fh.write(f"{tid}\t{n}\t{kept.get(tid, 0)}\t{int(tid in drop_taxids)}\n")
    summary = {"input": os.path.abspath(a.fasta), "input_md5": file_md5(a.fasta), "output": out_fa, "output_md5": file_md5(out_fa),
               "sequences_in": sum(present.values()), "sequences_out": sum(kept.values()), "taxids_in": len(present), "taxids_out": len(kept),
               "exclude_taxid": sorted(a.exclude_taxid), "excluded_taxids": sorted(drop_taxids), "removed_by_taxid": removed_taxid,
               "removed_by_exact_match": dict(removed_exact), "exclude_proteomes": _parse_kv(a.exclude_proteome),
               "lineage_source": a.lineage or "", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(a.out_dir, "filter_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("sequences_in", "sequences_out", "taxids_in", "taxids_out", "removed_by_taxid", "removed_by_exact_match", "excluded_taxids")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
