#!/usr/bin/env python3
"""NCBI esummary v2.0 metadata of every EST of one species -> the ``--est-meta`` TSV of b6_c0_metadata.py.

    b6_est_esummary.py --label Athaliana --taxid 3702 --expect-count 1529700 --out DIR/Athaliana/est_esummary.tsv

Query = the one the frozen EST download used (evidence/est/est_fetch.sh): ``txid<taxid>[Organism] AND
gbdiv_est[PROP]`` on db=nuccore, with the history server; records are pulled with esummary version 2.0 in batches of
``--batch`` (default 5000).  Output columns (b6_c0_metadata.EST_META_COLUMNS): accession_version, clone_lib, clone,
biosample.  ``clone_lib`` / ``clone`` come from the record's paired SubType / SubName lists (pipe-separated, equal
length, else the record fails closed); ``biosample`` from BioSample.  A tab, CR or LF inside a value is replaced by
one space and counted (the TSV has no quoting); nothing else is edited.

Fail-closed checks: the esearch count must equal ``--expect-count`` (the COUNT the frozen download recorded; a change
means NCBI's EST set moved since the download); every batch must return exactly the requested number of records
(short batches are re-fetched in halves); accessions must be unique and the total must equal the count.  Output is
written to ``<out>.part`` and renamed; ``<out>.sha256`` and ``<out>.PROVENANCE.json`` (query, counts, dates, sha256)
are written next to it.  Rate: 3 requests/s without NCBI_API_KEY, 10 with it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
COLUMNS = ("accession_version", "clone_lib", "clone", "biosample")
KEY = os.environ.get("NCBI_API_KEY", "")
PAUSE = 0.12 if KEY else 0.4
REQUEST_SECONDS = int(os.environ.get("B6_ESUMMARY_REQUEST_SECONDS", "45"))  # a good 5,000-record reply takes ~4 s


class FetchError(RuntimeError):
    pass


def call(endpoint, params, tries=20):
    if KEY:
        params = dict(params, api_key=KEY)
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(tries):
        try:
            time.sleep(PAUSE)
            # curl -m is a hard wall-clock limit per request; urlopen's timeout is per socket operation and a
            # trickling NCBI response (observed: 947 s for 500 records vs 3.5 s for 5,000) never trips it.
            res = subprocess.run(["curl", "-sS", "--fail", "--compressed", "-m", str(REQUEST_SECONDS), url],
                                 capture_output=True)
            if res.returncode != 0:
                raise FetchError(f"curl rc={res.returncode}: {res.stderr.decode(errors='replace')[:200]}")
            body = res.stdout
            root = ET.fromstring(body)
            err = root.find(".//ERROR")
            if err is not None and (err.text or "").strip():
                raise FetchError(f"{endpoint}: {err.text.strip()[:200]}")
            return root
        except (OSError, ET.ParseError, FetchError, subprocess.SubprocessError) as exc:
            last = exc
            time.sleep(5 if i < 10 else 60)    # stalls are sporadic: retry fast, back off only if persistent
    raise FetchError(f"{endpoint} failed after {tries} tries: {last}")


def clean(value, stats):
    if value is None:
        return ""
    v = value.strip()
    if any(c in v for c in "\t\r\n"):
        stats["control_chars_replaced"] += 1
        v = " ".join(v.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())
    return v


def parse(root, stats):
    out = []
    for d in root.iter("DocumentSummary"):
        acc = clean(d.findtext("AccessionVersion"), stats)
        if not acc:
            raise FetchError(f"docsum uid={d.get('uid')} has no AccessionVersion")
        types = (d.findtext("SubType") or "").split("|")
        names = (d.findtext("SubName") or "").split("|")
        if (d.findtext("SubType") or "") and len(types) != len(names):
            raise FetchError(f"{acc}: SubType/SubName lengths differ ({len(types)} vs {len(names)}); cannot pair")
        sub = dict(zip(types, names)) if d.findtext("SubType") else {}
        lib, clone = clean(sub.get("clone_lib"), stats), clean(sub.get("clone"), stats)
        stats["with_clone_lib"] += bool(lib)
        stats["with_clone"] += bool(clone)
        out.append((acc, lib, clone, clean(d.findtext("BioSample"), stats)))
    return out


def fetch_range(webenv, qk, start, n, stats):
    rows = parse(call("esummary.fcgi", dict(db="nuccore", version="2.0", query_key=qk, WebEnv=webenv,
                                            retstart=start, retmax=n)), stats)
    if len(rows) == n:
        return rows
    stats["short_batches"] += 1
    if n == 1:
        raise FetchError(f"retstart={start}: esummary returned {len(rows)} records for 1")
    h = n // 2
    return fetch_range(webenv, qk, start, h, stats) + fetch_range(webenv, qk, start + h, n - h, stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--taxid", required=True, type=int)
    ap.add_argument("--expect-count", required=True, type=int, help="COUNT of the frozen est_fetch.sh download")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=5000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    term = f"txid{a.taxid}[Organism] AND gbdiv_est[PROP]"
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        root = call("esearch.fcgi", dict(db="nuccore", term=term, usehistory="y", retmax=0))
        count, webenv, qk = int(root.findtext("Count")), root.findtext("WebEnv"), root.findtext("QueryKey")
        if count != a.expect_count:
            raise FetchError(f"{a.label}: esearch count {count} != frozen download COUNT {a.expect_count}")
        stats = dict(control_chars_replaced=0, with_clone_lib=0, with_clone=0, short_batches=0)
        seen = set()
        part = Path(str(out) + ".part")
        with open(part, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for start in range(0, count, a.batch):
                rows = fetch_range(webenv, qk, start, min(a.batch, count - start), stats)
                for r in rows:
                    if r[0] in seen:
                        raise FetchError(f"{a.label}: duplicate accession {r[0]}")
                    seen.add(r[0])
                    fh.write("\t".join(r) + "\n")
                print(f"{a.label} {start + len(rows)}/{count}", flush=True)
        if len(seen) != count:
            raise FetchError(f"{a.label}: {len(seen)} records != count {count}")
    except FetchError as exc:
        print(f"b6_est_esummary: {exc}", file=sys.stderr)
        return 1
    part.rename(out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{h}  {out.name}\n")
    prov = dict(label=a.label, taxid=a.taxid, db="nuccore", term=term, esummary_version="2.0", batch=a.batch,
                count=count, records=len(seen), started=started,
                finished=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                output=out.name, output_sha256=h, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                **stats)
    Path(str(out) + ".PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    print(json.dumps(prov, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
