#!/usr/bin/env python3
"""Per-read ``c0_metadata.tsv`` (+ ``.sha256``) for b6_build_c0.py -- the A11 / A16 / A18 read metadata.

    EST     b6_c0_metadata.py --source est --species Athaliana --roles DATASET_ROLES.est_v1.tsv
                --screened evidence/est/Athaliana/univec/est.univec.fa.gz
                (--genbank evidence/est/Athaliana/est.gb.gz [...] [--bioproject PRJNA...]
                 | --est-meta <esummary v2.0 TSV> --bioproject PRJNA...)
                --out evidence/est_align/Athaliana/c0_metadata.tsv
    ONT     b6_c0_metadata.py --source ont --species S --roles DATASET_ROLES.longread_v1.tsv --dataset D --run R
                --library <SRA/ENA experiment accession or author library key>
                --reads <run reads FASTQ/FASTA(.gz)> --protocol cDNA|direct_RNA|unresolved
                --umi-status present|absent|unresolved [--bioproject PRJ...] --out <align>/<sp>/<ds>/<run>/c0_metadata.tsv
    PacBio  b6_c0_metadata.py --source pacbio ... --library <...> --reads <CCS/FLNC reads> [--bioproject PRJ...] --out ...

Field sources (every field has exactly one documented source; a missing required field fails closed, nothing is
guessed; the '*' first data row carries the run-level values, per-read rows the per-read ones):

  read                 first whitespace token of the record header of the exact FASTA/FASTQ the frozen alignment was
                       made from (= minimap2's SAM QNAME).  EST: the A21-screened primary-arm FASTA
                       (univec/est.univec.fa.gz, 61_univec_trim.py names pieces '<acc>_part<i>').
  accession            EST: = read (the versioned GenBank accession, A21 '_partN' kept; b6_build_c0.parse_accession
                       splits stem / version / part).  Long reads: not written (the read name is the identity).
  post_trim_length     EST: b-a+1 of the header tag 'univec=<action>:<a>-<b>/<qlen>' (61_univec_trim.py), which must
                       equal the record's sequence length.
  library              EST: GenBank '/clone_lib' source qualifier of the original versioned accession (--genbank,
                       the rettype=gb flat files of PROTOCOL_B1 section 3; wrapped qualifier lines are joined with one
                       space), or the full 'clone_lib' string of the NCBI esummary v2.0 TSV (--est-meta; it often
                       starts with a BioSample id, e.g. 'SAMN00739408 Vitis vinifera cv. Pinot Noir PN40024 leaves
                       cDNA library').  Required; spaces are kept.  ONT/PacBio: --library, the BIOLOGICAL library of
                       the run (A11, PROTOCOL_B1 line 558: "A biological library is one independently prepared cDNA
                       library; runs, plates, read directions, accession series and download batches do not create
                       libraries").  Its value is the ENA/SRA EXPERIMENT accession of the run (SRX/ERX/DRX: in the SRA
                       model one experiment = one library preparation and its runs are sequencing runs of it), or an
                       author-supplied library key when the preparation is documented outside SRA (several
                       experiments known to be one preparation -> one key, recorded in the author's notes).  Required,
                       never defaulted from the run, and a value equal to the run accession is refused (fix round 6,
                       finding 1: `library = run` split one A16 unit -- a ZMW, a UMI, a PCR-equivalence chain -- seen
                       in two runs of one preparation into two molecules, so rule 4(b) closed only one of them).  Runs
                       sharing a --library merge into one A16 unit wherever the unit key matches; `run` stays a
                       separate column (DATASET_ROLES row, alignment / audit layout, c0_provenance inputs).
  clone                EST: GenBank '/clone' qualifier or the esummary 'clone'; empty when the record has none (A11
                       then falls back to the accession stem and the library signature rule).  Where the screened FASTA
                       header says 'cDNA clone X', X must agree with it (else fail).  Long reads: not written.
  bioproject           EST: GenBank 'DBLINK BioProject:' of the record, else --bioproject (explicit, recorded in the
                       '*' row; REQUIRED with --est-meta, whose records carry none -- never inferred); long reads: the
                       PRJ* accession in the DATASET_ROLES dataset path, else --bioproject.  Required.
  genotype_stratum     the DATASET_ROLES row (est/<species> or <dataset>/<run>) genotype_stratum.
  ingestion_qc_pass    true: the record is in the frozen post-ingestion input (A21-screened primary arm for EST, the
                       fetched and verified run for long reads); rejected records never reach the alignment.
  mapping_ambiguous    false at ingestion (A18.5); b6_build_c0 sets it true from the audit's equal-best placement.
  protocol, umi_status ONT: --protocol / --umi-status as resolved by the author from the run's library-prep record
                       (DATASET_ROLES has no such column).  'unresolved' writes empty cells, and b6_build_c0 then
                       fails closed exactly as 59_evidence_support.assign_molecules does.  PacBio/EST: not written.
  source_molecule      PacBio: '<movie>/<zmw>' (A16: one ZMW = one unit; 59_evidence_support keys it within the
                       --library, so one ZMW in two runs of one library is one unit and in two libraries two units --
                       the A11 split, decided by the author's per-run --library).  SRA-renamed reads keep read = 'SRRnnn.k' (the
                       QNAME) and carry the ZMW in the SECOND header token, e.g. '>SRR35145693.1
                       m54283U_201030_021721/11/ccs/1' ('/ccs' or '/ccs/<n>' suffix); a native '<movie>/<zmw>/ccs[/...]'
                       read name carries it in the first.  A missing or malformed ZMW token fails closed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import c0_schema as S  # noqa: E402  (shared A49.1 long-read library validator)

COLUMNS = ("read", "accession", "library", "bioproject", "genotype_stratum", "clone", "ingestion_qc_pass",
           "mapping_ambiguous", "post_trim_length", "protocol", "umi_status", "source_molecule")
UNIVEC_TAG = re.compile(r"(?:^|\s)univec=(keep|trim|split):(\d+)-(\d+)/(\d+)(?:\s|$)")
PART = re.compile(r"(.+?)_part\d+")
ZMW = re.compile(r"([^/\s]+)/(\d+)/ccs(?:/\S*)?")
ZMW_SRA = re.compile(r"([^/\s]+)/(\d+)/ccs(?:/\d+)?")
PRJ = re.compile(r"PRJ[A-Z]{1,2}\d+")
GENOTYPES = ("reference", "known_nonreference", "hybrid_pooled", "unknown")
# DATASET_ROLES spells the A18.5 "known non-reference" stratum `non_reference` (A47.3, PROTOCOL_B1 line 1112; 77 rows
# of DATASET_ROLES.longread_v1.tsv, 74 of them c2_training_eligible); the frozen c0-v1 schema names it
# `known_nonreference` (c0_schema.GENOTYPE_STRATA).  The manifest token is normalised HERE, at ingestion, so every
# downstream validator (c0_schema CHECK, b6_build_c0.process_read) sees the canonical value; no other alias exists.
GENOTYPE_ALIASES = {"non_reference": "known_nonreference"}


def canonical_genotype(value):
    """DATASET_ROLES genotype_stratum -> c0-v1 token; None when the value is in neither vocabulary."""
    value = GENOTYPE_ALIASES.get(value, value)
    return value if value in GENOTYPES else None


class MetadataError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise MetadataError(message)


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def role_row(roles, dataset, run):
    with open(roles, newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["dataset"] == dataset and (run is None or r["run"] == run)]
    require(len(rows) == 1, f"{roles}: {len(rows)} DATASET_ROLES rows for {dataset}/{run}, expected exactly 1")
    row = dict(rows[0])
    genotype = canonical_genotype(row["genotype_stratum"])
    require(genotype is not None, f"{dataset}/{run}: genotype_stratum {row['genotype_stratum']!r} is not one of {GENOTYPES} "
                                  f"nor a manifest alias {sorted(GENOTYPE_ALIASES)}")
    row["genotype_stratum"] = genotype
    return row


def read_headers(path):
    """(first header token, second header token | None) of every FASTA/FASTQ record (streamed; FASTQ by 4-line
    records).  The first token is minimap2's SAM QNAME."""
    def split(header):
        tokens = header.split()
        require(tokens, f"{path}: empty record header")
        return tokens[0], (tokens[1] if len(tokens) > 1 else None)

    with _open(path) as fh:
        first = fh.readline()
        if not first:
            return
        if first.startswith(">"):
            yield split(first[1:])
            for line in fh:
                if line.startswith(">"):
                    yield split(line[1:])
            return
        require(first.startswith("@"), f"{path}: neither FASTA nor FASTQ")
        header = first
        while header:
            require(header.startswith("@"), f"{path}: malformed FASTQ record header {header[:40]!r}")
            yield split(header[1:])
            for _ in range(3):
                fh.readline()
            header = fh.readline()


def read_names(path):
    """First header token of every FASTA/FASTQ record (streamed)."""
    for name, _ in read_headers(path):
        yield name


def pacbio_zmw(name, second):
    """A16 source molecule '<movie>/<zmw>' of a PacBio read.  SRA-renamed reads ('>SRR35145693.1
    m54283U_201030_021721/11/ccs/1') carry it in the SECOND header token ('/ccs' or '/ccs/<n>' suffix); a native read
    name '<movie>/<zmw>/ccs[/...]' carries it in the first.  Anything else fails closed."""
    m = ZMW.fullmatch(name)
    m2 = ZMW_SRA.fullmatch(second) if second is not None else None
    if m is not None:
        require(m2 is None or (m[1], m[2]) == (m2[1], m2[2]), f"{name}: read name and second token name different ZMWs")
        return f"{m[1]}/{m[2]}"
    require(second is not None, f"{name}: no second header token '<movie>/<zmw>/ccs[/<n>]' and the read name is not a "
                                "'<movie>/<zmw>/ccs' PacBio name; the source ZMW molecule cannot be established (A16)")
    require(m2 is not None, f"{name}: second header token {second!r} is not '<movie>/<zmw>/ccs[/<n>]'; the source ZMW "
                            "molecule cannot be established (A16)")
    return f"{m2[1]}/{m2[2]}"


def screened_records(path):
    """(name, header rest, sequence length) of the A21-screened FASTA."""
    name, rest, n = None, "", 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, rest, n
                parts = line[1:].rstrip("\n").split(" ", 1)
                name, rest, n = parts[0], (parts[1] if len(parts) > 1 else ""), 0
            else:
                n += len(line.strip())
    if name is not None:
        yield name, rest, n


def genbank_sources(paths):
    """versioned accession -> (clone_lib | None, clone | None, BioProject | None) from GenBank flat files (streamed)."""
    out = {}
    for path in paths:
        acc = lib = clone = bp = None
        in_source, qual = False, None
        quals = {}
        with _open(path) as fh:
            for line in fh:
                if line.startswith("LOCUS"):
                    acc, lib, clone, bp, in_source, qual, quals = None, None, None, None, False, None, {}
                elif line.startswith("VERSION"):
                    acc = line.split()[1] if len(line.split()) > 1 else None
                elif line.startswith("DBLINK") or (line.startswith("            ") and "BioProject:" in line):
                    m = re.search(r"BioProject:\s*(\S+)", line)
                    bp = m[1].rstrip(",") if m else bp
                elif line.startswith("     source "):
                    in_source = True
                elif in_source and line.startswith("     ") and not line.startswith("                     "):
                    in_source = False                         # next feature key
                elif in_source and line.startswith("                     "):
                    text = line.strip()
                    if text.startswith("/"):
                        key, _, value = text[1:].partition("=")
                        qual = key
                        quals[key] = value
                    elif qual is not None:
                        quals[qual] += " " + text
                elif line.startswith("//"):
                    require(acc, f"{path}: GenBank record without VERSION")
                    require(acc not in out, f"{path}: duplicate GenBank record {acc}")
                    unq = {k: v.strip().strip('"') for k, v in quals.items()}
                    out[acc] = (sys.intern(unq["clone_lib"]) if unq.get("clone_lib") else None, unq.get("clone") or None, bp)
                    acc, in_source, qual, quals = None, False, None, {}
    return out


EST_META_COLUMNS = ("accession_version", "clone_lib", "clone", "biosample")
CLONE_IN_HEADER = re.compile(r"cDNA clone (\S.*)")


def verify_sha256_sidecar(path):
    """``<path>.sha256`` ('<hex>' or sha256sum's '<hex>  <name>') must match the file."""
    side = Path(str(path) + ".sha256")
    require(Path(path).is_file() and side.is_file(), f"missing {path} or its sha256 sidecar {side.name}")
    tokens = side.read_text().split()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    require(tokens and tokens[0] == h.hexdigest(), f"{path}: sha256 {h.hexdigest()} != its sidecar {side.name}")
    return h.hexdigest()


def est_meta_sources(path):
    """versioned accession -> (clone_lib | None, clone | None, None) from the NCBI esummary v2.0 TSV (--est-meta,
    columns EST_META_COLUMNS; sha256 sidecar verified).  These records carry no BioProject (--bioproject is required)."""
    verify_sha256_sidecar(path)
    out = {}
    with _open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in EST_META_COLUMNS if c not in (reader.fieldnames or [])]
        require(not missing, f"{path}: --est-meta lacks column(s) {missing}")
        for lineno, r in enumerate(reader, 2):
            acc = (r["accession_version"] or "").strip()
            require(acc, f"{path}:{lineno}: empty accession_version")
            require(acc not in out, f"{path}: duplicate accession_version {acc}")
            lib, clone = (r["clone_lib"] or "").strip(), (r["clone"] or "").strip()
            out[acc] = (sys.intern(lib) if lib else None, clone or None, None)
    return out


def check_header_clone(name, rest, clone):
    """Where the screened FASTA header says 'cDNA clone <X>', X must begin with the recorded clone (then end, space,
    comma or semicolon).  A header clone without a recorded clone is allowed (no clone is inferred from headers)."""
    m = CLONE_IN_HEADER.search(rest)
    if m is None or not clone:
        return
    text = m[1]
    # NCBI renders '>' as '_' in FASTA titles (e.g. DN591027.1), while GenBank /clone retains it.
    # Compare display forms only; the original qualifier remains the A11 clone identity in est_rows.
    ok = any(text.startswith(display) and (len(text) == len(display) or text[len(display)] in " ,;")
             for display in (clone, clone.replace(">", "_")))
    require(ok, f"{name}: FASTA header 'cDNA clone {text[:60]}' disagrees with the recorded clone {clone!r}")


def est_rows(args):
    row = role_row(args.roles, f"est/{args.species}", None)
    gb = genbank_sources(args.genbank) if args.genbank else est_meta_sources(args.est_meta)
    source = "GenBank" if args.genbank else "--est-meta"
    yield {"read": "*", "genotype_stratum": row["genotype_stratum"], "ingestion_qc_pass": "true",
           "mapping_ambiguous": "false", **({"bioproject": args.bioproject} if args.bioproject else {})}
    for name, rest, n in screened_records(args.screened):
        m = UNIVEC_TAG.search(rest)
        require(m is not None, f"{args.screened}: record {name} has no 'univec=<action>:<a>-<b>/<qlen>' tag (A21)")
        length = int(m[3]) - int(m[2]) + 1
        require(length == n, f"{name}: univec tag length {length} != sequence length {n}")
        base = PART.fullmatch(name)[1] if PART.fullmatch(name) else name
        require(base in gb, f"{name}: accession {base} has no {source} record (library/clone unknown)")
        lib, clone, bp = gb[base]
        require(lib, f"{name}: {source} {base} has no /clone_lib qualifier; the library cannot be established")
        require(bp or args.bioproject, f"{name}: {source} {base} has no BioProject DBLINK and no --bioproject was given")
        check_header_clone(name, rest, clone)
        yield {"read": name, "accession": name, "library": lib, "clone": clone or "", "post_trim_length": str(length),
               **({"bioproject": bp} if bp else {})}


# A49.1 decision 4 (author 2026-09-23): the long-read library key is the ENA/SRA experiment accession, or
# '<sample accession>:<ENA library_name>' where one (sample, library_name) spans several experiments (JGI registers one
# experiment per SMRT cell).  library_name is free text (spaces allowed; no tab / CR / LF).  Nothing else is accepted.


def longread_library(args):
    """The run's biological library identity (A11): the explicit --library, never the run accession (docstring)."""
    library = (args.library or "").strip()
    require(library, f"{args.dataset}/{args.run}: --library is required for --source {args.source} (the ENA/SRA "
                     "experiment accession of the run or an author library key; A11: runs do not create libraries)")
    require(library != "unknown", f"{args.dataset}/{args.run}: --library 'unknown' is not a library identity")
    require(library != args.run, f"{args.dataset}/{args.run}: --library equals the run accession; a sequencing run is "
                                 "not a biological library (A11) -- give the experiment accession or the author's "
                                 "library key")
    require(S.is_longread_library(library),
            f"{args.dataset}/{args.run}: --library {library!r} is neither an ENA/SRA experiment accession "
            "(SRX/ERX/DRX<digits>) nor '<BioSample>:<library_name>' (SAMN/SAME/SAMD...:<name>, used where one prep spans "
            "several experiments; A49.1 decision 4)")
    return library


def longread_rows(args):
    row = role_row(args.roles, args.dataset, args.run)
    bp = args.bioproject or next(iter(PRJ.findall(args.dataset)), None)
    require(bp, f"{args.dataset}: no PRJ* accession in the dataset path and no --bioproject")
    star = {"read": "*", "library": longread_library(args), "bioproject": bp, "genotype_stratum": row["genotype_stratum"],
            "ingestion_qc_pass": "true", "mapping_ambiguous": "false"}
    if args.source == "ont":
        require(args.protocol and args.umi_status, "--protocol and --umi-status are required for ONT ('unresolved' allowed)")
        star["protocol"] = "" if args.protocol == "unresolved" else args.protocol
        star["umi_status"] = "" if args.umi_status == "unresolved" else args.umi_status
        require(star["umi_status"] != "present", "UMI-present runs need per-read sample/umi/locus, not produced here")
    yield star
    for name, second in read_headers(args.reads):
        if args.source == "pacbio":
            yield {"read": name, "source_molecule": pacbio_zmw(name, second)}
        else:
            yield {"read": name}


def write(rows, out, provenance):
    out = Path(out)
    for p in (out, Path(str(out) + ".sha256")):
        if os.path.lexists(p):
            raise FileExistsError(p)
    partial = out.with_name(out.name + ".partial")
    n = 0
    try:
        with open(partial, "x") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for r in rows:
                vals = [str(r.get(c, "")) for c in COLUMNS]
                require(not any(c in v for v in vals for c in "\t\r\n"), f"{r['read']}: tab/CR/LF in a field")
                fh.write("\t".join(vals) + "\n")
                n += r["read"] != "*"
        require(n > 0, "no read written")
        os.link(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{h.hexdigest()}  {out.name}\n")
    with open(str(out) + ".PROVENANCE.json", "x") as fh:
        json.dump(dict(provenance, reads=n, output_sha256=h.hexdigest()), fh, indent=1, sort_keys=True, default=str)
    return {"output": str(out), "sha256": h.hexdigest(), "reads": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=("est", "ont", "pacbio"))
    ap.add_argument("--species", required=True)
    ap.add_argument("--roles", required=True, help="DATASET_ROLES TSV holding the run's row")
    ap.add_argument("--screened", help="EST: A21-screened primary-arm FASTA")
    ap.add_argument("--genbank", nargs="+", help="EST: GenBank flat files (rettype=gb) of the raw ESTs")
    ap.add_argument("--est-meta", help="EST: NCBI esummary v2.0 TSV (accession_version, clone_lib, clone, biosample; "
                                       "<tsv>.sha256 verified), the alternative to --genbank; needs --bioproject")
    ap.add_argument("--dataset", help="long reads: DATASET_ROLES dataset")
    ap.add_argument("--run", help="long reads: run accession (a sequencing run, never a library)")
    ap.add_argument("--library", help="long reads: the run's biological library (A11) = its ENA/SRA experiment "
                                      "accession (SRX/ERX/DRX<digits>) or <BioSample>:<library_name> (A49.1); required, never the run")
    ap.add_argument("--reads", help="long reads: the exact reads file the frozen alignment used")
    ap.add_argument("--protocol", choices=("cDNA", "direct_RNA", "unresolved"))
    ap.add_argument("--umi-status", choices=("present", "absent", "unresolved"))
    ap.add_argument("--bioproject")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.source == "est":
            require(args.screened and (bool(args.genbank) != bool(args.est_meta)),
                    "--source est needs --screened and exactly one of --genbank / --est-meta")
            # esummary records carry no BioProject; it is never inferred (author decision pending): explicit per species.
            require(not args.est_meta or args.bioproject, "--est-meta needs --bioproject (the esummary TSV has none)")
            rows = est_rows(args)
        else:
            require(args.dataset and args.run and args.reads,
                    f"--source {args.source} needs --dataset, --run, --library and --reads")
            rows = longread_rows(args)
        prov = {k: v for k, v in vars(args).items()}
        if args.source != "est":
            prov["library"] = longread_library(args)      # validated before any file is written; = the '*' row value
        if args.source == "est" and args.est_meta:
            prov["est_meta_sha256"] = verify_sha256_sidecar(args.est_meta)
        result = write(rows, args.out, prov)
    except (MetadataError, FileExistsError, OSError, KeyError) as exc:
        print(f"b6_c0_metadata failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
