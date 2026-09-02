"""B5 database builder — production path for docs/gsf_spec_v1.md §6–§8 (issue #12/#15/#16).

Torch-free: uses gsf_contract for every GSF string, the frozen split table for every row, and
parameterised DuckDB inserts. Keeps the legacy geneList columns so datasets.py keeps working and
adds the provenance/split columns of the spec. `genome2GSFDataset` in preprocess.py delegates here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import duckdb


def _load_gsf_contract():
    """Import gsf_contract without dragging torch in through utils/__init__ (works when loaded by path too)."""
    try:
        from ..utils import gsf_contract as gc  # type: ignore
        return gc
    except Exception:
        import pathlib
        import types
        path = pathlib.Path(__file__).resolve().parents[1] / "utils" / "gsf_contract.py"
        mod = sys.modules.get("gsf_contract")
        if mod is None:
            mod = types.ModuleType("gsf_contract")
            mod.__file__ = str(path)
            sys.modules["gsf_contract"] = mod
            exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
        return mod


gc = _load_gsf_contract()

LEGACY_COLUMNS = ("geneModel", "start", "fin", "strand", "chromosome", "sequence", "gff",
                  "static_fpb", "static_tpb", "five_prime_buf", "three_prime_buf")
NEW_COLUMNS = ("species_id", "gene_id", "orthogroup_id", "split", "strict_holdout", "is_rc", "ordering_version",
               "build_version", "source_fasta_sha256", "source_gff_sha256", "split_file_sha256", "window_policy",
               "gsf_token_count", "contig_boundary", "n_transcripts")
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
STOP = {"TAA", "TAG", "TGA"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    out = {k: "".join(v) for k, v in seqs.items()}
    for k, v in list(out.items()):  # accept "1" and "Chr1" for the same contig
        out.setdefault(k[3:] if k.startswith("Chr") else f"Chr{k}", v)
    return out


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def validate_cds(gsf: str, seq: str) -> Tuple[bool, str]:
    """Every transcript's spliced CDS starts with ATG, ends with a stop and is a multiple of 3."""
    feats, txs, strand = gc._parse(gsf)
    for tx in txs:
        cds = [f for f in gc._transcription_order(tx, strand) if f[0] == "CDS"]
        if not cds:
            continue
        pieces = [seq[s:e] for (_t, s, e, _p) in cds]
        spliced = "".join(pieces) if strand == "+" else "".join(revcomp(p) for p in pieces)
        if len(spliced) % 3:
            return False, "CDS length not a multiple of 3"
        if not spliced.upper().startswith("ATG"):
            return False, "no ATG start"
        if spliced[-3:].upper() not in STOP:
            return False, "no stop codon"
    return True, ""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("CREATE TABLE IF NOT EXISTS geneList ("
            "geneModel VARCHAR, start INT, fin INT, strand VARCHAR, chromosome VARCHAR, sequence VARCHAR, gff VARCHAR, "
            "static_fpb INT, static_tpb INT, five_prime_buf INT, three_prime_buf INT, rn INT PRIMARY KEY, "
            "species_id VARCHAR, gene_id VARCHAR, orthogroup_id VARCHAR, split VARCHAR, strict_holdout BOOLEAN, is_rc BOOLEAN, "
            "ordering_version VARCHAR, build_version VARCHAR, source_fasta_sha256 VARCHAR, source_gff_sha256 VARCHAR, "
            "split_file_sha256 VARCHAR, window_policy VARCHAR, gsf_token_count INT, contig_boundary BOOLEAN, n_transcripts INT)")
    con.sql("CREATE SEQUENCE IF NOT EXISTS row_id START 1")
    con.sql("CREATE TABLE IF NOT EXISTS gene_split (species_id VARCHAR, gene_id VARCHAR, orthogroup_id VARCHAR, split VARCHAR, "
            "strict_holdout BOOLEAN, seed INT, source_version VARCHAR)")
    con.sql("CREATE TABLE IF NOT EXISTS build_manifest (species_id VARCHAR, fasta VARCHAR, fasta_sha256 VARCHAR, gff VARCHAR, "
            "gff_sha256 VARCHAR, split_file_sha256 VARCHAR, rc_mode VARCHAR, rows_inserted INT, rows_rc INT, rejected INT, "
            "rejected_reasons VARCHAR, build_version VARCHAR, ordering_version VARCHAR, window_policy VARCHAR, "
            "git_commit VARCHAR, built_at VARCHAR, duckdb_version VARCHAR)")


def read_split_table(path: str) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], str]:
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[(r["species_id"], r["gene_id"])] = r
    return rows, sha256(path)


def _window_for(gene: gc.Gene, chrom_len: int, max_len: int):
    ws, we = gc.pad_window(gene.start0, gene.end0)
    boundary = False
    if we > chrom_len:
        shift = we - chrom_len
        ws, we = max(0, ws - shift), chrom_len
        boundary = True
    if ws == 0 and gene.start0 - ws < (we - ws - (gene.end0 - gene.start0)) // 2:
        boundary = boundary or ws == 0
    return ws, we, boundary


def build_species(con, species_id: str, fasta: str, gff: str, split_rows: Dict[Tuple[str, str], Dict[str, str]], split_sha: str,
                  rc: str = "none", add_extra: int = 0, seed: int = 123, max_len: int = gc.MAX_WINDOW, clean: bool = False,
                  mode: str = "train", git_commit: str = "", fasta_sha: Optional[str] = None, gff_sha: Optional[str] = None,
                  allow_missing_split: bool = False) -> Dict:
    if rc not in gc.RC_MODES:
        raise ValueError(f"rc must be one of {gc.RC_MODES}")
    ensure_schema(con)
    genome = load_fasta(fasta)
    rng = random.Random(seed)
    fasta_sha = fasta_sha or sha256(fasta)
    gff_sha = gff_sha or sha256(gff)
    inserted = rc_rows = 0
    rejected: List[Dict] = []
    cols = ", ".join(LEGACY_COLUMNS + NEW_COLUMNS)
    placeholders = ", ".join("?" for _ in LEGACY_COLUMNS + NEW_COLUMNS)
    sql = f"INSERT INTO geneList (rn, {cols}) VALUES (nextval('row_id'), {placeholders})"
    with open(gff) as fh:
        for gene in gc.parse_gff3(fh):
            key = (species_id, gene.gene_id)
            if key not in split_rows:
                if not allow_missing_split:
                    raise gc.SplitError(f"{species_id}:{gene.gene_id} has no split entry")
                srow = {"split": None, "orthogroup_id": None, "strict_holdout": ""}
            else:
                srow = split_rows[key]
            if gene.chrom not in genome:
                rejected.append({"gene_id": gene.gene_id, "reason": f"chromosome {gene.chrom} missing from FASTA"})
                continue
            chrom_len = len(genome[gene.chrom])
            ws, we, boundary = _window_for(gene, chrom_len, max_len)
            L = we - ws
            if L > max_len:
                rejected.append({"gene_id": gene.gene_id, "reason": f"window {L} > {max_len}"})
                continue
            gsf = gc.gene_to_gsf(gene, ws) if mode == "train" else None
            if gsf is not None:
                try:
                    gc.check_caps(gsf, window_len=L)
                except gc.CapError as e:
                    rejected.append({"gene_id": gene.gene_id, "reason": str(e)})
                    continue
            seq = genome[gene.chrom][ws:we]
            if clean and gsf is not None:
                ok, why = validate_cds(gsf, seq)
                if not ok:
                    rejected.append({"gene_id": gene.gene_id, "reason": why})
                    continue
            fpb, tpb = gene.start0 - ws, we - gene.end0
            five, three = (rng.randrange(add_extra), rng.randrange(add_extra)) if add_extra else (0, 0)
            common = [species_id, gene.gene_id, srow.get("orthogroup_id") or None, srow["split"],
                      str(srow.get("strict_holdout", "")).lower() in ("1", "true", "yes"), False, gc.ORDERING_VERSION,
                      gc.BUILD_VERSION, fasta_sha, gff_sha, split_sha, gc.WINDOW_POLICY,
                      gc.count_tokens_v2(gsf) if gsf else None, boundary, len(gene.transcripts)]
            con.execute(sql, [gene.gene_id, ws, we, gene.strand, gene.chrom, seq, gsf, fpb, tpb, five, three] + common)
            inserted += 1
            want_rc = gsf is not None and (rc == "all" or (rc == "isoform-only" and len(gene.transcripts) >= 2))
            if want_rc:
                rgsf = gc.reverse_complement(gsf, L)
                rc_common = list(common)
                rc_common[5] = True
                rc_common[12] = gc.count_tokens_v2(rgsf)
                con.execute(sql, [f"{gene.gene_id}-rc", ws, we, "-" if gene.strand == "+" else "+", gene.chrom, revcomp(seq), rgsf,
                                  tpb, fpb, five, three] + rc_common)
                inserted += 1
                rc_rows += 1
    reasons: Dict[str, int] = {}
    for r in rejected:
        k = r["reason"].split(" ")[0]
        reasons[k] = reasons.get(k, 0) + 1
    con.execute("INSERT INTO build_manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [species_id, os.path.abspath(fasta), fasta_sha, os.path.abspath(gff), gff_sha, split_sha, rc, inserted, rc_rows,
                 len(rejected), json.dumps(reasons), gc.BUILD_VERSION, gc.ORDERING_VERSION, gc.WINDOW_POLICY, git_commit,
                 time.strftime("%Y-%m-%dT%H:%M:%S"), duckdb.__version__])
    return {"species_id": species_id, "rows": inserted, "rc_rows": rc_rows, "rejected": rejected}


def read_species_manifest(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build_b5_database(db: str, species_manifest: str, split_table: str, rc: str = "isoform-only", add_extra: int = 0, seed: int = 123,
                      max_len: int = gc.MAX_WINDOW, clean: bool = False, excluded_species: Iterable[str] = ("Zmays",),
                      verify_md5: bool = True, only_species: Optional[Set[str]] = None, git_commit: str = "") -> List[Dict]:
    manifest = read_species_manifest(species_manifest)
    split_rows, split_sha = read_split_table(split_table)
    excluded = set(excluded_species)
    con = duckdb.connect(db)
    ensure_schema(con)
    if con.sql("SELECT count(*) FROM gene_split").fetchone()[0] == 0:
        con.executemany("INSERT INTO gene_split VALUES (?,?,?,?,?,?,?)",
                        [[r["species_id"], r["gene_id"], r.get("orthogroup_id") or None, r["split"],
                          str(r.get("strict_holdout", "")).lower() in ("1", "true", "yes"), int(r.get("seed", 0) or 0), r.get("source_version", "")]
                         for r in split_rows.values()])
    results = []
    for m in manifest:
        sid = m["species_id"]
        if sid in excluded:
            raise ValueError(f"excluded species {sid} present in the species manifest")
        if only_species and sid not in only_species:
            continue
        fsha, gsha = sha256(m["fasta"]), sha256(m["gff"])
        if verify_md5 and (m.get("fasta_md5") or m.get("gff_md5")):
            fmd5 = hashlib.md5(open(m["fasta"], "rb").read()).hexdigest()
            gmd5 = hashlib.md5(open(m["gff"], "rb").read()).hexdigest()
            if m.get("fasta_md5") and fmd5 != m["fasta_md5"]:
                raise ValueError(f"{sid}: FASTA md5 {fmd5} != manifest {m['fasta_md5']}")
            if m.get("gff_md5") and gmd5 != m["gff_md5"]:
                raise ValueError(f"{sid}: GFF md5 {gmd5} != manifest {m['gff_md5']}")
        print(f"[{sid}] building...", file=sys.stderr)
        results.append(build_species(con, sid, m["fasta"], m["gff"], split_rows, split_sha, rc=rc, add_extra=add_extra, seed=seed,
                                     max_len=max_len, clean=clean, git_commit=git_commit, fasta_sha=fsha, gff_sha=gsha))
    con.close()
    return results


def validate_b5_database(db: str, excluded_species: Iterable[str] = ("Zmays",), maize_patterns: Tuple[str, ...] = ("Zm", "GRMZM")) -> Dict:
    con = duckdb.connect(db, read_only=True)
    rows = [dict(zip(["species_id", "gene_id", "orthogroup_id", "split", "is_rc", "strict_holdout"], r)) for r in
            con.sql("SELECT species_id, gene_id, orthogroup_id, split, is_rc, strict_holdout FROM geneList").fetchall()]
    violations = gc.validate_split(rows, excluded_species=set(excluded_species))
    for pat in maize_patterns:
        n = con.sql(f"SELECT count(*) FROM geneList WHERE geneModel LIKE '{pat}%'").fetchone()[0]
        if n:
            violations.append(f"{n} rows whose geneModel starts with {pat}")
    over = con.sql(f"SELECT count(*) FROM geneList WHERE gsf_token_count > {gc.CAPS['tokens']}").fetchone()[0]
    if over:
        violations.append(f"{over} rows exceed the token cap")
    bad_null = con.sql("SELECT count(*) FROM geneList WHERE gff = '' OR gff = 'None'").fetchone()[0]
    if bad_null:
        violations.append(f"{bad_null} rows store empty-string labels instead of NULL")
    counts = {k: v for k, v in con.sql("SELECT split, count(*) FROM geneList GROUP BY split").fetchall()}
    per_species = {k: v for k, v in con.sql("SELECT species_id, count(*) FROM geneList GROUP BY species_id").fetchall()}
    con.close()
    return {"violations": violations, "rows_by_split": counts, "rows_by_species": per_species, "ok": not violations}
