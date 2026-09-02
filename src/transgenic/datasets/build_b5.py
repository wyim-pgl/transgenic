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
NEW_COLUMNS = ("species_id", "gene_id", "orthogroup_id", "split", "strict_holdout", "is_rc", "ordering_version", "build_version",
               "source_fasta_sha256", "source_gff_sha256", "split_file_sha256", "window_policy", "gsf_token_count", "contig_boundary", "n_transcripts",
               "train_weight", "qc_flags", "gene_id_original")
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
        if any(spliced[i:i + 3].upper() in STOP for i in range(0, len(spliced) - 3, 3)):
            return False, "internal stop codon"
    return True, ""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("CREATE TABLE IF NOT EXISTS geneList ("
            "geneModel VARCHAR, start INT, fin INT, strand VARCHAR, chromosome VARCHAR, sequence VARCHAR, gff VARCHAR, "
            "static_fpb INT, static_tpb INT, five_prime_buf INT, three_prime_buf INT, rn INT PRIMARY KEY, "
            "species_id VARCHAR, gene_id VARCHAR, orthogroup_id VARCHAR, split VARCHAR, strict_holdout BOOLEAN, is_rc BOOLEAN, "
            "ordering_version VARCHAR, build_version VARCHAR, source_fasta_sha256 VARCHAR, source_gff_sha256 VARCHAR, "
            "split_file_sha256 VARCHAR, window_policy VARCHAR, gsf_token_count INT, contig_boundary BOOLEAN, n_transcripts INT, "
            "train_weight DOUBLE DEFAULT 1.0, qc_flags VARCHAR, gene_id_original VARCHAR)")
    con.sql("CREATE SEQUENCE IF NOT EXISTS row_id START 1")
    con.sql("CREATE TABLE IF NOT EXISTS gene_key_map (species_id VARCHAR, gene_id VARCHAR, gene_id_original VARCHAR, name_original VARCHAR, "
            "chromosome VARCHAR, start0 INT, end0 INT)")
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


def _window_for(gene: gc.Gene, chrom_len: int, max_len: int, policy: str = gc.WINDOW_POLICY, rng=None, tier_up_prob: float = 0.0):
    """Symmetric window; when the contig is too short the window is shifted, and if the contig is shorter than the
    window it keeps its multiple-of-6144 length by N-padding the sequence on the right (contig_boundary = True)."""
    if policy == gc.WINDOW_POLICY_V2:
        ws, we, _tier = gc.pad_window_tiered(gene.start0, gene.end0, rng=rng, tier_up_prob=tier_up_prob)
    else:
        ws, we = gc.pad_window(gene.start0, gene.end0)
    L = we - ws
    boundary = False
    if we > chrom_len:
        shift = we - chrom_len
        ws = max(0, ws - shift)
        we = ws + L
        boundary = True
    if gene.end0 > chrom_len:
        raise ValueError(f"{gene.gene_id}: annotation end {gene.end0} beyond contig length {chrom_len}")
    return ws, we, boundary


def loss_mask_decision(gene_flags: Dict[str, set], transcript_ids) -> Tuple[float, List[str], List[str]]:
    """Protocol A22 (GeenuFF flags): hard flag on the gene or on every transcript -> train_weight 0 (row kept, masked);
    hard flags on some transcripts -> those transcripts leave the label; soft flags -> nothing changes."""
    try:
        from .qc_flags import is_hard  # type: ignore
    except Exception:
        import pathlib
        ns: Dict = {}
        exec(compile((pathlib.Path(__file__).resolve().parent / "qc_flags.py").read_text(), "qc_flags.py", "exec"), ns)
        is_hard = ns["is_hard"]
    tids = list(transcript_ids)
    hard_gene = {f for f in gene_flags.get("*", set()) if is_hard(f)}
    hard_by_tx = {t: {f for f in gene_flags.get(t, set()) if is_hard(f)} for t in tids}
    all_hard = sorted(hard_gene | {f for v in hard_by_tx.values() for f in v})
    if hard_gene:
        return 0.0, [], all_hard
    keep = [t for t in tids if not hard_by_tx[t]]
    if not keep:
        return 0.0, [], all_hard
    return 1.0, keep, all_hard


def read_qc_flags(path) -> Dict[Tuple[str, str], Dict[str, set]]:
    """Read one or several A22-schema flag files (species_id, gene_id, transcript_id, flag, start, end) and merge them:
    the GeenuFF file of 62_geenuff_qc.py (A22) and the Swiss-Prot caution audit of 63_swissprot_sensitivity.py (A30)."""
    paths = [path] if isinstance(path, (str, bytes)) or hasattr(path, "__fspath__") else list(path)
    out: Dict[Tuple[str, str], Dict[str, set]] = {}
    for p in paths:
        with open(p) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            idx = {h: i for i, h in enumerate(header)}
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) < 4:
                    continue
                key = (c[idx["species_id"]], c[idx["gene_id"]])
                tx = c[idx["transcript_id"]] or "*"
                out.setdefault(key, {}).setdefault(tx, set()).add(c[idx["flag"]])
    return out


def build_species(con, species_id: str, fasta: str, gff: str, split_rows: Dict[Tuple[str, str], Dict[str, str]], split_sha: str,
                  rc: str = "none", add_extra: int = 0, seed: int = 123, max_len: int = gc.MAX_WINDOW, clean: bool = False,
                  mode: str = "train", git_commit: str = "", fasta_sha: Optional[str] = None, gff_sha: Optional[str] = None,
                  allow_missing_split: bool = False, qc_flags: Optional[Dict[Tuple[str, str], Dict[str, set]]] = None,
                  window_policy: str = gc.WINDOW_POLICY, tier_up_prob: float = 0.0) -> Dict:
    if rc not in gc.RC_MODES:
        raise ValueError(f"rc must be one of {gc.RC_MODES}")
    ensure_schema(con)
    genome = load_fasta(fasta)
    rng = random.Random(seed)
    fasta_sha = fasta_sha or sha256(fasta)
    gff_sha = gff_sha or sha256(gff)
    inserted = rc_rows = 0
    rejected: List[Dict] = []
    con.execute("BEGIN TRANSACTION")
    try:
        return _build_species_body(con, species_id, fasta, gff, split_rows, split_sha, rc, add_extra, seed, max_len, clean, mode,
                                   git_commit, fasta_sha, gff_sha, allow_missing_split, qc_flags, genome, rng, inserted, rc_rows, rejected,
                                   window_policy, tier_up_prob)
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _build_species_body(con, species_id, fasta, gff, split_rows, split_sha, rc, add_extra, seed, max_len, clean, mode, git_commit,
                        fasta_sha, gff_sha, allow_missing_split, qc_flags, genome, rng, inserted, rc_rows, rejected,
                        window_policy=gc.WINDOW_POLICY, tier_up_prob=0.0):
    if window_policy not in (gc.WINDOW_POLICY, gc.WINDOW_POLICY_V2, gc.WINDOW_POLICY_V3):
        raise ValueError(f"unknown window policy {window_policy}")
    if window_policy == gc.WINDOW_POLICY_V3:
        return _build_species_tiles(con, species_id, fasta, gff, split_rows, split_sha, rc, seed, clean, mode, git_commit,
                                    fasta_sha, gff_sha, allow_missing_split, qc_flags, genome, rng, rejected, tier_up_prob)
    if window_policy == gc.WINDOW_POLICY_V2 and max_len < gc.MAX_WINDOW_V2:
        max_len = gc.MAX_WINDOW_V2
    cols = ", ".join(LEGACY_COLUMNS + NEW_COLUMNS)
    placeholders = ", ".join("?" for _ in LEGACY_COLUMNS + NEW_COLUMNS)
    sql = f"INSERT INTO geneList (rn, {cols}) VALUES (nextval('row_id'), {placeholders})"
    with open(gff) as fh:
        for gene in gc.parse_gff3(fh, species_code=gc.species_code(species_id)):
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
            con.execute("INSERT INTO gene_key_map VALUES (?,?,?,?,?,?,?)", [species_id, gene.gene_id, gene.gene_id_original, gene.name_original,
                                                                          gene.chrom, gene.start0, gene.end0])
            # A22: annotation-quality flags -> loss masking / transcript filtering
            train_weight, qc_list = 1.0, []
            if qc_flags and (species_id, gene.gene_id) in qc_flags:
                train_weight, keep_tx, qc_list = loss_mask_decision(qc_flags[(species_id, gene.gene_id)], gene.transcripts.keys())
                if train_weight > 0 and len(keep_tx) < len(gene.transcripts):
                    gene = gc.Gene(gene.gene_id, gene.chrom, gene.strand, gene.start0, gene.end0, {t: gene.transcripts[t] for t in keep_tx},
                                   gene.gene_id_original, gene.name_original)
                    qc_list = qc_list + ["transcripts_dropped"]
            chrom_len = len(genome[gene.chrom])
            ws, we, boundary = _window_for(gene, chrom_len, max_len, policy=window_policy, rng=rng if mode == "train" else None,
                                           tier_up_prob=tier_up_prob if mode == "train" else 0.0)
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
            if len(seq) < L:
                seq = seq + "N" * (L - len(seq))   # contig shorter than the window: keep the 6144-multiple length
            if clean and gsf is not None:
                ok, why = validate_cds(gsf, seq)
                if not ok:
                    rejected.append({"gene_id": gene.gene_id, "reason": why})
                    continue
            fpb, tpb = gene.start0 - ws, we - gene.end0
            five, three = (rng.randrange(add_extra), rng.randrange(add_extra)) if add_extra else (0, 0)
            common = [species_id, gene.gene_id, srow.get("orthogroup_id") or None, srow["split"],
                      str(srow.get("strict_holdout", "")).lower() in ("1", "true", "yes"), False, gc.ORDERING_VERSION,
                      gc.BUILD_VERSION, fasta_sha, gff_sha, split_sha, window_policy,
                      gc.count_tokens_v2(gsf) if gsf else None, boundary, len(gene.transcripts), train_weight, ";".join(qc_list) or None,
                      gene.gene_id_original]
            con.execute(sql, [gene.gene_id, ws, we, gene.strand, gene.chrom, seq, gsf, fpb, tpb, five, three] + common)
            inserted += 1
            want_rc = gsf is not None and (rc == "all" or (rc == "isoform-only" and len(gene.transcripts) >= 2))
            if want_rc:
                rgsf = gc.reverse_complement(gsf, L)
                rc_common = list(common)
                rc_common[5] = True
                rc_common[12] = gc.count_tokens_v2(rgsf)  # train_weight / qc_flags are inherited unchanged
                con.execute(sql, [f"{gene.gene_id}-rc", ws, we, "-" if gene.strand == "+" else "+", gene.chrom, revcomp(seq), rgsf,
                                  tpb, fpb, five, three] + rc_common)
                inserted += 1
                rc_rows += 1
    con.sql("CREATE TABLE IF NOT EXISTS rejected_records (species_id VARCHAR, gene_id VARCHAR, reason VARCHAR)")
    if rejected:
        con.executemany("INSERT INTO rejected_records VALUES (?,?,?)", [[species_id, r["gene_id"], r["reason"]] for r in rejected])
    reasons: Dict[str, int] = {}
    for r in rejected:
        k = r["reason"].split(":")[0].split(" >")[0]
        reasons[k] = reasons.get(k, 0) + 1
    con.execute("INSERT INTO build_manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [species_id, os.path.abspath(fasta), fasta_sha, os.path.abspath(gff), gff_sha, split_sha, rc, inserted, rc_rows,
                 len(rejected), json.dumps(reasons), gc.BUILD_VERSION, gc.ORDERING_VERSION, window_policy, git_commit,
                 time.strftime("%Y-%m-%dT%H:%M:%S"), duckdb.__version__])
    con.execute("COMMIT")
    return {"species_id": species_id, "rows": inserted, "rc_rows": rc_rows, "rejected": rejected}


def read_species_manifest(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build_b5_database(db: str, species_manifest: str, split_table: str, rc: str = "isoform-only", add_extra: int = 0, seed: int = 123,
                      max_len: int = gc.MAX_WINDOW, clean: bool = False, excluded_species: Iterable[str] = ("Zmays",),
                      verify_md5: bool = True, only_species: Optional[Set[str]] = None, git_commit: str = "",
                      qc_flags_path: Optional["str | List[str]"] = None, window_policy: str = gc.WINDOW_POLICY,
                      tier_up_prob: float = 0.0) -> List[Dict]:
    manifest = read_species_manifest(species_manifest)
    split_rows, split_sha = read_split_table(split_table)
    qc = read_qc_flags(qc_flags_path) if qc_flags_path else None
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
                                     max_len=max_len, clean=clean, git_commit=git_commit, fasta_sha=fsha, gff_sha=gsha, qc_flags=qc,
                                     window_policy=window_policy, tier_up_prob=tier_up_prob))
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
    # physical checks: window length, sequence length, stored token count, RC pairing, required fields
    for (gm, st, fn, seq_len, gsf, tok, is_rc, sp, gid, ov, pol) in con.sql(
            "SELECT geneModel, start, fin, length(sequence), gff, gsf_token_count, is_rc, species_id, gene_id, ordering_version, window_policy FROM geneList").fetchall():
        if st is None or fn is None or seq_len is None:
            violations.append(f"{gm}: missing start/fin/sequence")
            continue
        L = fn - st
        cap = gc.MAX_WINDOW_V2 if pol in (gc.WINDOW_POLICY_V2, gc.WINDOW_POLICY_V3) else gc.MAX_WINDOW
        if L % gc.WINDOW_UNIT or L > cap:
            violations.append(f"{gm}: window {L} is not an allowed multiple of {gc.WINDOW_UNIT} (policy {pol})")
        if seq_len != L:
            violations.append(f"{gm}: sequence length {seq_len} != window {L}")
        expected_tok = (gc.count_tokens_v3(gsf) if pol == gc.WINDOW_POLICY_V3 else gc.count_tokens_v2(gsf)) if gsf is not None else None
        if gsf is not None and tok != expected_tok:
            violations.append(f"{gm}: stored token count {tok} != recomputed {expected_tok}")
        if sp is None or gid is None or ov != gc.ORDERING_VERSION:
            violations.append(f"{gm}: missing species/gene id or ordering_version")
    pairs = con.sql("SELECT f.geneModel, f.gff, r.gff, f.fin - f.start, f.window_policy FROM geneList f JOIN geneList r ON r.species_id = f.species_id "
                    "AND r.gene_id = f.gene_id AND r.is_rc AND NOT f.is_rc").fetchall()
    for gm, fg, rg, L, pol in pairs:
        rcf = gc.reverse_complement_v3 if pol == gc.WINDOW_POLICY_V3 else gc.reverse_complement
        if fg is not None and rcf(fg, L) != rg:
            violations.append(f"{gm}: rc row is not the reverse complement of the forward row")
    dup = con.sql("SELECT species_id, gene_id, is_rc, count(*) c FROM geneList GROUP BY 1,2,3 HAVING c > 1").fetchall()
    for sp, gid, rcflag, c in dup:
        violations.append(f"duplicate row {sp}:{gid} is_rc={rcflag} x{c}")
    over = con.sql(f"SELECT count(*) FROM geneList WHERE gsf_token_count > {gc.CAPS['tokens']}").fetchone()[0]
    if over:
        violations.append(f"{over} rows exceed the token cap")
    bad_null = con.sql("SELECT count(*) FROM geneList WHERE gff = '' OR gff = 'None'").fetchone()[0]
    if bad_null:
        violations.append(f"{bad_null} rows store empty-string labels instead of NULL")
    masked = con.sql("SELECT count(*) FROM geneList WHERE train_weight = 0").fetchone()[0]
    counts = {k: v for k, v in con.sql("SELECT split, count(*) FROM geneList GROUP BY split").fetchall()}
    per_species = {k: v for k, v in con.sql("SELECT species_id, count(*) FROM geneList GROUP BY species_id").fetchall()}
    con.close()
    return {"violations": violations, "rows_by_split": counts, "rows_by_species": per_species, "rows_loss_masked": masked, "ok": not violations}


def _tile_split(splits: List[Optional[str]], strict: bool) -> Optional[str]:
    """A window's split is the most restrictive split of the genes it contains (test > valid > train); strict held-out -> test."""
    if strict or "test" in splits:
        return "test"
    if "valid" in splits:
        return "valid"
    if splits:
        return "train"
    return "train"  # empty windows carry no gene: usable for training the <empty> label


def _build_species_tiles(con, species_id, fasta, gff, split_rows, split_sha, rc, seed, clean, mode, git_commit, fasta_sha, gff_sha,
                         allow_missing_split, qc_flags, genome, rng, rejected, tier_up_prob):
    """tile6144-v3 (protocol A26): every tier tiles each contig with a seeded offset; the label of a tile is the canonical
    concatenation of all genes fully inside it (or <empty>). Edge-crossing genes are excluded and counted; empty tiles are kept
    with EMPTY_KEEP_PROB; a tile containing a hard-flagged (A22) gene gets train_weight 0; the tile split is the most
    restrictive split among its genes."""
    cols = ", ".join(LEGACY_COLUMNS + NEW_COLUMNS)
    placeholders = ", ".join("?" for _ in LEGACY_COLUMNS + NEW_COLUMNS)
    sql = f"INSERT INTO geneList (rn, {cols}) VALUES (nextval('row_id'), {placeholders})"
    con.sql("CREATE TABLE IF NOT EXISTS window_genes (species_id VARCHAR, window_id VARCHAR, gene_id VARCHAR, is_rc BOOLEAN)")
    by_chrom: Dict[str, List[gc.Gene]] = {}
    gene_meta: Dict[str, Dict] = {}
    with open(gff) as fh:
        for gene in gc.parse_gff3(fh, species_code=gc.species_code(species_id)):
            key = (species_id, gene.gene_id)
            if key not in split_rows and not allow_missing_split:
                raise gc.SplitError(f"{species_id}:{gene.gene_id} has no split entry")
            srow = split_rows.get(key, {"split": None, "orthogroup_id": None, "strict_holdout": ""})
            con.execute("INSERT INTO gene_key_map VALUES (?,?,?,?,?,?,?)", [species_id, gene.gene_id, gene.gene_id_original, gene.name_original,
                                                                          gene.chrom, gene.start0, gene.end0])
            train_weight, qc_list = 1.0, []
            if qc_flags and key in qc_flags:
                train_weight, keep_tx, qc_list = loss_mask_decision(qc_flags[key], gene.transcripts.keys())
                if train_weight > 0 and len(keep_tx) < len(gene.transcripts):
                    gene = gc.Gene(gene.gene_id, gene.chrom, gene.strand, gene.start0, gene.end0, {t: gene.transcripts[t] for t in keep_tx},
                                   gene.gene_id_original, gene.name_original)
            if gene.chrom not in genome:
                rejected.append({"gene_id": gene.gene_id, "reason": f"chromosome {gene.chrom} missing from FASTA"})
                continue
            try:
                gc.check_caps(gc.gene_to_gsf(gene, gene.start0))
            except gc.CapError as e:
                rejected.append({"gene_id": gene.gene_id, "reason": str(e)})
                continue
            by_chrom.setdefault(gene.chrom, []).append(gene)
            gene_meta[gene.gene_id] = {"split": srow["split"], "strict": str(srow.get("strict_holdout", "")).lower() in ("1", "true", "yes"),
                                       "weight": train_weight, "qc": qc_list}
    inserted = rc_rows = 0
    con.sql("CREATE TABLE IF NOT EXISTS tile_blocks (species_id VARCHAR, chromosome VARCHAR, start0 INT, end0 INT, split VARCHAR)")
    block_rng = random.Random(f"{seed}:{species_id}:blocks")
    for chrom in sorted(by_chrom):
        genes = by_chrom[chrom]
        chrom_len = len(genome[chrom])
        forced = [(g.start0, g.end0) for g in genes if gene_meta[g.gene_id]["strict"]]
        blocks = gc.block_splits(chrom_len, block_rng, forced_test=forced)
        con.executemany("INSERT INTO tile_blocks VALUES (?,?,?,?,?)", [[species_id, chrom, a, b, sp] for a, b, sp in blocks])
        for tier in gc.WINDOW_TIERS:
            offset = rng.randrange(tier) if mode == "train" else 0
            for ws, we in gc.tile_windows(chrom_len, tier, offset):
                inside, partial = gc.genes_in_window(genes, ws, we)
                if not inside and rng.random() > gc.EMPTY_KEEP_PROB:
                    continue
                split = gc.tile_split(blocks, ws, we)
                # A29 leakage masking: a gene whose orthogroup split is more restrictive than the tile split is
                # N-masked in the sequence and left out of the label (train tile must never label a test/valid gene)
                leak = [g for g in inside if gene_meta[g.gene_id]["split"] and gc.SPLIT_RANK.get(gene_meta[g.gene_id]["split"], 0) > gc.SPLIT_RANK[split]]
                labelled = [g for g in inside if g not in leak]
                gsf = gc.window_to_gsf_v3(labelled, ws) if mode == "train" else None
                L = we - ws
                if gsf is not None:
                    try:
                        gc.check_caps_v3(gsf, window_len=L)
                    except gc.CapError as e:
                        rejected.append({"gene_id": f"{chrom}:{ws}-{we}", "reason": f"window {e}"})
                        continue
                seq = genome[chrom][ws:we]
                if len(seq) < L:
                    seq = seq + "N" * (L - len(seq))
                if leak:
                    seq = gc.leak_mask(seq, ws, leak)
                weight = 0.0 if any(gene_meta[g.gene_id]["weight"] == 0 for g in labelled) else 1.0
                if allow_missing_split and not inside:
                    split = None
                wid = f"{species_id}:{chrom}:{ws}-{we}"
                qc = ";".join(sorted({f for g in labelled for f in gene_meta[g.gene_id]["qc"]})) or None
                if partial:
                    qc = (qc + ";" if qc else "") + f"edge_partial={partial}"
                if leak:
                    qc = (qc + ";" if qc else "") + f"leak_masked={len(leak)}"
                inside = labelled
                common = [species_id, wid, None, split, any(gene_meta[g.gene_id]["strict"] for g in inside), False, gc.ORDERING_VERSION,
                          gc.BUILD_VERSION, fasta_sha, gff_sha, split_sha, gc.WINDOW_POLICY_V3,
                          gc.count_tokens_v3(gsf) if gsf else None, we > chrom_len, sum(len(g.transcripts) for g in inside), weight, qc, wid]
                con.execute(sql, [wid, ws, we, "+", chrom, seq, gsf, 0, 0, 0, 0] + common)
                con.executemany("INSERT INTO window_genes VALUES (?,?,?,?)", [[species_id, wid, g.gene_id, False] for g in inside]) if inside else None
                inserted += 1
                if gsf is not None and inside and (rc == "all" or (rc == "isoform-only" and any(len(g.transcripts) >= 2 for g in inside))):
                    rgsf = gc.reverse_complement_v3(gsf, L)
                    rcc = list(common); rcc[5] = True; rcc[12] = gc.count_tokens_v3(rgsf)
                    con.execute(sql, [wid + "-rc", ws, we, "-", chrom, revcomp(seq), rgsf, 0, 0, 0, 0] + rcc)
                    con.executemany("INSERT INTO window_genes VALUES (?,?,?,?)", [[species_id, wid, g.gene_id, True] for g in inside])
                    inserted += 1; rc_rows += 1
    con.sql("CREATE TABLE IF NOT EXISTS rejected_records (species_id VARCHAR, gene_id VARCHAR, reason VARCHAR)")
    if rejected:
        con.executemany("INSERT INTO rejected_records VALUES (?,?,?)", [[species_id, r["gene_id"], r["reason"]] for r in rejected])
    reasons: Dict[str, int] = {}
    for r in rejected:
        k = r["reason"].split(":")[0].split(" >")[0]
        reasons[k] = reasons.get(k, 0) + 1
    con.execute("INSERT INTO build_manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [species_id, os.path.abspath(fasta), fasta_sha, os.path.abspath(gff), gff_sha, split_sha, rc, inserted, rc_rows,
                 len(rejected), json.dumps(reasons), gc.BUILD_VERSION, gc.ORDERING_VERSION, gc.WINDOW_POLICY_V3, git_commit,
                 time.strftime("%Y-%m-%dT%H:%M:%S"), duckdb.__version__])
    con.execute("COMMIT")
    return {"species_id": species_id, "rows": inserted, "rc_rows": rc_rows, "rejected": rejected}
