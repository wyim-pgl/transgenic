#!/usr/bin/env python3
"""Figure 3 regeneration: de novo inference over the held-out test split.

Replicates the training-time split (train_HyenaTransgenic.py: manual_seed(123),
random_split 75/10/15 over isoformDataHyena(db, exclude_prefix='Zm')), then runs
de novo generation with the published 400M checkpoint on:
  - the 60,550 held-out test rows (9 training species, from fig3_test_rows.tsv)
  - all 49,431 Z. mays rows (entire genome withheld)

Predictions are written per species as GFF3 (RC rows coordinate-flipped) for
per-species GFFCompare evaluation (Fig. 3A) and per-gene metrics (Fig. 3C/D).

Protocol matches the benchmark: beam search, num_beams=2, do_sample=True,
max_length=2048 (run_genome_annotation.py).
"""
import os, sys, csv, time, json
sys.path.insert(0, "/home/framazan/transgenic/src")

import torch
import duckdb
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from transgenic.utils.gsf import gffString2GFF3

DB = "/home/framazan/data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
TSV = "/home/framazan/fig3_test_rows.tsv"
OUTDIR = "/home/framazan/fig3_test_preds"
DONE = "/home/framazan/fig3_test_preds.done"
MODEL = "jlomas/HyenaTransgenic-768L12A6-400M"
ENC = "LongSafari/hyenadna-large-1m-seqlen-hf"
BATCH = 4

SPECIES = [
    ("AT", "A_thaliana"), ("Glyma", "G_max"), ("Pp", "P_patens"),
    ("Potri", "P_trichocarpa"), ("Sobic", "S_bicolor"), ("Bradi", "B_distachyon"),
    ("Seita", "S_italica"), ("Vitvi", "V_vinifera"), ("LOC_Os", "O_sativa"),
    ("Zm", "Z_mays"),
]

# 361 gene models in the training DB carry no species prefix: MSU v7 places 185
# O. sativa genes on the ChrSy/ChrUn pseudo-molecules, and RefGen_V4 retains 176
# legacy GRMZM maize IDs. Both sets are present in the reference annotations used
# for evaluation (2,306 ChrSy/ChrUn lines; 865 GRMZM lines), so they belong here.
EXTRA_PREFIX = [("ChrSy", "O_sativa"), ("ChrUn", "O_sativa"), ("GRMZM", "Z_mays")]
UNASSIGNED = "_unassigned"

def species_of(gm):
    for pref, sp in SPECIES + EXTRA_PREFIX:
        if gm.startswith(pref):
            return sp
    return None

os.makedirs(OUTDIR, exist_ok=True)
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

print("loading model...", flush=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval().to(device)
gffTok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
dnaTok = AutoTokenizer.from_pretrained(ENC, trust_remote_code=True)

con = duckdb.connect(DB, config={"access_mode": "READ_ONLY"})

# ---- collect work items -----------------------------------------------------
rns = []
with open(TSV) as fh:
    r = csv.reader(fh, delimiter="\t")
    next(r)
    for row in r:
        rns.append(int(row[0]))
zm = con.sql("SELECT rn FROM geneList WHERE geneModel LIKE 'Zm%' ORDER BY rn").fetchall()
rns += [r[0] for r in zm]
print(f"total rows: {len(rns)}", flush=True)

done = set()
if os.path.exists(DONE):
    done = {int(x) for x in open(DONE) if x.strip()}
    print(f"resume: {len(done)} rows already done", flush=True)
rns = [r for r in rns if r not in done]

# fetch metadata in chunks
rows = {}
B = 10000
for i in range(0, len(rns), B):
    chunk = rns[i:i+B]
    q = ("SELECT rn, geneModel, start, fin, strand, chromosome, sequence "
         f"FROM geneList WHERE rn IN ({','.join(map(str, chunk))})")
    for rn, gm, st, fin, strand, ch, seq in con.sql(q).fetchall():
        rows[rn] = (gm, st, fin, strand, ch, seq)

# group by (seqlen) for padding-free batches
bylen = defaultdict(list)
for rn in rns:
    if rn in rows:
        bylen[len(rows[rn][5])].append(rn)

batches = []
for L, group in sorted(bylen.items(), key=lambda kv: -len(kv[1])):
    for i in range(0, len(group), BATCH):
        batches.append(group[i:i+BATCH])
print(f"{len(batches)} batches", flush=True)

out_fh = {sp: open(os.path.join(OUTDIR, f"{sp}_test400M.gff3"), "a") for _, sp in SPECIES}
# Catch-all so that an unrecognised prefix is reported and skipped rather than
# aborting the run, as it did after V. vinifera on 2026-07-27.
out_fh[UNASSIGNED] = open(os.path.join(OUTDIR, f"{UNASSIGNED}_test400M.gff3"), "a")
done_fh = open(DONE, "a")

def flip_rc(lines, st, fin):
    # gffString2GFF3 has already shifted features to absolute genome coordinates
    # (region_start + 1 + local offset), so the mirror must be taken over the
    # absolute window [st+1, fin]: mirroring with `fin - x + 1` alone treats the
    # coordinates as window-local and drops every -rc prediction near the
    # chromosome start, exactly region_start short (see
    # revision/results/fig3_gap_diagnosis/gap_diagnosis_summary.json).
    out = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        f = ln.split("\t")
        s, e = int(f[3]), int(f[4])
        ns, ne = fin - e + 1 + st, fin - s + 1 + st
        f[3], f[4] = str(ns), str(ne)
        f[6] = "+" if f[6] == "-" else "-" if f[6] == "+" else f[6]
        out.append("\t".join(f))
    return out

def write_pred(rn, out_row):
    pred = gffTok.batch_decode(out_row.detach().cpu().numpy(), skip_special_tokens=True)[0]
    gm, st, fin, strand, ch, seq = rows[rn]
    lines = gffString2GFF3(pred, ch, st, f"GM={gm}")
    if gm.endswith("-rc"):
        lines = flip_rc(lines, st, fin)
    sp = species_of(gm)
    if sp is None:
        print(f"WARNING: no species prefix for gene model {gm!r} (rn {rn}); "
              f"writing to {UNASSIGNED}", flush=True)
        sp = UNASSIGNED
    if lines != [""]:
        out_fh[sp].write("\n".join(lines) + "\n")
        out_fh[sp].flush()
    done_fh.write(f"{rn}\n")

t0 = time.time()
n_done = 0
for bi, batch in enumerate(batches):
    seqs = [rows[rn][5] for rn in batch]
    toks = dnaTok(seqs, return_tensors="pt", padding=False)
    ii = toks["input_ids"][:, :-1].to(device) if toks["input_ids"].shape[1] > 1 else toks["input_ids"].to(device)
    am = (ii != dnaTok.pad_token_id)
    try:
        with torch.inference_mode():
            outs = model.generate(inputs=ii, attention_mask=am, num_return_sequences=1,
                                  max_length=2048, num_beams=2, do_sample=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        for rn in batch:  # fall back to single-row generation
            seq = rows[rn][5]
            t1 = dnaTok(seq, return_tensors="pt")
            i1 = t1["input_ids"][:, :-1].to(device)
            a1 = (i1 != dnaTok.pad_token_id)
            with torch.inference_mode():
                o1 = model.generate(inputs=i1, attention_mask=a1, num_return_sequences=1,
                                    max_length=2048, num_beams=2, do_sample=True)
            write_pred(rn, o1)
        n_done += len(batch)
        done_fh.flush()
        continue
    for j, rn in enumerate(batch):
        write_pred(rn, outs[j:j+1])
    done_fh.flush()
    n_done += len(batch)
    if bi % 50 == 0:
        el = time.time() - t0
        rate = n_done / el if el else 0
        eta = (len(rns) - n_done) / rate / 3600 if rate else 0
        print(f"[{bi}/{len(batches)}] {n_done}/{len(rns)} rows, {rate:.2f} rows/s, ETA {eta:.1f} h", flush=True)

print("ALL DONE", flush=True)
