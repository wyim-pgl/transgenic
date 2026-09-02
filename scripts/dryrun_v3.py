#!/usr/bin/env python3
"""One-batch dry run of the GSF v3 pipeline on a GPU (issue #49): synthetic tile DB -> tokenizer v3 round trip ->
isoformDataHyena(split, v3) -> 400M model with vocab 290 / decoder positions 8,192 / encoder input up to 129,024 ->
forward+backward in bf16 with gradient checkpointing; prints peak memory per window tier.
Run from the repo root inside an environment with torch/transformers/duckdb: python scripts/dryrun_v3.py --out /tmp/v3dry
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def synthetic_inputs(out, n_genes=40, chrom_len=140000, seed=7):
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(chrom_len))
    fasta = os.path.join(out, "g.fa"); open(fasta, "w").write(">c1\n" + seq + "\n")
    lines, split = [], ["species_id\tgene_id\torthogroup_id\tsplit\tstrict_holdout\tseed\tsource_version"]
    pos = 500
    for i in range(n_genes):
        gid = f"g{i:03d}"; strand = rng.choice("+-"); n_ex = rng.randint(1, 6)
        s = pos; feats = []
        for e in range(n_ex):
            L = rng.randint(90, 600); feats.append((s, s + L - 1)); s += L + rng.randint(80, 400)
        gs, ge = feats[0][0], feats[-1][1]
        lines.append(f"c1\tt\tgene\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid}")
        lines.append(f"c1\tt\tmRNA\t{gs}\t{ge}\t.\t{strand}\t.\tID={gid}.1;Parent={gid}")
        cum = 0
        order = feats if strand == "+" else feats[::-1]
        for (a, b) in order:
            ph = (3 - cum % 3) % 3; cum += b - a + 1
            lines.append(f"c1\tt\tCDS\t{a}\t{b}\t.\t{strand}\t{ph}\tID={gid}.c;Parent={gid}.1")
        pos = ge + rng.randint(500, 2500)
        split.append(f"Ath\t{gid}\tOG{i}\t{'train' if i % 5 else 'valid'}\tfalse\t123\tv1")
    gff = os.path.join(out, "g.gff3"); open(gff, "w").write("\n".join(lines) + "\n")
    sp = os.path.join(out, "split.tsv"); open(sp, "w").write("\n".join(split) + "\n")
    man = os.path.join(out, "species.tsv"); open(man, "w").write(f"species_id\tspecies\ttable_s1_version\tfasta\tfasta_md5\tgff\tgff_md5\tnote\nAth\tA\tT\t{fasta}\t\t{gff}\t\t\n")
    return man, sp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/tmp/v3dry"); ap.add_argument("--config", default="configs/b5_400m_win_v3.json")
    ap.add_argument("--skip-model", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transgenic.datasets.build_b5 import build_b5_database, validate_b5_database
    from transgenic.utils import gsf_contract as gc
    man, sp = synthetic_inputs(a.out)
    db = os.path.join(a.out, "v3.duckdb"); [os.remove(f) for f in (db, db + ".wal") if os.path.exists(f)]
    res = build_b5_database(db, man, sp, rc="all", verify_md5=False, window_policy="tile6144-v3", tier_up_prob=0.0)
    print("build:", json.dumps({"rows": res[0]["rows"], "rejected": len(res[0]["rejected"])}))
    rep = validate_b5_database(db); print("validate ok:", rep["ok"], rep["violations"][:3])
    # tokenizer v3 round trip
    from transgenic.model.tokenization_transgenic import GFFTokenizer
    tok = GFFTokenizer(vocab_version="v3"); print("vocab_size:", tok.vocab_size)
    import duckdb
    con = duckdb.connect(db, read_only=True)
    rows = con.sql("SELECT gff, fin - start FROM geneList WHERE gff IS NOT NULL ORDER BY fin - start").fetchall(); con.close()
    for gsf, L in rows[:3] + rows[-1:]:
        ids = tok(gsf)["input_ids"]; back = tok.decode(ids, skip_special_tokens=True)
        n_expected = gc.count_tokens_v3(gsf)
        ok = back.replace(' ', '') == gsf.replace(' ', '')
        print(f"window {L}: tokens={len(ids)} expected={n_expected} roundtrip={'ok' if ok else 'MISMATCH'}")
        if not ok:
            print("   gsf :", gsf[:160]); print("   back:", back[:160])
    if a.skip_model:
        return
    import torch
    from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
    from transgenic.training.b5_runtime import load_b5_config, model_kwargs
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    cfg = load_b5_config(a.config)
    ds = isoformDataHyena(db, mode="train", encoder_model=cfg["encoder_model"], global_attention=False, split="train", gff_vocab_version="v3")
    print("dataset rows:", len(ds))
    config = HyenaTransgenicConfig(**model_kwargs(cfg))
    model = transgenicForConditionalGeneration(config).cuda(); model.gradient_checkpointing_enable(); model.train()
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    for tier in gc.WINDOW_TIERS:
        idx = [i for i in range(len(ds)) if ds[i][0].shape[-1] == tier] if hasattr(ds[0][0], "shape") else []
        if not idx:
            continue
        batch = hyena_collate_fn([ds[idx[0]]])
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch[0].cuda(), attention_mask=batch[1].cuda(), labels=batch[2].cuda(), return_dict=True)
        out.loss.backward(); torch.cuda.synchronize()
        print(f"tier {tier}: loss={out.loss.item():.3f} peak_mem_GB={torch.cuda.max_memory_allocated()/1e9:.1f} sec={time.time()-t0:.1f} label_tokens={batch[2].shape[-1]}")
        model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
