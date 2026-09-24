#!/usr/bin/env python3
"""Rule 15 four-rank integration for the B6 trainer (plan B6_WP2_WP4_PLAN_v4 sections 9-10, CP4).

Launch with ``accelerate launch --num_processes=4`` (revision/scripts/b6/integration_4rank.sbatch). Checks, all
on the production code path (train/train_HyenaTransgenic_B6.run_b6, modeling_b6):
  1. compiled training on 4 ranks with suppress_errors off: any Dynamo fallback, any recompile of a seen shape
     after warm-up, or zero Inductor kernels raises;
  2. batches mix evidence rows, zero-track chain-less rows (a dropped tile) and RC twins whose DNA, label, tracks
     and chains are all transformed from the forward twin (asserted pair by pair before training);
  3. resume determinism: a run preempted at update ``--stop-at`` and resumed with --resume auto publishes a
     step checkpoint byte-identical to the uninterrupted run's;
  4. cached generation (encoder and chain memory once, KV cache, beams) equals per-step re-encoding (encoder,
     downsampling and chain gather rerun at every step, no cache), greedy and beam, on evidence, zero and RC rows.
Rank 0 writes one JSON report; exit 0 = PASS, 2 = FAIL.

``--data`` is required. ``--data synthetic`` uses revision/scripts/b6/wp4_synthetic.py twins (DNA, label, tracks and chains
transformed together). ``--data wp3-fixtures`` uses WP3's own fixture world through the real verify_inputs,
B6Dataset and b6_collate_fn_torch, asserting DNA, label (reverse_complement_v3 of the fixture's real v3 GSF),
tracks and chains of every forward/RC twin under one fixed draw; it needs the merged WP3 modules (mandatory after the b6-integration merge). ``--model tiny`` with
``torch.distributed.run`` on CPU is the local smoke form.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HF_HOME", "./HFmodels")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=("real", "tiny"), default="real")
    p.add_argument("--data", choices=("synthetic", "wp3-fixtures"), required=True,
                   help="synthetic: wp4_synthetic twins; wp3-fixtures: WP3's b6_fixtures world through the real "
                        "verify_inputs and B6Dataset (needs the merged WP3 modules)")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "b6_400m_win_v3.json"))
    p.add_argument("--tier", type=int, default=30720, help="tile length of the synthetic rows")
    p.add_argument("--rows", type=int, default=48, help="training rows (forward/RC pairs; split over ranks)")
    p.add_argument("--accumulation", type=int, default=8, help="recipe total, divided by the world size")
    p.add_argument("--n-pilot", type=int, default=5, help="scaled stand-in for the 4,142-update checkpoint")
    p.add_argument("--stop-at", type=int, default=3, help="preemption point of the resumed run")
    p.add_argument("--backend", default="inductor")
    p.add_argument("--mixed-precision", default="bf16", choices=("bf16", "no"))
    p.add_argument("--gen-tokens", type=int, default=12)
    p.add_argument("--nondeterministic", action="store_true",
                   help="skip torch.use_deterministic_algorithms (the resume byte check then measures GPU atomics too)")
    p.add_argument("--out", required=True, help="run root (straight/, resumed/, report.json)")
    return p.parse_args(argv)


def _trainer():
    spec = importlib.util.spec_from_file_location("train_b6", os.path.join(ROOT, "train", "train_HyenaTransgenic_B6.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_b6"] = mod
    spec.loader.exec_module(mod)
    return mod


class TinyHyena:
    """Stand-in encoder for the CPU smoke form only."""

    def __new__(cls, config):
        import torch

        class _T(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, config.d_model)

            def forward(self, input_ids):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids), hidden_states=None)
        return _T()


def model_factory(args, cfg):
    import torch
    from transgenic.model import modeling_HyenaTransgenic as b5
    from transgenic.model import modeling_b6 as b6
    from transgenic.training import b6_runtime as rt
    torch.manual_seed(rt.SEED)
    if args.model == "real":
        return b6.B6TransgenicForConditionalGeneration(b6.b6_config(**rt.b6_model_kwargs(cfg)))
    with patch.object(b5.AutoConfig, "from_pretrained", side_effect=lambda *a, **k: SimpleNamespace()), \
            patch.object(b5.AutoModel, "from_config", side_effect=lambda config, **k: TinyHyena(config)):
        return b6.B6TransgenicForConditionalGeneration(b6.b6_config(
            vocab_size=290, d_model=8, decoder_d_model=16, encoder_layers=2, decoder_layers=2, encoder_n_layer=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
            max_encoder_seqlen=args.tier, max_encoder_position_embeddings=args.tier,
            max_decoder_position_embeddings=64, attention_window=[4, 4], dropout=0.1, decoder_layerdrop=0.1,
            load_pretrained_encoder=False, b6_chain_d_model=16, b6_chain_heads=8, b6_chain_dropout=0.1))


def generation_parity(model, batch, tokens):
    """Cached HF generation vs per-step re-encoding with HF's own search loop (greedy and beam 2)."""
    import torch
    from transgenic.model import modeling_HyenaTransgenic as b5
    from transgenic.model import modeling_b6 as b6

    class ReEncode(b6.B6TransgenicForConditionalGeneration):
        def generate(self, inputs=None, **kw):
            return b5.transgenicForConditionalGeneration.generate(self, inputs, **kw)

        def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, reencode_ids=None,
                                          evidence_tracks=None, chain_idx=None, chain_feat=None, chain_mask=None,
                                          **kwargs):
            return dict(input_ids=reencode_ids, evidence_tracks=evidence_tracks, chain_idx=chain_idx,
                        chain_feat=chain_feat, chain_mask=chain_mask, decoder_input_ids=decoder_input_ids,
                        use_cache=False)

        def forward(self, input_ids=None, decoder_input_ids=None, evidence_tracks=None, chain_idx=None,
                    chain_feat=None, chain_mask=None, reencode_ids=None, **kwargs):
            return super().forward(input_ids, decoder_input_ids=decoder_input_ids, evidence_tracks=evidence_tracks,
                                   chain_idx=chain_idx, chain_feat=chain_feat, chain_mask=chain_mask, **kwargs)

    model = model.float().eval()
    ref = ReEncode.__new__(ReEncode)
    ref.__dict__ = model.__dict__                         # same modules and weights, reference methods
    chains = {k: batch[k] for k in ("chain_idx", "chain_feat", "chain_mask")}
    results = {}
    for beams in (1, 2):
        common = dict(max_new_tokens=tokens, eos_token_id=None, num_beams=beams, do_sample=False,
                      return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            cached = model.generate(batch["input_ids"], evidence_tracks=batch["evidence_tracks"], use_cache=True,
                                    **chains, **common)
            per_step = ref.generate(batch["input_ids"], reencode_ids=batch["input_ids"],
                                    evidence_tracks=batch["evidence_tracks"], use_cache=False, **chains, **common)
        max_diff = max(float((a - b).abs().max()) for a, b in zip(cached.scores, per_step.scores))
        results[f"beams{beams}"] = dict(sequences_equal=bool(torch.equal(cached.sequences, per_step.sequences)),
                                        max_score_diff=max_diff)
    return results


def _chain_set(pos, feat, mask, slotless=True):
    """Chain elements as a sorted multiset; the slot column is dropped because rule 6a keys slots by row."""
    import numpy as np
    keep = [c for c in range(feat.shape[1]) if not (slotless and c == 2)]
    return sorted((tuple(int(x) for x in pos[e]), tuple(int(feat[e, c]) for c in keep))
                  for e in np.flatnonzero(mask))


def wp3_fixture_data(workdir, seed, trainer, acc):
    """WP3's own fixture world (tests/b6_fixtures.make_world) verified by the real verify_inputs through the
    trainer's code path, forward/RC twins asserted, and WP3's B6Dataset + collate for training. Post-merge only:
    it imports b6_bindings, b6_context, b6_obs_cache, b6_aug, b6_tracks, b6_chains and b6_dataset."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from tests import b6_fixtures as fx
    from transgenic.datasets import b6_dataset as DS
    from transgenic.evidence import b6_aug as AG, b6_bindings as BD, b6_chains as CH, b6_context as CX
    from transgenic.evidence import b6_obs_cache as OC, b6_track_spec as TS, b6_tracks as TR
    from transgenic.training import b6_runtime as rt
    # The world is not byte-deterministic (DuckDB files, PROVENANCE text), so rank 0 builds ONE copy and every rank
    # verifies that copy; per-rank builds would give the ranks different identities.
    listing = os.path.join(workdir, "world_paths.json")
    if acc.is_main_process:
        os.makedirs(workdir, exist_ok=True)
        w = fx.make_world(workdir)
        with open(listing, "w") as fh:
            json.dump({"evidence": str(w.c0), "cache": str(w.cache_dir), "freeze": str(w.freeze_path),
                       "corpus": str(w.corpus), "references": {fx.SPECIES: str(w.fasta)}}, fh)
    acc.wait_for_everyone()
    with open(listing) as fh:
        paths = json.load(fh)
    cache = OC.ObsCache(paths["cache"])
    record = trainer.verify_training_inputs(paths, roles=fx.make_roles(), cache=cache, b6_bindings=BD,
                                            aug_seed=rt.AUG_SEED_BASE + seed)
    corpus = CX.B6Corpus(paths["corpus"], cache, expected_sha256=record["corpus_sha256"])
    train_rns, valid_rns, c3 = trainer.corpus_rows(paths["corpus"])
    twins = {}
    for rn, wid, tier, split, is_rc in c3:
        twins.setdefault(wid, {})[bool(is_rc)] = rn
    gc = fx.gsf_contract()                           # the gsf_contract the B6 corpus builder uses (b6_corpus_repair)
    checked = []
    for wid, pair in sorted(twins.items()):
        if set(pair) != {False, True}:
            continue
        fw = CX.corpus_context(corpus, pair[False], "train_track")
        rc = CX.corpus_context(corpus, pair[True], "train_track")
        owner = AG.Owner.from_pairs(fw.species, cache.owner_libraries(fw.population, fw.species))
        draw = AG.resolve_fixed(rt.AUG_SEED_BASE + seed, owner, 1.0)
        if rc.dna != fx._revcomp(fw.dna):
            raise AssertionError(f"{wid}: RC twin DNA is not the reverse complement")
        # Label RC (review 3 M8): a real, non-empty v3 GSF whose RC twin is reverse_complement_v3 of it.
        g_fw, g_rc = corpus.row(pair[False])[9], corpus.row(pair[True])[9]
        try:
            gc.check_caps_v3(g_fw, fw.L)
            gc.check_caps_v3(g_rc, rc.L)
        except Exception as exc:
            raise AssertionError(f"{wid}: twin label is not a v3 GSF ({type(exc).__name__}: {exc})") from exc
        if not gc.split_v3(g_fw):
            raise AssertionError(f"{wid}: forward twin label is empty; label RC would be vacuous")
        if g_rc != gc.reverse_complement_v3(g_fw, fw.L) or gc.reverse_complement_v3(g_rc, rc.L) != g_fw:
            raise AssertionError(f"{wid}: RC twin label != reverse_complement_v3(forward label)")
        t_fw, t_rc = TR.rasterise(cache, fw, draw), TR.rasterise(cache, rc, draw)
        if not np.array_equal(t_rc, TS.rc_tracks(t_fw)):
            raise AssertionError(f"{wid}: RC twin tracks != rc_tracks(forward)")
        c_fw, c_rc = CH.build_chains(cache, fw, draw), CH.build_chains(cache, rc, draw)
        mapped = TS.rc_chain(c_fw.chain_pos, c_fw.chain_feat, c_fw.chain_mask, fw.L)
        if _chain_set(*mapped) != _chain_set(c_rc.chain_pos, c_rc.chain_feat, c_rc.chain_mask):
            raise AssertionError(f"{wid}: RC twin chains != chain-rc-v1(forward) (slots aside)")
        checked.append(dict(wid=wid, fw=pair[False], rc=pair[True], tracks_nonzero=int(np.count_nonzero(t_fw)),
                            chain_elements=int(c_fw.chain_mask.sum()), label_genes=len(gc.split_v3(g_fw)),
                            label_fw=g_fw, label_rc=g_rc))
    if not checked:
        raise AssertionError("the WP3 fixture world has no forward/RC twin pair")
    # Real v3 GSF labels go through WP3's production GFF tokenizer (as synthetic mode does), not the fixture's
    # one-token-per-character test tokenizer, whose label lengths would overrun the tiny decoder's positions.
    tokenize = DS.gff_label_tokenizer("v3")
    common = dict(arm="B6", aug_seed=rt.AUG_SEED_BASE + seed, encode_dna=fx.char_encoder, tokenize_gsf=tokenize)
    train_ds = DS.B6Dataset(corpus, cache, mode="train_track", **common)
    eval_ds = DS.B6Dataset(corpus, cache, mode="valid_track", **common)
    for p in checked:                                # the label tokens the trainer is served for each RC twin
        want = tokenize(gc.reverse_complement_v3(p["label_fw"], int(fx.TILE)))
        if not np.array_equal(np.asarray(train_ds[(p["rc"], 0)]["labels"]), want):
            raise AssertionError(f"{p['wid']}: served RC labels != tokens of reverse_complement_v3(forward label)")
    sampler = trainer.WP3EpochKeys(DS.B6EpochSampler, train_rns, seed)
    def reverify():
        """What production main() does at every link: a fresh cache handle and the real verify_inputs."""
        return trainer.verify_training_inputs(paths, roles=fx.make_roles(), cache=OC.ObsCache(paths["cache"]),
                                              b6_bindings=BD, aug_seed=rt.AUG_SEED_BASE + seed)

    return dict(record=record, reverify=reverify, train_ds=train_ds, eval_ds=eval_ds, sampler=sampler,
                eval_keys=[(int(rn), 0) for rn in valid_rns], collate=DS.b6_collate_fn_torch, pairs=checked,
                record_sha256_fn=BD.canonical_record_sha256, tile=int(fx.TILE),
                labels_note="label RC asserted: fixture labels are real v3 GSF (tests/b6_fixtures.tile_labels); each "
                            "RC twin label == reverse_complement_v3(forward) both ways and its served tokens match",
                n_train_rows=len(train_rns))


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, gather_object
    import wp4_synthetic as syn
    from transgenic.training import b6_runtime as rt

    trainer = _trainer()
    acc = Accelerator(mixed_precision=args.mixed_precision,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    cfg = rt.load_b6_config(args.config)
    cfg["_sha256"] = rt.file_sha256(args.config)
    cfg["max_epochs"] = 2
    bindings = syn.binding_record()                  # synthetic mode; wp3-fixtures replaces it with the verified record
    record_fn = syn.record_sha256
    backend = None if args.backend == "none" else args.backend
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "world_size": acc.num_processes, "args": vars(args)}

    link_records = []

    def train(run, **kw):
        link_bindings = bindings
        if wp3 is not None:                          # every simulated link re-verifies its inputs (review 3 M8)
            link_bindings = wp3["reverify"]()
            link_records.append(link_bindings["record_sha256"])
        spec = trainer.B6TrainSpec(arm=rt.ARMS["B6"], cfg=cfg, run_dir=os.path.join(args.out, run),
                                   bindings=link_bindings, accumulation_steps=args.accumulation, max_epochs=2,
                                   n_pilot=args.n_pilot, save_every_n_steps=0, mixed_precision=args.mixed_precision,
                                   compile_backend=backend, compile_warmup_steps=2,
                                   deterministic=not args.nondeterministic, record_sha256_fn=record_fn, **kw)
        if wp3 is not None:
            return trainer.run_b6(spec, model_factory(args, cfg), train_ds, eval_ds, collate_fn=wp3["collate"],
                                  train_sampler=wp3["sampler"], eval_sampler=wp3["eval_keys"], accelerator=acc)
        return trainer.run_b6(spec, model_factory(args, cfg), train_ds, eval_ds, collate_fn=syn.synthetic_collate,
                              train_sampler=syn.sampler_for(train_ds), eval_sampler=syn.eval_keys(eval_ds),
                              accelerator=acc)

    failures = []
    if args.data == "wp3-fixtures":
        wp3 = wp3_fixture_data(os.path.join(args.out, "wp3_world"), rt.SEED, trainer, acc)
        report.update(wp3_pairs=wp3["pairs"], wp3_note=wp3["labels_note"])
        bindings, record_fn = wp3["record"], wp3["record_sha256_fn"]
        args.tier = wp3["tile"]                      # the tiny model is sized to the fixture tile
        train_ds, eval_ds = wp3["train_ds"], wp3["eval_ds"]
    else:
        wp3 = None
        train_ds = syn.SyntheticB6Dataset(args.rows, length=args.tier, seed=1)
        eval_ds = syn.SyntheticB6Dataset(8, length=args.tier, seed=2, first_rn=90000)
    # The stop point and the step checkpoint must be reachable within the 2 epochs (review 3 M8): accelerate deals
    # ceil(rows / world) micro-batches to each rank per epoch; per-rank accumulation = accumulation / world.
    n_rows = wp3["n_train_rows"] if wp3 is not None else len(train_ds)
    per_rank_acc = args.accumulation // acc.num_processes
    reachable = (2 * -(-n_rows // acc.num_processes)) // max(1, per_rank_acc)
    report["reachable_updates"] = reachable
    if not (args.accumulation % acc.num_processes == 0 and 1 <= args.stop_at < args.n_pilot <= reachable):
        print(f"REFUSED: need 1 <= stop_at ({args.stop_at}) < n_pilot ({args.n_pilot}) <= reachable updates "
              f"({reachable} = 2 epochs x ceil({n_rows} rows / {acc.num_processes} ranks) / per-rank accumulation "
              f"{per_rank_acc}) and accumulation divisible by the world size", file=sys.stderr)
        return 4
    # RC coverage (review 1 M8): every forward/RC twin must agree in DNA, label, tracks and chains before use.
    for ds in ((train_ds, eval_ds) if wp3 is None else ()):
        for pair in range(len(ds) // 2):
            try:
                syn.assert_rc_pair(ds, pair)
            except AssertionError as exc:
                failures.append(f"rc pair: {exc}")
    report["rc_pairs_checked"] = (len(train_ds) + len(eval_ds)) // 2 if wp3 is None else len(wp3["pairs"])
    try:
        # One Accelerator serves three runs; free_memory() drops the previous run's registered model, optimizer,
        # scheduler and loaders so save_state/load_state see exactly one of each, as in a production link.
        report["straight"] = train("straight")
        acc.free_memory()
        torch._dynamo.reset()
        report["preempted"] = train("resumed", stop_at_step=args.stop_at)
        acc.free_memory()
        torch._dynamo.reset()
        report["resumed"] = train("resumed", resume="auto")
        acc.free_memory()
        torch._dynamo.reset()
    except Exception as exc:                       # the compile policy and every trainer refusal land here
        import traceback
        traceback.print_exc()
        failures.append(f"training: {type(exc).__name__}: {exc}")
    if wp3 is not None:
        report["link_record_sha256"] = link_records
        if len(link_records) != 3 or len(set(link_records)) != 1:
            failures.append(f"per-link verification: {len(link_records)} records, {len(set(link_records))} distinct")
    if not failures and acc.is_main_process:
        paths = [os.path.join(rt.step_checkpoint_dir(os.path.join(args.out, r), args.n_pilot), "model.safetensors")
                 for r in ("straight", "resumed")]
        same = all(os.path.exists(p) for p in paths) and open(paths[0], "rb").read() == open(paths[1], "rb").read()
        report["resume_byte_identical"] = same
        if not same:
            failures.append("resume determinism: step checkpoints differ")
    # Generation parity on this rank's rows: a forward/RC twin pair and (synthetic mode) a dropped tile.
    if wp3 is not None:
        p0 = wp3["pairs"][0]
        batch = wp3["collate"]([train_ds[(p0["fw"], 0)], train_ds[(p0["rc"], 0)]])
        report.setdefault("generation_row_kinds", ["wp3_forward", "wp3_rc"])
    else:
        ds = syn.SyntheticB6Dataset(4, length=args.tier, seed=7 + acc.process_index)
        syn.assert_rc_pair(ds, 0)
        batch = syn.synthetic_collate([ds[i] for i in range(3)])
        report.setdefault("generation_row_kinds", [ds.raw(i)["kind"] for i in range(3)])
    rt.validate_b6_batch(batch)
    batch = {k: (v.to(acc.device) if hasattr(v, "to") else v) for k, v in batch.items()}
    model = model_factory(args, cfg).to(acc.device)
    with torch.no_grad():
        model.get_encoder().evidence_proj.weight.normal_(0, 0.02)
        for layer in model.transgenic.decoder.layers:
            layer.chain_attn.alpha.fill_(0.5)
    parity = generation_parity(model, batch, args.gen_tokens)
    for name, res in parity.items():
        if not res["sequences_equal"]:
            failures.append(f"rank {acc.process_index} generation {name}: cached != per-step re-encoding")
    all_parity = gather_object([dict(rank=acc.process_index, parity=parity)])
    all_failures = gather_object(failures)
    status = "PASS" if not all_failures else "FAIL"
    if acc.is_main_process:
        report.update(generation=all_parity, failures=all_failures, status=status,
                      finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(os.path.join(args.out, "report.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(json.dumps({"status": status, "failures": all_failures}), flush=True)
    acc.wait_for_everyone()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
