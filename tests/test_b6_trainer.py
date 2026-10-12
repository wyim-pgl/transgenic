"""B6 trainer acceptance (plan v4.1 section 9, "Trainer acceptance"), CPU, tiny model, synthetic plan-7.7 data.

N_pilot is scaled down (6 instead of 4,099) and the step checkpoint published at it stands for step_04099;
the resume at step 5 stands for the resume at 4,000. Everything else is the production code path.
"""
import copy
import importlib.util
import unittest.mock
import json
import os
import sys
import tempfile
import unittest

import torch

from tests.test_b6_model import B6Base, tiny_b6
from transgenic.model import modeling_b6 as b6
from transgenic.training import b6_runtime as rt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "revision", "scripts", "b6"))
import wp4_synthetic as syn  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_b6", os.path.join(ROOT, "train", "train_HyenaTransgenic_B6.py"))
trainer = importlib.util.module_from_spec(_spec)
sys.modules["train_b6"] = trainer
_spec.loader.exec_module(trainer)

CFG = {"lr": 1e-3, "weight_decay": 0.02, "max_epochs": 3, "effective_batch_size": 2, "lr_warmup_fraction": 0.05,
       "gate_status": rt.GATE_STATUS, "_sha256": "c" * 64}


def _wp3_fixtures(test):
    """Writer B's tests/b6_fixtures.py (byte-identical copy). Since sha 1a7b6b07 it imports WP3's b6_bindings at
    module level, which lives only in the WP3 worktree until the b6-integration merge."""
    try:
        from tests import b6_fixtures as fx
    except ImportError as exc:
        test.skipTest(f"needs merged b6_bindings (mandatory after the b6-integration merge): {exc}")
    return fx


def _epoch(sampler, epoch):
    sampler.set_epoch(epoch)
    return sampler


class TrainerBase(B6Base):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = []

    def build(self):
        torch.manual_seed(rt.SEED)
        return b6.B6TransgenicForConditionalGeneration(tiny_b6(dropout=0.1, decoder_layerdrop=0.1,
                                                               b6_chain_dropout=0.1))

    def spec(self, arm="B6", run="run", **kw):
        base = dict(arm=rt.ARMS[arm], cfg=dict(CFG), run_dir=os.path.join(self.tmp.name, run),
                    bindings=syn.binding_record(), accumulation_steps=2, max_epochs=2, patience=3, n_pilot=6,
                    save_every_n_steps=0, mixed_precision="no", compile_backend=None, log=self.logs.append)
        base.update(kw)
        return trainer.B6TrainSpec(**base)

    def train_arm(self, spec, model=None, n_rows=8, train_ds=None):
        from accelerate import Accelerator
        train_ds = train_ds or syn.SyntheticB6Dataset(n_rows, seed=1)
        eval_ds = syn.SyntheticB6Dataset(4, seed=2, first_rn=5000)
        return trainer.run_b6(spec, model or self.build(), train_ds, eval_ds, collate_fn=syn.synthetic_collate,
                              train_sampler=syn.sampler_for(train_ds), eval_sampler=syn.eval_keys(eval_ds),
                              accelerator=Accelerator(cpu=True, mixed_precision="no"))


class ArmTests(TrainerBase):
    def test_one_run_per_arm(self):
        for arm in sorted(rt.ARMS):
            with self.subTest(arm=arm):
                model = self.build()
                seen = []

                def hook(module, args, kwargs, seen=seen):
                    if module.training:
                        seen.append({k: kwargs[k].detach().clone() for k in ("evidence_tracks", "chain_mask")})
                handle = model.register_forward_pre_hook(hook, with_kwargs=True)
                result = self.train_arm(self.spec(arm, run=f"arm_{arm}"), model)
                handle.remove()
                run_dir = os.path.join(self.tmp.name, f"arm_{arm}")
                self.assertTrue(result["done"])
                self.assertTrue(os.path.exists(os.path.join(run_dir, "TRAINING_DONE")))
                step_dir = rt.step_checkpoint_dir(run_dir, 6)
                for name in ("model.safetensors", "meta.json", rt.PRUNE_EXEMPT_MARKER):
                    self.assertTrue(os.path.exists(os.path.join(step_dir, name)), (arm, name))
                if rt.ARMS[arm].budget == "n_pilot":
                    self.assertEqual((result["global_step"], result["reason"]), (6, "n_pilot budget"))
                else:
                    self.assertEqual(result["global_step"], 8)            # 2 epochs x 4 updates, stop rule
                    self.assertEqual(len(result["history"]), 2)
                    self.assertIn("evidence", result["history"][0])
                tracks = torch.cat([s["evidence_tracks"] for s in seen])
                chains = torch.cat([s["chain_mask"] for s in seen])
                if arm == "C2":
                    self.assertEqual(torch.count_nonzero(tracks).item(), 0)
                    self.assertFalse(chains.any())
                elif arm == "C7":
                    self.assertGreater(torch.count_nonzero(tracks).item(), 0)
                    self.assertFalse(chains.any())
                elif arm == "C6":
                    self.assertEqual(torch.count_nonzero(tracks[..., 25:31]).item(), 0)
                    self.assertGreater(torch.count_nonzero(tracks[..., :25]).item(), 0)
                    self.assertTrue(chains.any())
                elif arm == "C4":
                    values = set(torch.unique(tracks[..., list(rt.COUNT_CHANNELS)].float()).tolist())
                    self.assertLessEqual(values, {0.0, 1.0})
                else:
                    self.assertTrue(chains.any())
                    self.assertGreater(len(set(torch.unique(tracks.float()).tolist())), 2)
                with open(os.path.join(run_dir, "run_config_b6.json")) as fh:
                    ident = json.load(fh)
                self.assertEqual(ident["arm"], arm)
                self.assertEqual((ident["seed"], ident["aug_seed"]), (123, 20261037))
                text = open(os.path.join(run_dir, "runtime_identity.txt")).read()
                self.assertIn("alpha.layer00=0.0", text)
                self.assertIn(f'link.evidence_channel="{rt.ARMS[arm].evidence_channel}"', text)

    def test_arms_share_initialisation_and_data_order(self):
        results = {}
        for arm in ("B6", "C2"):
            model = self.build()
            first = {k: v.clone() for k, v in model.state_dict().items()}
            results[arm] = (first, self.train_arm(self.spec(arm, run=f"order_{arm}", n_pilot=100), model))
        for key in results["B6"][0]:
            self.assertTrue(torch.equal(results["B6"][0][key], results["C2"][0][key]), key)
        orders = [[h["data_order_sha256"] for h in r[1]["history"]] for r in results.values()]
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0][0], orders[0][1])                 # a new permutation each epoch


class ResumeTests(TrainerBase):
    def test_step_checkpoint_byte_stable_across_resume(self):
        straight = self.spec("B6", run="straight")
        self.train_arm(straight)
        ref = open(os.path.join(rt.step_checkpoint_dir(straight.run_dir, 6), "model.safetensors"), "rb").read()
        ref_epoch = open(os.path.join(straight.run_dir, "epoch_02", "model.safetensors"), "rb").read()
        for stop in (1, 3, 5):                      # inside epoch 1, at its last update, inside epoch 2
            with self.subTest(stop=stop):
                run = f"resumed{stop}"
                first = self.train_arm(self.spec("B6", run=run, stop_at_step=stop))
                self.assertEqual((first["preempted"], first["global_step"]), (True, stop))
                resumed = self.spec("B6", run=run, resume="auto")
                self.assertFalse(os.path.exists(rt.step_checkpoint_dir(resumed.run_dir, 6)))
                self.assertTrue(self.train_arm(resumed)["done"])
                got = open(os.path.join(rt.step_checkpoint_dir(resumed.run_dir, 6), "model.safetensors"), "rb").read()
                self.assertEqual(got, ref)
                self.assertEqual(open(os.path.join(resumed.run_dir, "epoch_02", "model.safetensors"), "rb").read(),
                                 ref_epoch)
                text = open(os.path.join(resumed.run_dir, "runtime_identity.txt")).read()
                self.assertEqual(text.count("# link"), 2)

    def test_resume_skips_in_the_sampler_without_loading_skipped_rows(self):
        class Counting(syn.SyntheticB6Dataset):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.seen = []

            def __getitem__(self, i):
                self.seen.append(i)
                return super().__getitem__(i)

        self.train_arm(self.spec("C4", run="skip", stop_at_step=5))       # epoch 2, after 2 of its 8 rows
        ds = Counting(8, seed=1)
        self.train_arm(self.spec("C4", run="skip", resume="auto"), train_ds=ds)
        order = list(_epoch(syn.sampler_for(ds), 1))          # epoch 2's order (0-based epoch 1)
        self.assertEqual(ds.seen[:2], order[2:4])                 # update 6 = rows 3-4 of the epoch
        self.assertFalse(set(order[:2]) & set(ds.seen))           # the two skipped rows were never loaded
        self.assertLessEqual(len(ds.seen), 3)                     # plus at most accelerate's one-batch lookahead

    def test_weight_file_is_deterministic_and_round_trips(self):
        model = self.build()
        with torch.no_grad():
            for layer in model.transgenic.decoder.layers:
                layer.chain_attn.alpha.fill_(0.25)
        paths = [os.path.join(self.tmp.name, f"w{i}.safetensors") for i in range(2)]
        aliases = [rt.save_weights_deterministic(model, p) for p in paths]
        self.assertEqual(open(paths[0], "rb").read(), open(paths[1], "rb").read())
        self.assertEqual(aliases[0], {"transgenic.decoder.embed_tokens.weight": "lm_head.weight",
                                      "transgenic.decoder_embed_tokens.weight": "lm_head.weight"})
        torch.manual_seed(99)
        other = b6.B6TransgenicForConditionalGeneration(tiny_b6())
        rt.load_weights(other, paths[0])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
        self.assertIs(other.lm_head.weight, other.transgenic.decoder_embed_tokens.weight)

    def test_resume_refused_when_a_binding_changes(self):
        self.train_arm(self.spec("B6", run="r", stop_at_step=2))
        changed = syn.binding_record(corpus_sha256="f" * 64)
        with self.assertRaisesRegex(RuntimeError, "binding.corpus_sha256 differs"):
            self.train_arm(self.spec("B6", run="r", resume="auto", bindings=changed))
        with self.assertRaisesRegex(RuntimeError, "arm differs"):
            self.train_arm(self.spec("C4", run="r", resume="auto"))

    def test_resume_accepts_a_record_with_tuples(self):
        record = syn.binding_record(coverage=[("Athaliana", "est"), ("Athaliana", "protein")])
        self.train_arm(self.spec("B6", run="tup", stop_at_step=2, bindings=record))
        self.assertTrue(self.train_arm(self.spec("B6", run="tup", resume="auto", bindings=record))["done"])

    def test_resume_refused_for_every_required_key(self):
        self.train_arm(self.spec("B6", run="k", stop_at_step=2))
        with open(os.path.join(self.tmp.name, "k", "run_config_b6.json")) as fh:
            first = json.load(fh)
        self.assertEqual(set(first), set(rt.IDENTITY_REQUIRED_KEYS))
        rt.check_run_config_b6(first, first)
        for key in rt.IDENTITY_REQUIRED_KEYS:
            with self.subTest(key=key):
                dropped = {k: v for k, v in first.items() if k != key}
                with self.assertRaisesRegex(RuntimeError, "missing from the run's first identity"):
                    rt.check_run_config_b6(dropped, first)
                with self.assertRaisesRegex(RuntimeError, "missing from the current identity"):
                    rt.check_run_config_b6(first, dropped)
                other = copy.deepcopy(first)
                other[key] = ["changed"]
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    rt.check_run_config_b6(first, other)

    def test_binding_family_and_evlabel_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "eval_only"):
            rt.identity_from_bindings(syn.binding_record(family="eval_only"))
        with self.assertRaisesRegex(ValueError, "evlabel"):
            rt.identity_from_bindings(syn.binding_record(evlabel="present"))
        record = syn.binding_record()
        record.pop("interval_manifest_sha256")
        with self.assertRaisesRegex(ValueError, "interval_manifest_sha256"):
            rt.identity_from_bindings(record)
        with self.assertRaisesRegex(ValueError, "aug_seed"):
            rt.identity_from_bindings(syn.binding_record(aug_seed=20260915))
        for key in ("provenance", "draws", "zero_lists", "record_sha256", "revalidated"):
            record = syn.binding_record()
            record.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                rt.identity_from_bindings(record)
        with self.assertRaisesRegex(ValueError, "provenance"):
            rt.identity_from_bindings(syn.binding_record(provenance={"freeze": "short"}))
        ident = rt.identity_from_bindings(syn.binding_record())
        self.assertEqual(ident["draw_spec"]["aug_spec_id"], "b6-aug-uniform-blake2b-v2")
        self.assertEqual(ident["binding.reference_sha256"], None)
        record = syn.binding_record()
        self.assertEqual(rt.identity_from_bindings(record, syn.record_sha256)["binding.record_sha256"],
                         record["record_sha256"])
        tampered = dict(record, corpus_sha256="0" * 64)                   # edited after sealing
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            rt.identity_from_bindings(tampered, syn.record_sha256)

    def test_b5_data_order_is_accelerates_seedable_sampler(self):
        from accelerate.data_loader import SeedableRandomSampler
        rns = list(range(1000, 1000 + 3 * 97, 3))
        sampler = SeedableRandomSampler(range(len(rns)), data_seed=123)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            expected = [rns[i] for i in sampler]
            self.assertEqual(rt.b5_data_order(rns, 123, epoch), expected)
        self.assertNotEqual(rt.b5_data_order(rns, 123, 0), rt.b5_data_order(rns, 123, 1))


class ReviewRound1Tests(TrainerBase):
    """Codex WP4 review 1: H1 corpus schema, H2 foreign checkpoints, M5 per-link compile check, M6 identity,
    M7 canonical order and draw logs, M8 paired RC rows."""

    def test_corpus_rows_on_the_pinned_schema_and_the_wp3_fixture_corpus(self):
        import duckdb
        import numpy as np
        from transgenic.datasets import build_b5
        path = os.path.join(self.tmp.name, "pinned.db")
        con = duckdb.connect(path)
        build_b5.ensure_schema(con)
        rows = [(1, "w1", 0, 30720, "train", False, "g", 1.0), (2, "w1", 0, 30720, "train", True, "g", 1.0),
                (3, "w2", 0, 61440, "valid", False, "g", 1.0), (4, "w3", 0, 30720, "train", False, "g", 0.0),
                (5, "w4", 0, 30720, "train", False, None, 1.0), (6, "w5", 0, 30720, "test", False, "g", 1.0)]
        for rn, gid, st, fin, split, rc, gff, w in rows:
            con.execute("INSERT INTO geneList (rn, gene_id, start, fin, split, is_rc, gff, train_weight) "
                        "VALUES (?,?,?,?,?,?,?,?)", [rn, gid, st, fin, split, rc, gff, w])
        con.close()
        train, valid, c3 = trainer.corpus_rows(path)
        self.assertEqual((train, valid), ([1, 2], [3]))
        self.assertEqual(c3[:2], [(1, "w1", 30720, "train", False), (2, "w1", 30720, "train", True)])
        self.assertNotIn(6, [r[0] for r in c3])
        broken = os.path.join(self.tmp.name, "broken.db")
        con = duckdb.connect(broken)
        con.execute("CREATE TABLE geneList (rn INT, wid VARCHAR, start INT, fin INT, split VARCHAR, is_rc BOOLEAN, gff VARCHAR)")
        con.close()
        with self.assertRaisesRegex(ValueError, "gene_id"):
            trainer.corpus_rows(broken)

    def test_corpus_rows_on_the_wp3_fixture_corpus(self):
        fx = _wp3_fixtures(self)
        _, genome = fx.write_genome(self.tmp.name)
        corpus = fx.write_corpus(self.tmp.name, genome)
        train, valid, c3 = trainer.corpus_rows(corpus)
        by_wid = {}
        for rn, wid, tier, split, is_rc in c3:
            by_wid.setdefault(wid, []).append(bool(is_rc))
        self.assertTrue(train)
        self.assertTrue(any(sorted(v) == [False, True] for v in by_wid.values()))   # twins share gene_id

    def _checkpoint_run(self, arm, run, **kw):
        self.train_arm(self.spec(arm, run=run, stop_at_step=2, **kw))
        return os.path.join(self.tmp.name, run, "latest_state")

    def test_resume_refuses_a_foreign_arm_or_seed_checkpoint_before_loading(self):
        import shutil
        mine = self._checkpoint_run("B6", "own")
        foreign = self._checkpoint_run("C2", "foreign")
        shutil.rmtree(mine)
        shutil.copytree(foreign, mine)                        # a C2 checkpoint inside the B6 run
        loads = []
        from accelerate import Accelerator
        with unittest.mock.patch.object(Accelerator, "load_state", side_effect=lambda *a, **k: loads.append(a)):
            with self.assertRaisesRegex(RuntimeError, "arm differs"):
                self.train_arm(self.spec("B6", run="own", resume="auto"))
        self.assertEqual(loads, [])                           # refused before any state was loaded
        own = self._checkpoint_run("B6", "own2")
        ident = json.load(open(os.path.join(own, rt.CHECKPOINT_IDENTITY)))
        ident["seed"] = 999
        json.dump(ident, open(os.path.join(own, rt.CHECKPOINT_IDENTITY), "w"))
        with self.assertRaisesRegex(RuntimeError, "seed differs"):
            self.train_arm(self.spec("B6", run="own2", resume="auto"))
        os.remove(os.path.join(own, rt.CHECKPOINT_IDENTITY))
        with self.assertRaisesRegex(RuntimeError, "identity is unknown"):
            self.train_arm(self.spec("B6", run="own2", resume="auto"))

    def test_resume_refuses_a_different_initialisation(self):
        self._checkpoint_run("B6", "init")
        torch.manual_seed(7)                                  # same shapes, different weights
        other = b6.B6TransgenicForConditionalGeneration(tiny_b6(dropout=0.1, decoder_layerdrop=0.1,
                                                                b6_chain_dropout=0.1))
        with self.assertRaisesRegex(RuntimeError, "init_weights_sha256 differs"):
            self.train_arm(self.spec("B6", run="init", resume="auto"), model=other)
        self.assertEqual(rt.state_digest(self.build()), rt.state_digest(self.build()))

    def test_every_link_verifies_compiled_execution(self):
        import torch._dynamo as dynamo
        from unittest.mock import patch
        dynamo.reset()
        try:
            self.train_arm(self.spec("B6", run="link", compile_backend="aot_eager", compile_warmup_steps=2,
                                     stop_at_step=3))
            dynamo.reset()
            with patch.object(rt, "compiled_graph_count", lambda: 7):   # this link compiles nothing
                # Checkpoint replay now detects the missing graphs before the link warm-up check.
                with self.assertRaisesRegex(RuntimeError, "compiler replay graph delta"):
                    self.train_arm(self.spec("B6", run="link", compile_backend="aot_eager", compile_warmup_steps=2,
                                             resume="auto"))
            dynamo.reset()
            aot = lambda m, backend="inductor": torch.compile(m, backend="aot_eager")  # noqa: E731
            with patch.object(rt, "compile_b6", aot), patch.object(rt, "inductor_activity", lambda: 5):
                with self.assertRaisesRegex(RuntimeError, "no Inductor kernel"):      # link shorter than warm-up
                    self.train_arm(self.spec("C4", run="ind", compile_backend="inductor", compile_warmup_steps=100))
        finally:
            dynamo.reset()

    def test_canonical_data_order_and_draw_logs_agree_across_arms_and_resume(self):
        self.train_arm(self.spec("B6", run="b6", n_pilot=100))
        self.train_arm(self.spec("C2", run="c2", n_pilot=100))
        self.train_arm(self.spec("B6", run="res", n_pilot=100, stop_at_step=5))
        self.train_arm(self.spec("B6", run="res", n_pilot=100, resume="auto"))

        def order(run):
            out = {}
            for line in open(os.path.join(self.tmp.name, run, "data_order.jsonl")):
                e = json.loads(line)
                out.setdefault(e["epoch"], set()).add(e["first1000_sha256"])
            return out

        def draws(run):
            d = os.path.join(self.tmp.name, run, "draw_log")
            return sorted(set(l for f in os.listdir(d) for l in open(os.path.join(d, f))))

        self.assertEqual(order("b6"), order("c2"))
        self.assertEqual(order("b6"), order("res"))
        self.assertTrue(all(len(v) == 1 for v in order("res").values()))
        ds = syn.SyntheticB6Dataset(8, seed=1)
        expected = rt.data_order_hash(syn.sampler_for(ds).epoch_rows(0))
        self.assertEqual(order("b6")[1], {expected})
        self.assertEqual(draws("b6"), draws("c2"))
        self.assertEqual(draws("b6"), draws("res"))
        self.assertEqual(len(draws("b6")), 16)                 # 8 rows x 2 epochs, each (row, epoch) once

    def test_rc_rows_are_internally_consistent_twins(self):
        ds = syn.SyntheticB6Dataset(12, seed=3)
        for pair in range(6):
            syn.assert_rc_pair(ds, pair)
        fw, rc = ds[(0, 0)], ds[(1, 0)]
        self.assertEqual(rc["ids"].tolist(), [{7: 10, 10: 7, 8: 9, 9: 8, 11: 11}[x] for x in fw["ids"].tolist()][::-1])
        self.assertFalse(torch.equal(fw["labels"], rc["labels"]))
        bad = syn.SyntheticB6Dataset(4, seed=3)
        good_raw = bad.raw
        bad.raw = lambda i: dict(good_raw(i), seq=good_raw(i)["seq"][::-1]) if i == 1 else good_raw(i)
        with self.assertRaisesRegex(AssertionError, "reverse complement"):
            syn.assert_rc_pair(bad, 0)


class ReviewRound2Tests(TrainerBase):
    """Codex WP4 review 2: the trainer's WP3 verification path against the real b6_bindings.verify_inputs."""

    def test_verify_training_inputs_runs_the_real_wp3_verifier_with_the_freeze_path(self):
        fx = _wp3_fixtures(self)
        try:
            from transgenic.evidence import b6_bindings as BD
            from transgenic.evidence import b6_obs_cache as OC
        except ImportError as exc:
            self.skipTest(f"needs merged b6_bindings (mandatory after the b6-integration merge): {exc}")
        w = fx.make_world(self.tmp.name)
        paths = {"evidence": str(w.c0), "cache": str(w.cache_dir), "freeze": str(w.freeze_path),
                 "corpus": str(w.corpus), "references": {fx.SPECIES: str(w.fasta)}}
        record = trainer.verify_training_inputs(paths, roles=w.roles, cache=OC.ObsCache(w.cache_dir),
                                                b6_bindings=BD, aug_seed=rt.AUG_SEED)
        self.assertEqual(record["record_sha256"], BD.canonical_record_sha256(record))
        self.assertIn("freeze", record["provenance"])
        ident = rt.identity_from_bindings(record, BD.canonical_record_sha256)
        self.assertEqual(ident["binding.corpus_sha256"], record["corpus_sha256"])
        self.assertEqual(ident["binding.reference_sha256"], record["reference_sha256"])
        with self.assertRaises(BD.BindingError):                          # WP3 refuses a parsed freeze dict
            BD.verify_inputs(accept=("training",), evidence=paths["evidence"], roles=w.roles,
                             cache=OC.ObsCache(w.cache_dir), freeze=w.freeze, corpus=paths["corpus"],
                             aug_seed=rt.AUG_SEED, draws=None)
        result = self.train_arm(self.spec("C4", run="wp3rec", bindings=record,
                                          record_sha256_fn=BD.canonical_record_sha256))
        self.assertTrue(result["done"])


class CompileAndAbortTests(TrainerBase):
    def test_b6_compiles_without_ddp_graph_partitioning(self):
        import torch._dynamo as dynamo
        dynamo.reset()
        self.addCleanup(dynamo.reset)
        model = torch.nn.Linear(3, 2)
        x = torch.ones(1, 3)
        expected = model(x).detach()
        with dynamo.config.patch(optimize_ddp=True, suppress_errors=True):
            compiled = rt.compile_b6(model, backend="aot_eager")
            self.assertFalse(dynamo.config.optimize_ddp)
            self.assertFalse(dynamo.config.suppress_errors)
            result = compiled(x)
            result.sum().backward()
            self.assertTrue(torch.equal(result.detach(), expected))
            self.assertTrue(torch.equal(model.weight.grad, torch.ones_like(model.weight)))

    def test_resume_refuses_old_partitioned_compile_policy(self):
        first = self.spec("B6", run="old-compile", stop_at_step=1, compile_backend="aot_eager")
        self.train_arm(first)
        cfg_path, _ = rt.identity_paths(first.run_dir)
        with open(cfg_path) as fh:
            identity = json.load(fh)
        # Literal policy recorded by the pre-fix trainer, even on the same library versions.
        identity["compile_policy"] = "torch.compile backend=aot_eager; suppress_errors=False; fallback=failure"
        with open(cfg_path, "w") as fh:
            json.dump(identity, fh)
        with self.assertRaisesRegex(RuntimeError, "compile_policy"):
            self.train_arm(self.spec("B6", run="old-compile", resume="auto", compile_backend="aot_eager"))

    def test_injected_compile_failure_raises(self):
        import torch._dynamo as dynamo

        def broken_backend(gm, example_inputs):
            raise RuntimeError("injected compile failure")

        dynamo.reset()
        try:
            with self.assertRaisesRegex(Exception, "injected compile failure"):
                self.train_arm(self.spec("B6", run="broken", compile_backend=broken_backend))
            self.assertFalse(dynamo.config.suppress_errors)
        finally:
            dynamo.reset()

    def test_compiled_training_has_no_recompile_after_warmup(self):
        import torch._dynamo as dynamo
        dynamo.reset()
        try:
            result = self.train_arm(self.spec("B6", run="compiled", compile_backend="aot_eager", compile_warmup_steps=2))
            self.assertTrue(result["done"])
        finally:
            dynamo.reset()

    def test_compile_guard_and_kernel_check(self):
        guard = rt.CompileGuard(warmup_steps=2, graphs_per_shape=2)
        guard.observe(1, (1, 48), 1, False)
        guard.observe(2, (1, 48), 3, False)                # warm-up recompile allowed
        guard.observe(3, (1, 96), 4, False)                # new tier: allowed
        guard.observe(4, (1, 48), 4, False)
        with self.assertRaisesRegex(RuntimeError, "recompile after warm-up"):
            guard.observe(5, (1, 48), 5, False)
        with self.assertRaisesRegex(RuntimeError, "suppress_errors"):
            rt.CompileGuard(1).observe(1, (1, 48), 1, True)
        with self.assertRaisesRegex(RuntimeError, "no Inductor kernel"):
            rt.check_inductor_kernels("inductor", 10, 10)
        rt.check_inductor_kernels("inductor", 10, 11)

    def test_deterministic_mode_runs_and_is_recorded(self):
        from unittest.mock import patch
        try:
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
                result = self.train_arm(self.spec("C4", run="det", deterministic=True))
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertTrue(result["done"])
        with open(os.path.join(self.tmp.name, "det", "run_config_b6.json")) as fh:
            self.assertIs(json.load(fh)["deterministic"], True)

    def test_memory_abort_writes_marker_and_stops(self):
        spec = self.spec("B6", run="mem", memory_probe=lambda: 111 * 10 ** 9)
        with self.assertRaises(trainer.MemoryAbort):
            self.train_arm(spec)
        with open(os.path.join(spec.run_dir, rt.MEMORY_ABORT_MARKER)) as fh:
            marker = json.load(fh)
        self.assertEqual(marker["optimizer_step"], 1)
        self.assertFalse(rt.memory_abort([110 * 10 ** 9], None))
        self.assertTrue(rt.memory_abort([1, 110 * 10 ** 9 + 1], None, write=False))

    def test_alpha_outside_weight_decay(self):
        model = self.build()
        opt = rt.build_b6_optimizer(model, 1e-3, 0.02, expected_alphas=2)
        groups = {g["name"]: g for g in opt.param_groups}
        alphas = {id(p) for n, p in model.named_parameters() if n.endswith("chain_attn.alpha")}
        self.assertEqual({id(p) for p in groups["alpha_no_decay"]["params"]}, alphas)
        self.assertEqual(groups["alpha_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["decay"]["weight_decay"], 0.02)
        self.assertFalse(alphas & {id(p) for p in groups["decay"]["params"]})
        self.assertEqual(sum(len(g["params"]) for g in opt.param_groups), len(list(model.parameters())))
        with self.assertRaisesRegex(ValueError, "expected 12"):
            rt.build_b6_optimizer(model, 1e-3, 0.02, expected_alphas=12)

    def test_main_refuses_ungated_config(self):
        # The production config is GATED (author decision 15); an UNGATED copy must still be refused.
        cfg = os.path.join(self.tmp.name, "ungated.json")
        body = json.load(open(os.path.join(ROOT, "configs", "b6_400m_win_v3.json")))
        body.pop("gate_evidence")
        json.dump(dict(body, gate_status=rt.GATE_STATUS), open(cfg, "w"))
        with self.assertRaisesRegex(SystemExit, "UNGATED"):
            trainer.main(["--config", cfg, "--arm", "B6", "--bindings", "x", "--output-dir", self.tmp.name])
        with self.assertRaises(SystemExit):
            rt.parse_args_b6(["--config", cfg, "--arm", "B6", "--bindings", "x", "--output-dir", "o", "--seed", "456"])


class ConfigTests(unittest.TestCase):
    def test_b6_config_is_win_v3_plus_frozen_block(self):
        b5 = json.load(open(os.path.join(ROOT, "configs", "b5_400m_win_v3.json")))
        cfg = rt.load_b6_config(os.path.join(ROOT, "configs", "b6_400m_win_v3.json"))
        for key in ("d_model_encoder", "d_model_decoder", "encoder_layers", "decoder_layers", "attention_window",
                    "dropout", "lr", "weight_decay", "effective_batch_size", "max_epochs", "window_tiers",
                    "max_encoder_seqlen", "vocab_size", "max_label_tokens", "encoder_model"):
            self.assertEqual(cfg[key], b5[key], key)
        self.assertEqual(cfg["gate_status"], rt.GATED_STATUS)          # author decision 15 (2026-09-26)
        kw = rt.b6_model_kwargs(cfg)
        self.assertEqual((kw["evidence_channels"], kw["b6_chain_d_model"], kw["b6_chain_heads"]), (31, 512, 8))


if __name__ == "__main__":
    unittest.main()
