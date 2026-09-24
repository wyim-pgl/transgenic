"""Chain-less DDP ranks (plan v4.1 section 9): 2 CPU processes over gloo, one rank with real chains and one
with every chain masked (C2/C7 batches, or a B6 rank whose tiles all lost their chains). Every chain
parameter is in both ranks' graphs, so DDP reduces the same buckets on both, and the replicas stay equal."""
import json
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, out_dir):
    # Diagnostics for the intermittent SIGABRT (review 2): native stderr (gloo/c10d messages, abort reasons) and
    # Python tracebacks of every thread on a fatal signal go to a per-rank log that the test prints on failure.
    import faulthandler
    log = open(os.path.join(out_dir, f"rank{rank}.stderr"), "w", buffering=1)
    os.dup2(log.fileno(), 2)
    faulthandler.enable(file=log, all_threads=True)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), TORCH_DISTRIBUTED_DEBUG="DETAIL",
                      TORCH_SHOW_CPP_STACKTRACES="1")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    from tests.test_b6_model import TinyHyena, chain_batch, masked_chains, tiny_b6
    from transgenic.model import modeling_HyenaTransgenic as b5
    from transgenic.model import modeling_b6 as b6
    from transgenic.training import b6_runtime as rt
    with patch.object(b5.AutoConfig, "from_pretrained", side_effect=lambda *a, **k: SimpleNamespace()), \
            patch.object(b5.AutoModel, "from_config", side_effect=lambda config, **k: TinyHyena(config)):
        torch.manual_seed(123)
        model = b6.B6TransgenicForConditionalGeneration(tiny_b6())
    with torch.no_grad():
        model.get_encoder().evidence_proj.weight.normal_(0, 0.5)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    opt = rt.build_b6_optimizer(ddp, 1e-2, 0.02, expected_alphas=2)
    g = torch.Generator().manual_seed(rank)
    ids = torch.randint(0, 16, (1, 48), generator=g)
    labels = torch.tensor([[7, 9, 30, 2]])
    tracks = torch.rand(1, 48, 31, generator=g).half() if rank == 0 else torch.zeros(1, 48, 31, dtype=torch.half)
    chains = chain_batch(1, 8, [30]) if rank == 0 else masked_chains(1)
    record = []
    for _ in range(3):
        loss = ddp(ids, labels=labels, evidence_tracks=tracks, **chains).loss
        loss.backward()
        alpha_grad = float(model.transgenic.decoder.layers[0].chain_attn.alpha.grad)
        wd_grad = float(model.transgenic.chain_memory.w_d.weight.grad.abs().sum())
        record.append(dict(alpha_grad=alpha_grad, w_d_grad=wd_grad))
        opt.step()
        opt.zero_grad(set_to_none=True)
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    record.append(dict(replicas_equal=all(torch.equal(gathered[0], x) for x in gathered),
                       alphas=model.chain_alphas()))
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(record, fh)
    dist.destroy_process_group()


class ChainlessRankDDPTests(unittest.TestCase):
    def test_chainless_rank_trains_in_step_with_a_chain_rank(self):
        with tempfile.TemporaryDirectory() as out:
            try:
                mp.start_processes(_worker, args=(2, _free_port(), out), nprocs=2, join=True, start_method="spawn")
            except Exception as exc:
                logs = "\n".join(f"--- rank{r}.stderr ---\n" + open(os.path.join(out, f"rank{r}.stderr")).read()
                                 for r in range(2) if os.path.exists(os.path.join(out, f"rank{r}.stderr")))
                self.fail(f"{type(exc).__name__}: {exc}\n{logs}")
            ranks = [json.load(open(os.path.join(out, f"rank{r}.json"))) for r in range(2)]
        for r in ranks:
            self.assertTrue(r[-1]["replicas_equal"])
            self.assertEqual(r[-1]["alphas"], ranks[0][-1]["alphas"])
            self.assertNotEqual(r[0]["alpha_grad"], 0.0)          # reduced: the chain rank's gradient arrives
            self.assertEqual(r[0]["w_d_grad"], 0.0)                 # alpha = 0 at step 0: gate closed
            self.assertGreater(r[1]["w_d_grad"], 0.0)              # gate opened by the first update
        self.assertTrue(all(a != 0.0 for a in ranks[0][-1]["alphas"]))


if __name__ == "__main__":
    unittest.main()
