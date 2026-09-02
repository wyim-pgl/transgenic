"""HuggingFace LogitsProcessor that enforces the GSF grammar during generation (protocol A24).

Usage:
    from transgenic.model.constrained_decoding import GSFGrammarLogitsProcessor
    lp = GSFGrammarLogitsProcessor(gff_tokenizer, window_lens=[6144, 12288, ...])   # one window length per batch row
    model.generate(..., logits_processor=LogitsProcessorList([lp]), num_beams=k)
Every step, the tokens generated so far for each beam are replayed through gsf_grammar.allowed_next and all
other vocabulary entries get -inf. The processor is stateless across calls (it re-parses the prefix), so it
works with beam search, sampling and resumption alike.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import LogitsProcessor

from ..utils import gsf_grammar


class GSFGrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, window_lens: Sequence[int], num_beams: int = 1, v2: bool = True, v3: bool = False):
        self.tok = tokenizer
        self.id2tok = {i: t for t, i in tokenizer.get_vocab().items()}
        self.tok2id = {t: i for i, t in self.id2tok.items()}
        self.window_lens = list(window_lens)
        self.num_beams = num_beams
        self.v2 = v2
        self.v3 = v3
        self.stats = {"steps": 0, "masked_positions": 0}

    def _row_window(self, row: int) -> int:
        # rows are ordered batch-major: beam b of sample i is row i*num_beams + b
        return self.window_lens[min(row // max(1, self.num_beams), len(self.window_lens) - 1)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.shape[0]):
            tokens = [self.id2tok.get(int(i), "<unk>") for i in input_ids[row].tolist()]
            allowed = gsf_grammar.allowed_next(tokens, self._row_window(row), v2=self.v2, v3=self.v3)
            if not allowed:                      # should not happen; fall back to closing the sequence
                allowed = {"</s>"}
            ids = [self.tok2id[t] for t in allowed if t in self.tok2id]
            mask[row, ids] = 0.0
            self.stats["masked_positions"] += scores.shape[1] - len(ids)
        self.stats["steps"] += 1
        return scores + mask
