# src/transgenic/model/constrained_decoding.py

- GSFGrammarLogitsProcessor · class · L21-L47 — class GSFGrammarLogitsProcessor(LogitsProcessor)
- __init__ · method · L22-L30 — def __init__(self, tokenizer, window_lens: Sequence[int], num_beams: int = 1, v2: bool = True, v3: bool = False)
- _row_window · method · L32-L34 — def _row_window(self, row: int) -> int: # rows are ordered batch-major: beam b of sample i is row i*num_beams + b
- __call__ · method · L36-L47 — def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor
