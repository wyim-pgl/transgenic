# src/transgenic/model/tokenization_transgenic.py

- GFFTokenizer · class · L41-L320 — class GFFTokenizer(PreTrainedTokenizer)
- __init__ · method · L46-L126 — def __init__(self, vocab=None, vocab_version="v2", **kwargs)
- vocab_size · method · L129-L131 — def vocab_size(self)
- get_vocab · method · L133-L135 — def get_vocab(self)
- _tokenize · method · L137-L190 — def _tokenize(self, text)
- _tokenize_block_v2 · method · L192-L235 — def _tokenize_block_v2(self, text)
- _convert_token_to_id · method · L237-L239 — def _convert_token_to_id(self, token)
- _convert_id_to_token · method · L241-L243 — def _convert_id_to_token(self, index)
- convert_tokens_to_string · method · L245-L295 — def convert_tokens_to_string(self, tokens)
- save_vocabulary · method · L297-L320 — def save_vocabulary(self, save_directory, filename_prefix=None)
