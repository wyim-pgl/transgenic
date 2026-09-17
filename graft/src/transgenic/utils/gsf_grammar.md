# src/transgenic/utils/gsf_grammar.py

- feature_type · function · L23-L27 — def feature_type(name: str) -> Optional[str]
- State · class · L31-L46 — class State
- _new_gene · function · L49-L61 — def _new_gene(st: State) -> State
- _advance · function · L64-L124 — def _advance(st: State, tok: str) -> State
- replay · function · L127-L133 — def replay(tokens: Sequence[str]) -> State
- _next_names · function · L136-L137 — def _next_names(st: State) -> Set[str]
- allowed_next · function · L140-L151 — def allowed_next(tokens: Sequence[str], window_len: int, v2: bool = True, v3: bool = False) -> Set[str]
- allowed_next_block · function · L154-L226 — def allowed_next_block(st: State, window_len: int, v2: bool = True, v3: bool = False) -> Set[str]
- validate_gsf · function · L229-L288 — def validate_gsf(gsf: str, window_len: int) -> List[str]
