"""Grammar-constrained decoding for GSF (protocol A24). Pure Python: given the tokens generated so far it
returns the set of token strings the decoder may emit next, so that a beam can never produce
coordinate reversals, out-of-window coordinates, feature-list disorder, mixed strands, wrong phase
letters, undefined or duplicated transcript members, out-of-order transcript members, or a transcript
count that disagrees with the <txN> plan. Frame consistency (CDS length % 3) cannot be forced token by
token without deadlocks; it is reported by validate_gsf() and handled by the frozen structural filter.

Token stream (GFFTokenizer): <s> d+ NAME d+ STRAND PHASE (';' | <txN>) ... '>' NAME ('|' is not a token)
NAME* (<iso> NAME*)* </s>     -- v2 vocabulary; v1 uses ';' instead of <iso> and has no <txN>.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

DIGITS = tuple(str(i) for i in range(10))
PHASES_CDS = ("A", "B", "C")
FEATURE_TYPES = ("CDS", "five_prime_UTR", "three_prime_UTR")
CAPS = {"CDS": 150, "five_prime_UTR": 50, "three_prime_UTR": 50}
MAX_TX = 15


def feature_type(name: str) -> Optional[str]:
    for t in FEATURE_TYPES:
        if name.startswith(t) and name[len(t):].isdigit():
            return t
    return None


@dataclass
class State:
    stage: str = "start"                 # start | name | end | strand | phase | sep | after_tx | tx
    num: str = ""                        # digits of the number being emitted
    cur: Dict[str, object] = field(default_factory=dict)
    features: Dict[str, Tuple[int, int, str]] = field(default_factory=dict)   # name -> (start, end, type)
    counts: Dict[str, int] = field(default_factory=lambda: {t: 0 for t in FEATURE_TYPES})
    strand: Optional[str] = None
    last_start: int = -1
    planned_tx: Optional[int] = None
    transcripts: List[List[str]] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    v1: bool = False


def _advance(st: State, tok: str) -> State:
    """Apply one emitted token (assumed allowed) to the state."""
    if st.stage == "start":
        if tok in DIGITS:
            st.num += tok
            return st
        # a NAME ends the start number
        st.cur = {"start": int(st.num), "name": tok, "type": feature_type(tok)}
        st.num = ""
        st.stage = "end"
        return st
    if st.stage == "end":
        if tok in DIGITS:
            st.num += tok
            return st
        st.cur["end"] = int(st.num)
        st.num = ""
        st.cur["strand"] = tok
        st.strand = st.strand or tok
        st.stage = "phase"
        return st
    if st.stage == "phase":
        name = st.cur["name"]
        st.features[name] = (st.cur["start"], st.cur["end"], st.cur["type"])
        st.counts[st.cur["type"]] += 1
        st.last_start = st.cur["start"]
        st.stage = "sep"
        return st
    if st.stage == "sep":
        if tok == ";":
            st.stage = "start"
        elif tok.startswith("<tx"):
            st.planned_tx = int(tok[3:-1])
            st.stage = "after_tx"
        elif tok == ">":              # v1 vocabulary: no planning token
            st.v1 = True
            st.stage = "tx"
            st.transcripts.append([])
        return st
    if st.stage == "after_tx":
        st.stage = "tx"
        st.transcripts.append([])
        return st
    if st.stage == "tx":
        if tok in ("<iso>", ";"):
            st.transcripts.append([])
        elif tok == "</s>":
            st.stage = "done"
        else:
            st.transcripts[-1].append(tok)
        return st
    return st


def replay(tokens: Sequence[str]) -> State:
    st = State()
    for tok in tokens:
        if tok in ("<s>", "<pad>"):
            continue
        st = _advance(st, tok)
    return st


def _next_names(st: State) -> Set[str]:
    return {f"{t}{st.counts[t] + 1}" for t in FEATURE_TYPES if st.counts[t] < CAPS[t]}


def allowed_next(tokens: Sequence[str], window_len: int, v2: bool = True) -> Set[str]:
    """Token strings allowed after `tokens` (which start with <s>)."""
    st = replay(tokens)
    max_digits = len(str(window_len))
    if st.stage == "done":
        return {"<pad>"}
    if st.stage == "start":
        out: Set[str] = set()
        cand = st.num
        if len(cand) < max_digits:
            for d in DIGITS:
                if cand == "" and d == "0" and max_digits > 1:
                    # a leading zero is only valid as the single digit 0
                    if st.last_start <= 0:
                        out.add(d)
                    continue
                v = int(cand + d)
                if v < window_len - 1 and v >= st.last_start:
                    out.add(d)
                elif v < window_len - 1 and len(cand + d) < max_digits and int((cand + d) + "9" * (max_digits - len(cand + d) - 0)) >= st.last_start:
                    out.add(d)  # may still reach a value >= last_start with more digits
        if cand != "" and st.last_start <= int(cand) < window_len - 1:
            out |= _next_names(st)
        return out
    if st.stage == "end":
        out = set()
        cand = st.num
        start = st.cur["start"]
        if len(cand) < max_digits:
            for d in DIGITS:
                if cand == "" and d == "0":
                    continue
                v = int(cand + d)
                if v <= window_len and (v > start or len(cand + d) < max_digits):
                    out.add(d)
        if cand != "" and start < int(cand) <= window_len:
            out |= {st.strand} if st.strand else {"+", "-"}
        return out
    if st.stage == "phase":
        return set(PHASES_CDS) if st.cur["type"] == "CDS" else {"."}
    if st.stage == "sep":
        out = {";"} if any(st.counts[t] < CAPS[t] for t in FEATURE_TYPES) else set()
        if st.counts["CDS"] >= 1:
            out |= {f"<tx{i}>" for i in range(1, min(MAX_TX, len(st.features)) + 1)} if v2 else {">"}
        return out
    if st.stage == "after_tx":
        return {">"}
    if st.stage == "tx":
        cur = st.transcripts[-1]
        used = set(cur)
        out = set()
        for name, (s, e, t) in st.features.items():
            if name in used:
                continue
            if cur:
                ls, le, _ = st.features[cur[-1]]
                if st.strand == "+" and s < le:
                    continue
                if st.strand == "-" and e > ls:
                    continue
            out.add(name)
        has_cds = any(st.features[n][2] == "CDS" for n in cur)
        n_tx = len(st.transcripts)
        target = st.planned_tx if st.planned_tx else None
        if cur and (has_cds or not any(f[2] == "CDS" for f in st.features.values())):
            if target is None or n_tx < target:
                if v2 and target is not None and n_tx < target:
                    out.add("<iso>")
                if not v2 or target is None:
                    out.add(";")
                    out.add("</s>")
            if target is None or n_tx == target:
                out.add("</s>")
        return out
    return set()


def validate_gsf(gsf: str, window_len: int) -> List[str]:
    """Post-hoc grammar audit of a complete GSF string (used for reporting and by the structural filter)."""
    v: List[str] = []
    feats_str, _, tx_str = gsf.partition(">")
    feats: Dict[str, Tuple[int, int, str, str, str]] = {}
    strands = set()
    prev_start = -1
    for item in [x for x in feats_str.split(";") if x]:
        try:
            s, name, e, strand, ph = item.split("|")
            s, e = int(s), int(e)
        except ValueError:
            v.append(f"malformed feature {item!r}")
            continue
        t = feature_type(name)
        if t is None:
            v.append(f"unknown feature name {name}")
        if not (0 <= s < e <= window_len):
            v.append(f"{name}: coordinates {s}-{e} outside [0,{window_len}] or reversed")
        if s < prev_start:
            v.append(f"{name}: feature list not coordinate-sorted")
        prev_start = s
        if t == "CDS" and ph not in PHASES_CDS:
            v.append(f"{name}: CDS phase {ph!r}")
        if t and t != "CDS" and ph != ".":
            v.append(f"{name}: UTR phase {ph!r}")
        if name in feats:
            v.append(f"{name}: duplicate feature name")
        feats[name] = (s, e, t or "", strand, ph)
        strands.add(strand)
    if len(strands) > 1:
        v.append("mixed strands")
    strand = next(iter(strands), "+")
    txs = [t.split("|") for t in tx_str.split(";") if t]
    if not txs:
        v.append("no transcripts")
    for i, tx in enumerate(txs, 1):
        seen = set()
        last = None
        cds_len = 0
        for name in tx:
            if name not in feats:
                v.append(f"transcript {i}: undefined feature {name}")
                continue
            if name in seen:
                v.append(f"transcript {i}: repeated feature {name}")
            seen.add(name)
            s, e, t, _, ph = feats[name]
            if last is not None:
                ls, le = feats[last][0], feats[last][1]
                if (strand == "+" and s < le) or (strand == "-" and e > ls):
                    v.append(f"transcript {i}: {name} out of transcription order / overlapping")
            last = name
            if t == "CDS":
                cds_len += e - s
        if not any(feats.get(n, ("", "", ""))[2] == "CDS" for n in tx):
            v.append(f"transcript {i}: no CDS")
        elif cds_len % 3:
            v.append(f"transcript {i}: CDS length {cds_len} not a multiple of 3")
    return v
