"""§4 token cap 2,048 with the v2 tokenizer counting rule (digits, names, ';', <txN>, '>', <iso>, </s>)."""
import pytest


def test_count_tokens_v2_matches_tokenizer_rule_on_readme_example(gsf):
    s = "0|CDS1|51|+|A;100|CDS2|181|+|C;250|CDS3|301|+|B>CDS1|CDS2|CDS3"
    # <s> + features: (1+1+2+1+1) + ';' ... last ';' -> <tx1>; then '>' ; transcripts 3 names; </s>
    expected = 1 + (1 + 1 + 2 + 1 + 1 + 1) + (3 + 1 + 3 + 1 + 1 + 1) + (3 + 1 + 3 + 1 + 1 + 1) + 1 + 3 + 1
    assert gsf.count_tokens_v2(s) == expected


def test_iso_tokens_counted_between_transcripts(gsf):
    s = "0|CDS1|51|+|A;100|CDS2|181|+|C>CDS1|CDS2;CDS2"
    assert gsf.count_tokens_v2(s) == gsf.count_tokens_v2("0|CDS1|51|+|A;100|CDS2|181|+|C>CDS1|CDS2") + 1 + 1  # <iso> + CDS2


def test_token_cap_rejects_over_2048(gsf):
    feats = ";".join(f"{i*20}|CDS{i+1}|{i*20+10}|+|A" for i in range(150))
    s = feats + ">" + "|".join(f"CDS{i+1}" for i in range(150))
    n = gsf.count_tokens_v2(s)
    if n > 2048:
        with pytest.raises(gsf.CapError):
            gsf.check_caps(s)
    else:
        gsf.check_caps(s)
    s_big = feats + ">" + ";".join("|".join(f"CDS{i+1}" for i in range(150)) for _ in range(15))
    assert gsf.count_tokens_v2(s_big) > 2048
    with pytest.raises(gsf.CapError):
        gsf.check_caps(s_big)
