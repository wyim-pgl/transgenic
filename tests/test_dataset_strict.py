"""A35 applied to the dataset (issue #62): a row that cannot be used stops a frozen-recipe run.

`isoformDataHyena.__getitem__` used to answer every failure by returning a randomly chosen different
row with one line on stderr. Counts stayed right, the loss stayed plausible, and the epoch quietly
trained on a resampled distribution — the same shape as the batch loop A35 closed one level above.
`strict=True` raises instead; the legacy path keeps the substitute so published behaviour is unchanged.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
duckdb = pytest.importorskip("duckdb")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def dataset_cls():
    from transgenic.datasets.datasets import isoformDataHyena
    return isoformDataHyena


def _b5_db(path, rows):
    """A minimal B5-shaped geneList: rows is a list of (rn, geneModel, sequence)."""
    con = duckdb.connect(str(path))
    con.sql("CREATE TABLE geneList (rn INT, geneModel VARCHAR, start INT, fin INT, strand VARCHAR, "
            "chromosome VARCHAR, sequence VARCHAR, gff VARCHAR, static_fpb INT, static_tpb INT, "
            "five_prime_buf INT, three_prime_buf INT, species_id VARCHAR, split VARCHAR, train_weight DOUBLE)")
    for rn, gm, seq in rows:
        con.execute("INSERT INTO geneList VALUES (?,?,0,?,'+','Chr1',?,'0|CDS1|300|+|A>CDS1',0,0,0,0,'Athaliana','train',1.0)",
                    [rn, gm, max(len(seq), 1), seq])
    con.close()


def _make(dataset_cls, db, **kw):
    return dataset_cls(str(db), mode="train", split="train", gff_vocab_version="v3", **kw)


def test_empty_sequence_raises_under_strict(tmp_path, dataset_cls):
    db = tmp_path / "b5.db"
    _b5_db(db, [(1, "good", "ACGT" * 32), (2, "empty", "")])
    ds = _make(dataset_cls, db, strict=True)
    bad = ds._index_map.index(2)
    with pytest.raises(RuntimeError) as e:
        ds[bad]
    msg = str(e.value)
    assert "rn=2" in msg and "empty" in msg.lower() and "A35" in msg


def test_empty_sequence_substitutes_without_strict(tmp_path, dataset_cls, capsys):
    """The legacy path is deliberately unchanged: it returns some other row and only warns."""
    db = tmp_path / "b5.db"
    _b5_db(db, [(1, "good", "ACGT" * 32), (2, "empty", "")])
    ds = _make(dataset_cls, db, strict=False)
    bad = ds._index_map.index(2)
    out = ds[bad]
    assert out[3] == "good"                                   # substituted, not the requested row
    assert "empty sequence" in capsys.readouterr().err


def test_unreadable_row_raises_under_strict(tmp_path, dataset_cls, monkeypatch):
    db = tmp_path / "b5.db"
    _b5_db(db, [(1, "good", "ACGT" * 32), (2, "also_good", "ACGT" * 32)])
    ds = _make(dataset_cls, db, strict=True)

    class Broken:
        def sql(self, *a, **k):
            raise duckdb.IOException("simulated read failure")

    monkeypatch.setattr(ds, "_get_connection", lambda: Broken())
    with pytest.raises(RuntimeError) as e:
        ds[0]
    msg = str(e.value)
    assert "could not be read" in msg and "IOException" in msg and "A35" in msg


def test_strict_defaults_off_so_published_behaviour_is_unchanged(tmp_path, dataset_cls):
    db = tmp_path / "b5.db"
    _b5_db(db, [(1, "good", "ACGT" * 32)])
    assert _make(dataset_cls, db).strict is False


def test_b5_trainer_constructs_the_dataset_strict():
    """The frozen-recipe path must ask for it; a default of False is only safe if the caller opts in."""
    src = (ROOT / "train" / "train_HyenaTransgenic.py").read_text()
    b5_calls = [l for l in src.splitlines() if "isoformDataHyena(" in l and 'split="train"' in l or
                "isoformDataHyena(" in l and 'split="valid"' in l]
    assert b5_calls, "the B5 dataset construction moved; update this test"
    for line in b5_calls:
        assert "strict=True" in line, line


def test_a40_empty_target_matches_contract_through_dataset(tmp_path, dataset_cls, monkeypatch, gsf):
    from transgenic.datasets import datasets
    from transgenic.model.tokenization_transgenic import GFFTokenizer

    class Encoder:
        pad_token_id = 4
        def __call__(self, seq, **kwargs):
            return {'input_ids': torch.zeros((1, len(seq) + 1), dtype=torch.long)}
    monkeypatch.setattr(datasets.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: Encoder())
    db = tmp_path / 'empty-target.db'
    _b5_db(db, [(1, 'empty-label', 'ACGT' * 32), (2, 'coding', 'ACGT' * 32)])
    with duckdb.connect(str(db)) as con:
        con.execute("UPDATE geneList SET gff='<empty>' WHERE rn=1")
    ds = _make(dataset_cls, db, strict=True)
    tokenizer = GFFTokenizer(vocab_version='v3')
    expected = ['<s>', '<empty>', '</s>']
    assert tokenizer._tokenize('<empty>') == expected
    assert tokenizer.encode('<empty>', add_special_tokens=False) == [tokenizer.vocab[t] for t in expected]
    for idx, label in enumerate(['<empty>', '0|CDS1|300|+|A>CDS1']):
        actual = ds[idx][2].tolist()[0]
        assert actual == [tokenizer.vocab[t] for t in tokenizer._tokenize(label)]
        assert len(actual) == gsf.count_tokens_v3(label)
    ds._con.close()
    ds._con = None
