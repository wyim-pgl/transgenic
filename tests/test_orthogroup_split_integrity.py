"""§7 split table: orthogroup-level 75/10/15 seed 123, strict held-out -> test, no maize, RC inherits."""
import random


def _orthogroups(n_genes=3000, n_groups=1000, seed=1):
    rng = random.Random(seed)
    return {f"g{i}": f"OG{rng.randrange(n_groups):04d}" for i in range(n_genes)}


def test_make_split_is_deterministic_and_orthogroup_level(gsf):
    og = _orthogroups()
    a = gsf.make_split(og, seed=123, fractions=(0.75, 0.10, 0.15), strict_holdout=set())
    b = gsf.make_split(og, seed=123, fractions=(0.75, 0.10, 0.15), strict_holdout=set())
    assert a == b
    by_group = {}
    for g, s in a.items():
        by_group.setdefault(og[g], set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())
    frac = {k: sum(1 for v in a.values() if v == k) / len(a) for k in ("train", "valid", "test")}
    assert 0.65 < frac["train"] < 0.85 and 0.05 < frac["valid"] < 0.15 and 0.08 < frac["test"] < 0.22


def test_strict_holdout_genes_and_their_orthogroups_are_test(gsf):
    og = _orthogroups()
    held = {"g1", "g2"}
    split = gsf.make_split(og, seed=123, fractions=(0.75, 0.10, 0.15), strict_holdout=held)
    for g, grp in og.items():
        if grp in {og[h] for h in held}:
            assert split[g] == "test"


def test_validate_split_reports_every_violation_class(gsf):
    rows = [
        {"species_id": "Athaliana", "gene_id": "g1", "orthogroup_id": "OG1", "split": "train", "is_rc": False, "strict_holdout": False},
        {"species_id": "Athaliana", "gene_id": "g1", "orthogroup_id": "OG1", "split": "test", "is_rc": True, "strict_holdout": False},   # RC differs
        {"species_id": "Athaliana", "gene_id": "g2", "orthogroup_id": "OG1", "split": "test", "is_rc": False, "strict_holdout": False},  # OG crosses splits
        {"species_id": "Athaliana", "gene_id": "g3", "orthogroup_id": "OG2", "split": "train", "is_rc": False, "strict_holdout": True},  # held-out not test
        {"species_id": "Zmays", "gene_id": "Zm1", "orthogroup_id": "OG3", "split": "train", "is_rc": False, "strict_holdout": False},   # maize present
    ]
    v = gsf.validate_split(rows, excluded_species={"Zmays"})
    joined = "\n".join(v).lower()
    for key in ("rc", "orthogroup", "strict", "zmays"):
        assert key in joined, f"missing violation class {key}: {v}"


def test_validate_split_clean_table_has_no_violations(gsf):
    rows = [
        {"species_id": "Athaliana", "gene_id": "g1", "orthogroup_id": "OG1", "split": "train", "is_rc": False, "strict_holdout": False},
        {"species_id": "Athaliana", "gene_id": "g1", "orthogroup_id": "OG1", "split": "train", "is_rc": True, "strict_holdout": False},
        {"species_id": "Osativa", "gene_id": "o1", "orthogroup_id": "OG1", "split": "train", "is_rc": False, "strict_holdout": False},
        {"species_id": "Athaliana", "gene_id": "g3", "orthogroup_id": "OG2", "split": "test", "is_rc": False, "strict_holdout": True},
    ]
    assert gsf.validate_split(rows, excluded_species={"Zmays"}) == []


def test_validation_only_evidence_never_labels_train_or_valid(gsf):
    labels = [{"gene_id": "g1", "split": "train", "source_role": "b1_validation_only"}, {"gene_id": "g2", "split": "test", "source_role": "b1_validation_only"}, {"gene_id": "g3", "split": "train", "source_role": "c2_training_eligible"}]
    v = gsf.validate_evidence_roles(labels)
    assert len(v) == 1 and "g1" in v[0]
