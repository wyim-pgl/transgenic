"""b6_routea_agreement --population (plan §7.4 D4): one zero list per population, QC names the population and
its upstream shas.  Reuses the route A/B world of test_b6_routea_agreement."""
import csv
import json

from test_b6_routea_agreement import H, P, V, build_case, est, m  # noqa: F401  (est is a pytest fixture)


def run_pop(case, pop):
    d = case["dir"]
    out = f"zero.{pop}.tsv"
    argv = ["--species", "Athaliana", "--route-b", str(case["route_b"]), "--miniprot-gff", str(case["gff"]),
            "--accepted", str(case["accepted"]), "--mapping-ambiguous", str(case["ambiguous"]),
            "--est-c0", str(case["est_c0"]), "--protein-c0", str(case["protein_c0"]), "--population", pop,
            "--out-intervals", str(d / out), "--qc", str(d / (out + ".qc.json"))]
    m.main(argv)
    with open(d / out) as f:
        zero = {(z["reason"], z["start0"], z["end0"]) for z in csv.DictReader(f, delimiter="\t")}
    return json.loads((d / (out + ".qc.json")).read_text()), zero


def test_three_populations_differ_as_specified(tmp_path, est):  # noqa: F811
    # held-out EST row (would disagree with protein H's route A), a train-tiles-excluded row, a clean control
    b = [(H, "orthodb", "held", 40000, 40030, "+", 1), (V, "orthodb", "tte", 100000, 100030, "+", 1),
         (P, "orthodb", "train", 10000, 10030, "+", 1)]
    gff = [("MP1", H, "Chr1", "+", [(40001, 40030, 0)], 500, 0.9), ("MP2", V, "Chr1", "+", [(100001, 100030, 0)], 500, 0.9),
           ("MP3", P, "Chr1", "+", [(10001, 10030, 0)], 500, 0.9)]
    case = build_case(tmp_path, est, b, gff)
    qt, zt = run_pop(case, "train_track")
    qv, zv = run_pop(case, "valid_track")
    qe, ze = run_pop(case, "eval_input")
    dis = "route_a_b_disagreement"
    assert zt == {(dis, "10000", "10030")}
    assert zv == {(dis, "10000", "10030"), (dis, "100000", "100030")}      # train-tiles-excluded admitted
    assert ze == {(dis, "40000", "40030")}                                 # only the excluded_heldout EST
    for q, pop in ((qt, "train_track"), (qv, "valid_track"), (qe, "eval_input")):
        assert q["population"] == pop
        assert set(q["upstream"]) == {"protein_c0", "est_c0"} and all(len(v) == 64 for v in q["upstream"].values())
    assert qe["route_b_leakage_filter"]["rows_used"] == 1
