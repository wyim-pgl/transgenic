"""Scale: a synthetic ONT run of 200,000 reads builds in a streaming pass under a stated RSS bound.

The builder runs in a child process with --memory-limit 256MB; its peak RSS is read with
resource.getrusage(RUSAGE_CHILDREN).ru_maxrss (the maximum over waited-for children, KiB on Linux).
"""
import json
import os
import resource
import subprocess
import sys

import pytest

import test_b6_c0_world as W

duckdb = pytest.importorskip("duckdb")

N_READS = 200_000
RSS_BOUND_MB = 900          # interpreter + duckdb/pyarrow (~150 MB) + DuckDB 256 MB limit + bounded batches
RUN = "SRRBIG"


def _write_run(tmp):
    align, audit = tmp / "align", tmp / "audit"
    d, ad = align / W.SPECIES / "proj1" / RUN, audit / W.SPECIES / "proj1" / RUN
    d.mkdir(parents=True)
    ad.mkdir(parents=True)
    cl_f = "minimap2 -t 32 -ax splice -k14 -G 200000 --secondary=no --cs=long ref.fa reads.fq.gz"
    cl_a = cl_f.replace("--secondary=no", "--secondary=yes -N 50 -p 0.5")
    body = []
    for i in range(N_READS):
        pos = 1001 + (i * 37) % 240_000
        body.append(f"r{i}\t0\tChr1\t{pos}\t60\t100M\t*\t0\t0\t*\t*\tcs:Z::100\tAS:i:100\tts:A:+")
    body = "\n".join(body) + "\n"
    for path, cl in ((d / f"{RUN}.sam", cl_f), (ad / f"{RUN}.audit.sam", cl_a)):
        path.write_text(W.header(cl) + body)
    (d / "DONE").write_text(f"sam_md5={W.md5(d / f'{RUN}.sam')}\nuf=none\n")
    (ad / "DONE").write_text(f"sam_md5={W.md5(ad / f'{RUN}.audit.sam')}\nsource=ont\nrun={RUN}\n")
    meta = d / "c0_metadata.tsv"
    with open(meta, "w") as fh:
        fh.write("read\tlibrary\tbioproject\tgenotype_stratum\tingestion_qc_pass\tmapping_ambiguous\tprotocol\tumi_status\n")
        fh.write(f"*\tSRX9000001\tPRJ2\treference\ttrue\tfalse\tdirect_RNA\tabsent\n")
        fh.writelines(f"r{i}\t\t\t\t\t\t\t\n" for i in range(N_READS))
    (d / "c0_metadata.tsv.sha256").write_text(f"{W.sha(meta)}  c0_metadata.tsv\n")
    roles = W.write_roles(tmp, extra_rows=[f"{W.ONT_DATASET}\t{RUN}\t{W.SPECIES}\treference\tPromethION\tRNA-Seq\t1\t{N_READS}"
                                           "\t\t\t\tc2_training_eligible\tt\t"])
    lr = roles[0]
    lr.write_text("\n".join(l for l in lr.read_text().splitlines() if "\tSRR1\t" not in l) + "\n")
    return roles


def test_200k_read_run_streams_under_the_rss_bound(tmp_path):
    world = W.build_corpus(tmp_path)
    roles = _write_run(tmp_path)
    args = W.cli_args(world, tmp_path, source="ont", roles=roles, extra=("--memory-limit", "256MB", "--threads", "2"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    script = W.ROOT / "revision" / "scripts" / "evidence" / "b6_build_c0.py"
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.run([sys.executable, str(script), *args], env=env, capture_output=True, text=True)
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    assert proc.returncode == 0, proc.stderr[-2000:]
    record = json.load(open(args[args.index("--qc") + 1]))
    print(f"\nscale: {N_READS} reads, child peak RSS {peak_mb:.0f} MB (earlier children peak {before / 1024:.0f} MB)")
    assert record["qc"]["reads_in"] == N_READS and record["qc"]["molecules_in"] == N_READS
    con = duckdb.connect(args[args.index("--out") + 1], read_only=True)
    assert con.execute("SELECT count(*) FROM evidence_alignment").fetchone()[0] == N_READS
    con.close()
    assert peak_mb < RSS_BOUND_MB
