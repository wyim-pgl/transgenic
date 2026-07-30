import os
import shutil
import subprocess
import tempfile

def count_isoforms_with_gffread(gff_path: str) -> int:
    """Use gffread to normalize to GTF and count transcript entries."""
    if shutil.which("gffread") is None:
        return -1

    with tempfile.NamedTemporaryFile(suffix=".gtf", delete=False) as tf:
        gtf_path = tf.name

    try:
        subprocess.run(
            ["gffread", "-T", gff_path, "-o", gtf_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        isoforms = 0
        with open(gtf_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) > 2 and cols[2] == "transcript":
                    isoforms += 1
        return isoforms
    finally:
        if os.path.exists(gtf_path):
            os.remove(gtf_path)

print(count_isoforms_with_gffread('/home/framazan/transgenic/examples/ath_chr4_predict.cleaned.gff'))