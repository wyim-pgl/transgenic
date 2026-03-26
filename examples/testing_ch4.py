import os
import shutil
import subprocess
import tempfile
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transgenic.datasets.datasets import hyena_collate_fn, isoformDataHyena
from transgenic.datasets.preprocess import genome2GSFDataset
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.tokenization_transgenic import GFFTokenizer
from transgenic.utils.gsf import gffString2GFF3


def generate_with_adaptive_chunks(
    model: Any,
    ii: torch.Tensor,
    am: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Generate with automatic chunk downscaling when CUDA OOM happens."""
    batch_n = ii.size(0)
    chunk_size = batch_n

    while chunk_size >= 1:
        try:
            out_chunks = []
            with torch.inference_mode():
                for start in range(0, batch_n, chunk_size):
                    end = min(start + chunk_size, batch_n)
                    out = model.generate(
                        inputs=ii[start:end],
                        attention_mask=am[start:end],
                        num_return_sequences=1,
                        max_length=max_length,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True,
                    )
                    seq = out.sequences if hasattr(out, "sequences") else out
                    out_chunks.append(seq)
            return torch.cat(out_chunks, dim=0)
        except torch.OutOfMemoryError:
            if ii.is_cuda:
                torch.cuda.empty_cache()
            chunk_size //= 2

    raise RuntimeError(
        "Generation failed due to CUDA OOM even with chunk_size=1. "
        "Lower BATCH_SIZE or MAX_GEN_LEN."
    )


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


def main() -> None:
    db_path = "ath_chr4_predict.db"
    output_gff_path = "ath_chr4_predict.gff"

    genome2GSFDataset(
        "ATH_Chr4.fas",
        "ATH_Chr4_gene.bed",
        db_path,
        anoType="bed",
        mode="predict",
    )

    ds = isoformDataHyena(db_path, mode="inference")

    cpu_count = os.cpu_count() or 4
    default_workers = min(8, max(2, cpu_count // 2))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    num_workers = int(os.getenv("NUM_WORKERS", str(default_workers)))
    prefetch_factor = int(os.getenv("PREFETCH_FACTOR", "4"))
    max_gen_len = int(os.getenv("MAX_GEN_LEN", "2048"))

    dl_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": hyena_collate_fn,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, prefetch_factor)
        dl_kwargs["persistent_workers"] = True

    dl = DataLoader(**dl_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    best_model_dir = "/home/framazan/checkpoints/accelerate_epoch7_step16571"
    if not os.path.isdir(best_model_dir):
        raise FileNotFoundError(
            f"{best_model_dir} not found. Run training until a checkpoint is saved."
        )

    model = transgenicForConditionalGeneration.from_pretrained(
        best_model_dir,
        local_files_only=True,
    )
    model = cast(Any, model)
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(
        f"Runtime config -> batch_size={batch_size}, num_workers={num_workers}, "
        f"prefetch_factor={max(2, prefetch_factor)}, max_gen_len={max_gen_len}"
    )

    gff_tokenizer = GFFTokenizer()
    genes_predicted = 0

    with open(output_gff_path, "w", encoding="utf-8") as out_f:
        for batch in tqdm(dl, desc="Predicting"):
            ii = batch[0].to(device, non_blocking=True)
            am = batch[1].to(device, non_blocking=True)

            output_tokens = generate_with_adaptive_chunks(
                model=model,
                ii=ii,
                am=am,
                max_length=max_gen_len,
            )

            decoded_batch = gff_tokenizer.batch_decode(
                output_tokens.detach().cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for idx, pred_raw in enumerate(decoded_batch):
                pred = (
                    pred_raw.replace("|</s>", "")
                    .replace("</s>", "")
                    .replace("<s>", "")
                )
                gff_lines = gffString2GFF3(
                    pred,
                    batch[4][idx],
                    batch[5][idx],
                    f"GM={batch[3][idx]}",
                )
                for line in gff_lines:
                    out_f.write(line + "\n")
                genes_predicted += 1

    isoforms_predicted = count_isoforms_with_gffread(output_gff_path)

    print(f"Genes predicted: {genes_predicted:,}")
    if isoforms_predicted >= 0:
        print(f"Isoforms predicted (gffread): {isoforms_predicted:,}")
    else:
        print("Isoforms predicted (gffread): unavailable (gffread not found in PATH)")


if __name__ == "__main__":
    main()