"""
Chr4 evaluation module for TransGenic training.

Provides `evaluate_chr4()` which can be called:
  1. Standalone via CLI: `python testing_ch4.py --checkpoint_dir <path>`
  2. From the training loop after each epoch (pass a model directly)

Runs inference on 500 random A. thaliana Chr4 gene regions, writes predicted
GFF3, runs gffcompare against the reference, parses the .stats file, and
returns a structured metrics dict. Also counts isoforms via gffread.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, cast

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset

from transgenic.datasets.datasets import hyena_collate_fn, isoformDataHyena
from transgenic.datasets.preprocess import genome2GSFDataset
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.tokenization_transgenic import GFFTokenizer
from transgenic.utils.gsf import gffString2GFF3

# ── Paths (all relative to THIS file's directory) ──────────────────────────
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
CHR4_FASTA = os.path.join(_EXAMPLES_DIR, "ATH_Chr4.fas")
CHR4_BED   = os.path.join(_EXAMPLES_DIR, "ATH_Chr4_gene.bed")
CHR4_REF   = os.path.join(_EXAMPLES_DIR, "ATH_Chr4.sorted.gff3")


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_batch(model, ii, am, max_length=2048):
    """Generate with automatic chunk downscaling on OOM."""
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
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=5,
                    )
                    seq = out.sequences if hasattr(out, "sequences") else out
                    out_chunks.append(seq)
            return torch.cat(out_chunks, dim=0)
        except torch.OutOfMemoryError:
            if ii.is_cuda:
                torch.cuda.empty_cache()
            chunk_size //= 2
    raise RuntimeError("Generation OOM even at chunk_size=1")


def _parse_gffcompare_stats(stats_path: str) -> Dict[str, float]:
    """Parse a gffcompare .stats file and return a flat metrics dict."""
    metrics = {}
    if not os.path.isfile(stats_path):
        return metrics

    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            # Parse "Query mRNAs : 123 in 456 loci (~7.8 transcripts per locus)"
            m = re.search(r"Query mRNAs\s*:\s*(\d+)\s+in\s+(\d+)\s+loci", line)
            if m:
                metrics["query_mRNAs"] = int(m.group(1))
                metrics["query_loci"] = int(m.group(2))
                tpl = re.search(r"~([\d.]+)\s+transcripts per locus", line)
                if tpl:
                    metrics["transcripts_per_locus"] = float(tpl.group(1))
                continue

            # Parse sensitivity/precision lines like:
            #    Base level:    88.1     |    67.7    |
            m = re.match(
                r"\s*(Base|Exon|Intron|Intron chain|Transcript|Locus)\s+level:\s+([\d.]+)\s+\|\s+([\d.]+)",
                line,
            )
            if m:
                level = m.group(1).strip().lower().replace(" ", "_")
                metrics[f"{level}_sensitivity"] = float(m.group(2))
                metrics[f"{level}_precision"] = float(m.group(3))
                continue

            # Parse "Matching transcripts: 2098"
            m = re.match(r"\s*Matching (intron chains|transcripts|loci):\s+(\d+)", line)
            if m:
                key = m.group(1).replace(" ", "_")
                metrics[f"matching_{key}"] = int(m.group(2))

    return metrics


def _count_isoforms_gffread(gff_path: str) -> int:
    """Count transcripts via gffread normalization."""
    if shutil.which("gffread") is None:
        return -1
    with tempfile.NamedTemporaryFile(suffix=".gtf", delete=False) as tf:
        gtf_path = tf.name
    try:
        subprocess.run(
            ["gffread", "-T", gff_path, "-o", gtf_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        count = 0
        with open(gtf_path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    cols = line.split("\t")
                    if len(cols) > 2 and cols[2] == "transcript":
                        count += 1
        return count
    except Exception:
        return -1
    finally:
        if os.path.exists(gtf_path):
            os.remove(gtf_path)


def _build_subset_ref_gff(ref_gff_path: str, gene_names: set, output_path: str) -> str:
    """Extract reference GFF3 features for only the specified gene names.

    Handles the parent-child hierarchy:
      gene  → ID=AT4GXXXXX
      mRNA  → Parent=AT4GXXXXX; ID=PAC:YYYYY
      exon/CDS → Parent=PAC:YYYYY

    Returns:
        Path to the subset reference GFF3 file.
    """
    # Pass 1: collect mRNA/transcript IDs that belong to our gene set
    child_ids = set()  # IDs of mRNAs whose parent is in gene_names
    with open(ref_gff_path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = cols[8]
            # Check if this feature's Parent is one of our genes
            parent_match = re.search(r"Parent=([^;]+)", attrs)
            if parent_match:
                parent_id = parent_match.group(1)
                if parent_id in gene_names:
                    # This is a direct child (mRNA) of a selected gene
                    id_match = re.search(r"ID=([^;]+)", attrs)
                    if id_match:
                        child_ids.add(id_match.group(1))

    # Pass 2: write features that belong to selected genes
    with open(ref_gff_path, "r") as fh, open(output_path, "w") as out:
        out.write("##gff-version 3\n")
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = cols[8]
            # Gene line: check if ID is in our set
            id_match = re.search(r"ID=([^;]+)", attrs)
            if id_match and id_match.group(1) in gene_names:
                out.write(line if line.endswith("\n") else line + "\n")
                continue
            # mRNA or child feature: check Parent
            parent_match = re.search(r"Parent=([^;]+)", attrs)
            if parent_match:
                parent_id = parent_match.group(1)
                if parent_id in gene_names or parent_id in child_ids:
                    # Track grandchild IDs too (for deeper hierarchies)
                    if id_match:
                        child_ids.add(id_match.group(1))
                    out.write(line if line.endswith("\n") else line + "\n")

    return output_path


def evaluate_chr4(
    model: Any,
    device: torch.device,
    *,
    n_samples: int = 500,
    batch_size: int = 4,
    max_gen_len: int = 2048,
    work_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run inference on `n_samples` random Chr4 gene regions, gffcompare, return metrics.

    Args:
        model: A transgenicForConditionalGeneration already on `device`.
        device: torch.device for inference.
        n_samples: Number of random gene regions to evaluate (default 500).
        batch_size: Inference batch size.
        max_gen_len: Maximum decoder output tokens.
        work_dir: Directory for temporary files. If None, uses a temp dir.
        seed: Random seed for reproducible sample selection.
        verbose: Print progress to stderr.

    Returns:
        Dict with gffcompare metrics + isoform count + gene count.
    """
    _set_seed(seed)

    cleanup_work_dir = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="transgenic_eval_chr4_")
    os.makedirs(work_dir, exist_ok=True)

    db_path = os.path.join(work_dir, "eval_chr4.db")
    pred_gff_path = os.path.join(work_dir, "eval_chr4_pred.gff")
    compare_prefix = os.path.join(work_dir, "eval_chr4_cmp")

    # ── 1. Build evaluation dataset ─────────────────────────────────────
    if not os.path.isfile(db_path):
        genome2GSFDataset(CHR4_FASTA, CHR4_BED, db_path, anoType="bed", mode="predict")

    ds = isoformDataHyena(db_path, mode="inference")

    # Select n_samples random indices (reproducible via seed)
    total = len(ds)
    n_samples = min(n_samples, total)
    indices = random.sample(range(total), n_samples)
    subset = Subset(ds, indices)

    dl = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=hyena_collate_fn,
        persistent_workers=True,
    )

    # ── 2. Run inference ────────────────────────────────────────────────
    gff_tokenizer = GFFTokenizer()
    model.eval()
    genes_predicted = 0
    genes_parsed = 0
    evaluated_gene_names = set()  # Track which genes we actually evaluated

    if verbose:
        from tqdm import tqdm
        dl_iter = tqdm(dl, desc="Chr4 eval", leave=False, file=sys.stderr)
    else:
        dl_iter = dl

    with open(pred_gff_path, "w", encoding="utf-8") as out_f:
        for batch in dl_iter:
            ii = batch[0].to(device, non_blocking=True)
            am = batch[1].to(device, non_blocking=True)

            output_tokens = _generate_batch(model, ii, am, max_length=max_gen_len)

            decoded_batch = gff_tokenizer.batch_decode(
                output_tokens.detach().cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for idx, pred_raw in enumerate(decoded_batch):
                gene_name = batch[3][idx]  # gene model name (e.g. AT4G00020)
                evaluated_gene_names.add(gene_name)
                pred = (
                    pred_raw.replace("|</s>", "")
                    .replace("</s>", "")
                    .replace("<s>", "")
                )
                gff_lines = gffString2GFF3(
                    pred,
                    batch[4][idx],   # chr
                    batch[5][idx],   # region_start
                    f"GM={gene_name}",  # gene model name
                )
                is_valid_parse = not (len(gff_lines) == 1 and gff_lines[0] == "")
                if is_valid_parse:
                    genes_parsed += 1
                    for line in gff_lines:
                        out_f.write(line + "\n")
                genes_predicted += 1

            # Free GPU memory between batches
            del ii, am, output_tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── 3. Build subset reference & run gffcompare ──────────────────────
    # Create a reference GFF3 containing ONLY the genes we evaluated, so
    # that sensitivity/precision are measured against the correct denominator.
    metrics: Dict[str, float] = {
        "genes_predicted": genes_predicted,
        "genes_parsed": genes_parsed,
        "parse_success_rate": (genes_parsed / max(1, genes_predicted)) * 100.0,
    }
    subset_ref_path = os.path.join(work_dir, "eval_chr4_ref_subset.gff3")

    if shutil.which("gffcompare") is not None:
        try:
            _build_subset_ref_gff(CHR4_REF, evaluated_gene_names, subset_ref_path)
            if verbose:
                # Count lines in subset ref for verification
                with open(subset_ref_path) as _f:
                    ref_lines = sum(1 for _ in _f)
                print(f"  Subset reference: {len(evaluated_gene_names)} genes, {ref_lines} GFF3 lines", file=sys.stderr)

            subprocess.run(
                ["gffcompare", "-r", subset_ref_path, "-o", compare_prefix, pred_gff_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            stats_path = compare_prefix + ".stats"
            parsed = _parse_gffcompare_stats(stats_path)
            metrics.update(parsed)
        except subprocess.CalledProcessError as e:
            print(f"Warning: gffcompare failed: {e.stderr}", file=sys.stderr)
    else:
        print("Warning: gffcompare not found in PATH, skipping comparison", file=sys.stderr)

    # ── 4. Count isoforms via gffread ───────────────────────────────────
    isoforms = _count_isoforms_gffread(pred_gff_path)
    if isoforms >= 0:
        metrics["isoforms_predicted"] = isoforms

    # ── 5. Clean up temporary files ─────────────────────────────────────
    if cleanup_work_dir:
        import shutil as _shutil
        try:
            _shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    if verbose:
        print(f"\n[Chr4 eval] genes={genes_predicted} parsed={genes_parsed} parse_success_rate={metrics['parse_success_rate']:.1f}", file=sys.stderr, end="")
        if isoforms >= 0:
            print(f" isoforms={isoforms}", file=sys.stderr, end="")
        for k in ("transcript_sensitivity", "transcript_precision",
                   "exon_sensitivity", "exon_precision",
                   "base_sensitivity", "base_precision",
                   "transcripts_per_locus"):
            if k in metrics:
                print(f" {k}={metrics[k]:.1f}", file=sys.stderr, end="")
        print("", file=sys.stderr)

    return metrics


# ── CLI entrypoint ──────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TransGenic on A. thaliana Chr4")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of random gene regions to evaluate (default: 500)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=2048)
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Directory for temp files (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transgenicForConditionalGeneration.from_pretrained(
        args.checkpoint_dir, local_files_only=True,
    )
    model = cast(Any, model)
    model.to(device)

    metrics = evaluate_chr4(
        model, device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_gen_len=args.max_gen_len,
        work_dir=args.work_dir,
        seed=args.seed,
    )

    print("\n=== Chr4 Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()