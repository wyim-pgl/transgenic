#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# PyTorch CUDA wheels for cu129 (GB10 working combo)
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.10.0+cu129 \
  torchvision==0.25.0+cu129 \
  torchaudio==2.10.0+cu129

# Your HF stack
python -m pip install \
  transformers==4.44.2 \
  datasets==3.0.0 \
  accelerate==0.34.2 \
  peft==0.12.0 \
  einops==0.8.0 \
  trl==0.11.4 \
  wandb==0.17.3 \
  sentencepiece

# Install transgenic module
python -m pip install .

# quick verification
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn((2048,2048), device="cuda", dtype=torch.float16)
    y = torch.randn((2048,2048), device="cuda", dtype=torch.float16)
    z = (x @ y).mean()
    torch.cuda.synchronize()
    print("matmul mean:", float(z))
PY
