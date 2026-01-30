#!/bin/bash
# Install script for transgenic on NVIDIA Blackwell (GB10) with CUDA 13.0
# Creates conda env then installs PyTorch nightly separately

set -e

ENV_NAME="${1:-transgenic-gb10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating conda environment: $ENV_NAME ==="
conda env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.gb10.yml" || \
    conda env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.gb10.yml"

echo ""
echo "=== Installing PyTorch nightly (cu128) for Blackwell ==="
conda run -n "$ENV_NAME" pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo "=== Installing additional pip packages from PyPI ==="
conda run -n "$ENV_NAME" pip install torcheval trl tyro lightning-utilities torchmetrics

echo ""
echo "=== Verifying CUDA / Blackwell support ==="
conda run -n "$ENV_NAME" python - <<'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"CUDA version:    {torch.version.cuda}")

if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    name = torch.cuda.get_device_name(dev)
    print(f"Device:          {name}")
    print(f"Compute cap:     sm_{cap[0]}{cap[1]}")
    
    # Quick matmul test
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = a @ b
    torch.cuda.synchronize()
    print("Matmul test:     ✅ passed")
else:
    print("⚠️  CUDA not available - check driver/PyTorch compatibility")
EOF

echo ""
echo "=== Done ==="
echo "Activate with: conda activate $ENV_NAME"
