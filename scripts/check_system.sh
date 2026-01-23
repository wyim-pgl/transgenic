#!/usr/bin/env bash
#
# Check system architecture and GPU availability for TransGenic installation
#

echo "============================================"
echo "TransGenic System Check"
echo "============================================"
echo

# Architecture
ARCH=$(uname -m)
echo "[Architecture]"
echo "  uname -m: $ARCH"

case "$ARCH" in
    x86_64)
        echo "  Type: x86_64 (Intel/AMD 64-bit)"
        ;;
    aarch64|arm64)
        echo "  Type: ARM 64-bit (aarch64)"
        ;;
    *)
        echo "  Type: Unknown ($ARCH)"
        ;;
esac
echo

# OS
echo "[Operating System]"
echo "  OS: $(uname -s)"
if [ -f /etc/os-release ]; then
    echo "  Distro: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
fi
echo

# GPU Check
echo "[GPU Check]"
if command -v nvidia-smi &> /dev/null; then
    echo "  nvidia-smi: Found"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' || echo "N/A")

    if [ -n "$GPU_NAME" ]; then
        echo "  GPU: $GPU_NAME"
        echo "  Driver: $GPU_DRIVER"
        echo "  CUDA: $CUDA_VERSION"
    else
        echo "  GPU: No GPU detected"
    fi
else
    echo "  nvidia-smi: Not found"
    echo "  GPU: None (CPU only)"
fi
echo

# Recommendation
echo "[Recommended Environment]"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "blackwell\|grace"; then
        echo "  → GB10 ARM CPU (NVIDIA Grace Blackwell)"
        echo "  → Use: environment.gb10.base.yml + scripts/install_ml_stack_gb10.sh"
    else
        echo "  → ARM architecture detected"
        echo "  → Use: environment.gb10.base.yml + scripts/install_ml_stack_gb10.sh"
    fi
elif [ "$ARCH" = "x86_64" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q .; then
        echo "  → x86 with CUDA GPU"
        echo "  → Use: environment.yml"
    else
        echo "  → x86 CPU Only (No GPU)"
        echo "  → Use: environment.cpu.yml"
    fi
else
    echo "  → Unknown architecture"
    echo "  → Try: environment.cpu.yml"
fi
echo

echo "============================================"
