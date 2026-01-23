import os, subprocess, sys, platform
import torch

def sh(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[failed] {cmd}\n{e}"

print("python:", sys.version.replace("\n"," "))
print("arch:", platform.machine())
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(dev)
    print("device:", prop.name)
    print("capability:", (prop.major, prop.minor))

    # 1) 간단 matmul (GPU kernel)
    a = torch.randn((4096, 4096), device=dev, dtype=torch.float16)
    b = torch.randn((4096, 4096), device=dev, dtype=torch.float16)
    c = (a @ b).mean()
    torch.cuda.synchronize()
    print("matmul ok, mean:", float(c))

    # 2) conv2d (GPU kernel)
    x = torch.randn((16, 3, 224, 224), device=dev, dtype=torch.float16)
    w = torch.randn((64, 3, 7, 7), device=dev, dtype=torch.float16)
    y = torch.nn.functional.conv2d(x, w, stride=2, padding=3).mean()
    torch.cuda.synchronize()
    print("conv2d ok, mean:", float(y))

    # 3) 시스템 측 확인
    print("nvidia-smi:\n", sh("nvidia-smi -L"))
else:
    print("CUDA not available. Likely CPU-only torch installed from PyPI.")

