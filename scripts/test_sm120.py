import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("arch list:", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
x = torch.randn(1024,1024, device="cuda")
print("ok, mean:", x.mean().item())
