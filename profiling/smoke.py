import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
print("smoke ok")
