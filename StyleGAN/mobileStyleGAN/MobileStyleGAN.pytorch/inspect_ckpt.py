import torch
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ckpt = torch.load("model/mobilestylegan_ffhq_v2.ckpt", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]

# 查看 student 的 channels 配置
print("=== student 关键层 shape ===")
for k, v in sd.items():
    if k.startswith("student.") and "weight" in k and "modulation" not in k and "net" not in k:
        print(f"  {k}: {v.shape}")
