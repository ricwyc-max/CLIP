__author__ = 'Eric'


import os
import torch
from torchvision.models import alexnet, AlexNet_Weights

# 保存到 AlexNet/model 目录（与 CLIP 保存方式一致）
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(save_dir, exist_ok=True)

print("[GetAlexNet] 正在从 torchvision 下载 AlexNet 预训练模型 (ImageNet-1K)...")
weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights)

save_path = os.path.join(save_dir, "alexnet_imagenet1k.pth")
print(f"[GetAlexNet] 正在保存模型权重到 {save_path} ...")
torch.save(model.state_dict(), save_path)
print("[GetAlexNet] AlexNet 模型保存完成！")

