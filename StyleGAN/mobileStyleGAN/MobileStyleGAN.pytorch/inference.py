import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import numpy as np
from core.models.mapping_network import MappingNetwork
from core.models.mobile_synthesis_network import MobileSynthesisNetwork
from core.utils import tensor_to_img

# 加载 checkpoint
ckpt = torch.load("model/mobilestylegan_ffhq_v2.ckpt", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]

# 构建 MappingNetwork
mnet_params = {
    "style_dim": 512,
    "n_layers": 8,
}
mapping_net = MappingNetwork(**mnet_params)
# 提取 mapping_net 权重，去掉前缀
mnet_sd = {k.replace("mapping_net.", ""): v for k, v in sd.items() if k.startswith("mapping_net.")}
mapping_net.load_state_dict(mnet_sd)

# 构建 MobileSynthesisNetwork
snet_params = {
    "style_dim": 512,
    "channels": [512, 512, 512, 512, 256, 128, 64],
}
student = MobileSynthesisNetwork(**snet_params)
# 提取 student 权重，去掉前缀
student_sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}
student.load_state_dict(student_sd)

# 设置为评估模式
mapping_net.eval()
student.eval()

# 提取 style_mean
style_mean = sd["style_mean"]

@torch.no_grad()
def generate(batch_size=1, truncated=False, device="cpu"):
    """
    输入: 随机噪声 z, shape (batch_size, 512)
    输出: 图片, shape (batch_size, 3, H, W), 值域 [-1, 1]
    """
    mapping_net.to(device)
    student.to(device)

    z = torch.randn(batch_size, 512).to(device)
    style = mapping_net(z)

    if truncated:
        style_mean_dev = style_mean.to(device)
        style = style_mean_dev + 0.5 * (style - style_mean_dev)

    img = student(style)["img"]
    return img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=10, help="生成图片数量")
    parser.add_argument("--truncated", action='store_true', help="使用截断模式（更稳定但多样性降低）")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    parser.add_argument("--output-path", type=str, default="./output", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print(f"生成 {args.n_images} 张图片...")
    for i in range(args.n_images):
        img = generate(batch_size=1, truncated=args.truncated, device=args.device)
        img_np = tensor_to_img(img[0].cpu())
        cv2.imwrite(os.path.join(args.output_path, f"{i}.png"), img_np)
        print(f"  保存: {i}.png")

    print(f"完成！图片保存在 {args.output_path}")
