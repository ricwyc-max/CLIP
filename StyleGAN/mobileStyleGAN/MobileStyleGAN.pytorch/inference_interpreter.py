"""
MobileStyleGAN 推理脚本

整体流程：
    随机噪声 z (batch, 512)
        ↓  MappingNetwork（8层全连接，将噪声映射为风格向量）
    风格向量 style (batch, 512)
        ↓  MobileSynthesisNetwork（轻量级合成网络，用逆小波变换生成图片）
    生成图片 img (batch, 3, 1024, 1024)，值域 [-1, 1]

checkpoint 文件结构（mobilestylegan_ffhq_v2.ckpt）：
    ├── mapping_net   → 映射网络权重（z → style）
    ├── student       → 学生合成网络权重（style → 图片）
    ├── synthesis_net → 教师合成网络权重（训练用，推理不需要）
    ├── loss          → 损失网络权重（训练用，推理不需要）
    ├── inception     → 评估网络权重（训练用，推理不需要）
    └── style_mean    → 风格均值（截断模式用，提高生成稳定性）
"""

import os
# 设置工作目录为脚本所在目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import numpy as np
from core.models.mapping_network import MappingNetwork
from core.models.mobile_synthesis_network import MobileSynthesisNetwork
from core.utils import tensor_to_img

# ============================================================
# 第一步：加载 checkpoint 文件
# ============================================================
# ckpt 是一个 dict，包含：
#   - "state_dict": 所有网络的权重参数
#   - "epoch": 训练轮次
#   - "global_step": 全局步数
#   - "optimizer_states": 优化器状态（推理不需要）
#   - 等等
# weights_only=False：因为 checkpoint 里包含非张量对象（如 PyTorch Lightning 的回调），
#   PyTorch 2.6+ 默认 weights_only=True 会报错，需要设为 False
# map_location="cpu"：把权重加载到 CPU，不管原来是在哪个设备上保存的
ckpt = torch.load("model/mobilestylegan_ffhq_v2.ckpt", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]  # 提取所有权重，是一个 OrderedDict

# ============================================================
# 第二步：构建并加载 MappingNetwork（映射网络）
# ============================================================
# MappingNetwork 的作用：把随机噪声 z 映射为风格向量 style
#   输入: z, shape (batch, 512)，随机高斯噪声
#   输出: style, shape (batch, 512)，有意义的风格表示
# 结构：PixelNorm + 8 个 EqualLinear 全连接层
#   - style_dim=512: 风格向量维度
#   - n_layers=8: 全连接层数量
mapping_net = MappingNetwork(style_dim=512, n_layers=8)

# checkpoint 里的 key 带有前缀 "mapping_net."，比如 "mapping_net.layers.1.weight"
# 但 MappingNetwork 自己的 state_dict 的 key 没有前缀，比如 "layers.1.weight"
# 所以需要去掉前缀才能 load_state_dict 匹配上
# 做法：遍历所有 key，只保留以 "mapping_net." 开头的，然后 replace 掉前缀
mnet_sd = {k.replace("mapping_net.", ""): v for k, v in sd.items() if k.startswith("mapping_net.")}
mapping_net.load_state_dict(mnet_sd)

# eval() 模式：关闭 Dropout 和 BatchNorm 的训练行为，推理时必须调用
mapping_net.eval()

# ============================================================
# 第三步：构建并加载 MobileSynthesisNetwork（学生合成网络）
# ============================================================
# MobileSynthesisNetwork 的作用：根据风格向量 style 生成图片
#   输入: style, shape (batch, 512) 或 (batch, wsize, 512)
#   输出: dict，包含 "img" (batch, 3, 1024, 1024) 等
# 结构：ConstantInput + StyledConv2d + 多个 MobileSynthesisBlock + IDWT（逆小波变换）
#   - style_dim=512: 风格向量维度，和 MappingNetwork 对应
#   - channels=[512, 512, 512, 512, 512, 256, 128, 64]:
#       每一层的通道数，8 个值对应 7 个层（第一个是初始输入通道）
#       通道数逐渐减少：512 → 512 → 512 → 512 → 512 → 256 → 128 → 64
#       对应分辨率逐渐增加：4x4 → 8x8 → ... → 1024x1024
student = MobileSynthesisNetwork(style_dim=512, channels=[512, 512, 512, 512, 512, 256, 128, 64])

# 同样去掉前缀 "student."，只提取学生网络的权重
# 注意：checkpoint 里还有 "synthesis_net."（教师网络），我们不需要它
student_sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}
student.load_state_dict(student_sd)
student.eval()

# ============================================================
# 第四步：提取 style_mean（风格均值）
# ============================================================
# style_mean 是训练时计算的所有风格向量的均值，shape (1, 512)
# 用于"截断"（truncation）模式：
#   style = style_mean + 0.5 * (style - style_mean)
#   效果：让生成的图片更稳定、质量更高，但多样性降低
#   原理：把 style 限制在均值附近的一个小范围内，避免极端值
style_mean = sd["style_mean"]

# ============================================================
# 第五步：定义生成函数
# ============================================================
@torch.no_grad()  # 推理时不需要计算梯度，节省显存和加速
def generate(batch_size=1, truncated=False, device="cpu"):
    """
    生成图片

    参数:
        batch_size: 一次生成几张图片，默认 1
        truncated: 是否使用截断模式，默认 False
            - False: 生成多样性强，但质量可能不稳定
            - True: 生成更稳定，但多样性降低
        device: "cpu" 或 "cuda"（需要有 NVIDIA 显卡）

    返回:
        img: 图片张量, shape (batch_size, 3, 1024, 1024), 值域 [-1, 1]
            - 3 个通道是 RGB
            - 值 -1 代表黑色，1 代表白色
    """
    # 把模型移到指定设备
    mapping_net.to(device)
    student.to(device)

    # 生成随机噪声 z，shape (batch_size, 512)
    # 这是模型的唯一输入，不同的 z 会生成不同的图片
    z = torch.randn(batch_size, 512).to(device)

    # 第一步：z → style
    # mapping_net 把随机噪声映射为有意义的风格向量
    style = mapping_net(z)  # shape: (batch_size, 512)

    # 如果开启截断，把 style 拉向均值，减少极端值
    if truncated:
        style_mean_dev = style_mean.to(device)
        # 公式：style_new = mean + 0.5 * (style - mean)
        # 0.5 是截断强度，越小生成越稳定但越单调
        style = style_mean_dev + 0.5 * (style - style_mean_dev)

    # 第二步：style → img
    # student 根据风格向量生成图片
    # 返回一个 dict，"img" 键对应生成的图片张量
    img = student(style)["img"]  # shape: (batch_size, 3, 1024, 1024)

    return img

# ============================================================
# 第六步：辅助函数（显示和保存）
# ============================================================
def show(img_tensor):
    """
    用 matplotlib 显示图片

    参数:
        img_tensor: 图片张量
            - shape (3, H, W) 单张图片
            - 或 shape (1, 3, H, W) 带 batch 维度
    """
    # 如果有 batch 维度，去掉它
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]

    # tensor_to_img: 把 [-1,1] 的张量转成 [0,255] 的 numpy 数组
    # rgb2bgr=False: matplotlib 需要 RGB 格式（OpenCV 需要 BGR）
    img_np = tensor_to_img(img_tensor.cpu(), rgb2bgr=False)

    from matplotlib import pyplot as plt
    plt.imshow(img_np)
    plt.axis("off")  # 不显示坐标轴
    plt.show()


def save(img_tensor, path="output.png"):
    """
    用 OpenCV 保存图片

    参数:
        img_tensor: 图片张量，shape (3, H, W) 或 (1, 3, H, W)
        path: 保存路径，默认 "output.png"
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]

    # tensor_to_img 默认 rgb2bgr=True，因为 OpenCV 用 BGR 格式
    img_np = tensor_to_img(img_tensor.cpu())
    cv2.imwrite(path, img_np)
    print(f"已保存: {path}")


# ============================================================
# 使用示例
# ============================================================
# 生成 1 张图片
# img = generate()

# 其他用法（取消注释即可）：
# img = generate(batch_size=4)        # 一次生成 4 张
# img = generate(truncated=True)      # 截断模式（更稳定）
img = generate(device="cuda")       # 用 GPU 加速（需要 NVIDIA 显卡）
# show(img)                           # 用 matplotlib 显示
# save(img, "test.png")               # 保存为文件

# 用 OpenCV 显示图片
# img[0] 取第一张图片（去掉 batch 维度）
# tensor_to_img 把张量转成 numpy 数组（BGR 格式，0-255）
img_np = tensor_to_img(img[0])
cv2.imshow('img', img_np)  # 显示窗口
cv2.waitKey(0)             # 等待按键
cv2.destroyAllWindows()    # 关闭所有窗口
cv2.imwrite("./testimg/test5.png",img_np)               # 保存为文件