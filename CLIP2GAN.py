"""
CLIP2GAN 封装类

将 CLIP（文本/图像编码）和 MobileStyleGAN（图像生成）封装为一个统一接口。
两个模型都是冻结的（不训练），后续可加 Bridge MLP 连接两个 512 维空间。

整体架构：
    文本 "a cat" → CLIP Text Encoder → text_embedding (512维)
                                              ↓
                                      [Bridge MLP (待实现)]
                                              ↓
                                        style (512维)
                                              ↓
                                      MobileStyleGAN → 生成图片 (3, 1024, 1024)

维度对齐：
    CLIP ViT-B-32 输出: 512 维
    MobileStyleGAN style_dim: 512 维
    两者恰好一致，可直接连接
"""

import os
import sys

# ========== 路径设置 ==========
# 获取当前文件所在目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# MobileStyleGAN 的代码在 StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/ 下
# 需要把它加到 sys.path 才能 import core.models
_MOBILE_STYLEGAN_ROOT = os.path.join(
    _CURRENT_DIR, "StyleGAN", "mobileStyleGAN", "MobileStyleGAN.pytorch"
)
if _MOBILE_STYLEGAN_ROOT not in sys.path:
    sys.path.insert(0, _MOBILE_STYLEGAN_ROOT)

# CLIP 的模型缓存目录
_CLIP_MODEL_DIR = os.path.join(_CURRENT_DIR, "CLIP", "mobileCLIP", "model")

import torch
import torch.nn as nn
import numpy as np
import open_clip
from PIL import Image

# MobileStyleGAN 模块
from core.models.mapping_network import MappingNetwork
from core.models.mobile_synthesis_network import MobileSynthesisNetwork


class CLIP2GAN:
    """
    CLIP2GAN 封装类

    封装 CLIP 和 MobileStyleGAN 两个冻结模型，提供统一的推理接口。

    使用示例:
        model = CLIP2GAN(device="cpu")

        # 文本编码
        text_feat = model.encode_text(["a cat", "a dog"])  # (2, 512)

        # 随机生成图片
        z = torch.randn(1, 512)
        img = model.z_to_image(z)  # (1, 3, 1024, 1024)

        # 评估生成图片与文本的相似度
        img_feat = model.encode_image(model.preprocess_img(img[0]))
        sim = model.similarity(text_feat, img_feat)
    """

    def __init__(self, device="cpu", stylegan_ckpt=None):
        """
        初始化 CLIP2GAN，加载两个冻结模型。

        参数:
            device: "cpu" 或 "cuda"
            stylegan_ckpt: MobileStyleGAN 权重路径，默认为 model/mobilestylegan_ffhq_v2.ckpt
        """
        self.device = device

        # ========== 1. 加载 CLIP ==========
        print("[CLIP2GAN] 加载 CLIP 模型...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model.eval()
        # 冻结 CLIP 所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("[CLIP2GAN] CLIP 加载完成 (ViT-B-32, 512维)")

        # ========== 2. 加载 MobileStyleGAN ==========
        print("[CLIP2GAN] 加载 MobileStyleGAN 模型...")
        if stylegan_ckpt is None:
            stylegan_ckpt = os.path.join(_MOBILE_STYLEGAN_ROOT, "model", "mobilestylegan_ffhq_v2.ckpt")

        ckpt = torch.load(stylegan_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        # 构建并加载 MappingNetwork (z → style)
        self.mapping_net = MappingNetwork(style_dim=512, n_layers=8)
        mnet_sd = {k.replace("mapping_net.", ""): v for k, v in sd.items() if k.startswith("mapping_net.")}
        self.mapping_net.load_state_dict(mnet_sd)
        self.mapping_net.eval()

        # 构建并加载 MobileSynthesisNetwork (style → img)
        self.synthesis_net = MobileSynthesisNetwork(
            style_dim=512, channels=[512, 512, 512, 512, 512, 256, 128, 64]
        )
        snet_sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}
        self.synthesis_net.load_state_dict(snet_sd)
        self.synthesis_net.eval()

        # 风格均值（截断用）
        self.style_mean = sd["style_mean"]

        # 冻结 MobileStyleGAN 所有参数
        for param in self.mapping_net.parameters():
            param.requires_grad = False
        for param in self.synthesis_net.parameters():
            param.requires_grad = False
        print("[CLIP2GAN] MobileStyleGAN 加载完成 (style_dim=512, output=1024x1024)")

        # 移到指定设备
        self.to(device)
        print(f"[CLIP2GAN] 初始化完成，设备: {device}")

    def to(self, device):
        """移动所有模型到指定设备"""
        self.device = device
        self.clip_model.to(device)
        self.mapping_net.to(device)
        self.synthesis_net.to(device)
        self.style_mean = self.style_mean.to(device)

    # ================================================================
    # CLIP 相关方法
    # ================================================================

    @torch.no_grad()
    def encode_text(self, texts):
        """
        用 CLIP 编码文本，返回归一化的文本特征向量。

        参数:
            texts: list[str]，如 ["a cat", "a dog"]

        返回:
            Tensor, shape (N, 512)，L2 归一化后的文本特征
        """
        tokens = self.clip_tokenizer(texts).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        # L2 归一化
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def encode_image(self, images):
        """
        用 CLIP 编码图像，返回归一化的图像特征向量。

        参数:
            images: Tensor, shape (N, 3, 224, 224)，已经过 CLIP 预处理的图像

        返回:
            Tensor, shape (N, 512)，L2 归一化后的图像特征
        """
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        # L2 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def preprocess_img(self, img_tensor):
        """
        将 MobileStyleGAN 生成的图片 (3, 1024, 1024) 预处理为 CLIP 输入格式 (1, 3, 224, 224)。

        参数:
            img_tensor: shape (3, 1024, 1024)，值域 [-1, 1]

        返回:
            Tensor, shape (1, 3, 224, 224)，CLIP 预处理后的图像
        """
        # 从 [-1,1] 转到 [0,1]
        img = (img_tensor.clamp(-1, 1) + 1) / 2
        # 转成 PIL Image
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        # 用 CLIP 的预处理管道
        return self.clip_preprocess(pil_img).unsqueeze(0)

    @torch.no_grad()
    def similarity(self, text_features, image_features):
        """
        计算文本特征和图像特征的余弦相似度。

        参数:
            text_features: shape (N, 512)，文本特征（已归一化）
            image_features: shape (M, 512)，图像特征（已归一化）

        返回:
            Tensor, shape (N, M)，余弦相似度矩阵
        """
        return text_features @ image_features.T

    # ================================================================
    # MobileStyleGAN 相关方法
    # ================================================================

    @torch.no_grad()
    def z_to_image(self, z, truncated=True, truncation_psi=0.5):
        """
        从随机噪声 z 生成图片。

        流程: z → MappingNetwork → style → (截断) → MobileSynthesisNetwork → 图片

        参数:
            z: Tensor, shape (B, 512)，随机噪声（标准正态分布）
            truncated: 是否使用截断模式，默认 True
            truncation_psi: 截断强度，默认 0.5
                - 1.0: 无截断，多样性最高
                - 0.5: 适中（默认）
                - 0.0: 生成均值脸

        返回:
            Tensor, shape (B, 3, 1024, 1024)，值域 [-1, 1]
        """
        z = z.to(self.device)

        # z → style (W 空间)
        style = self.mapping_net(z)

        # 截断：把 style 拉向均值
        if truncated:
            style = self.style_mean + truncation_psi * (style - self.style_mean)

        # style → 图片
        img = self.synthesis_net(style)["img"]
        return img

    @torch.no_grad()
    def get_style_from_z(self, z):
        """
        获取 z 对应的 W 空间风格向量（不生成图片）。

        参数:
            z: Tensor, shape (B, 512)

        返回:
            Tensor, shape (B, 512)，W 空间风格向量
        """
        return self.mapping_net(z.to(self.device))

    @torch.no_grad()
    def style_to_image(self, style):
        """
        从 W 空间风格向量直接生成图片（跳过 MappingNetwork）。

        参数:
            style: Tensor, shape (B, 512) 或 (B, 23, 512)

        返回:
            Tensor, shape (B, 3, 1024, 1024)，值域 [-1, 1]
        """
        return self.synthesis_net(style.to(self.device))["img"]

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def tensor_to_img(img_tensor, rgb2bgr=True):
        """
        将图片张量转为 numpy 数组。

        参数:
            img_tensor: shape (3, H, W) 或 (1, 3, H, W)，值域 [-1, 1]
            rgb2bgr: 是否转为 BGR 格式（OpenCV 用），默认 True

        返回:
            numpy array, shape (H, W, 3), uint8, 值域 [0, 255]
        """
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        t = img_tensor.clamp(-1, 1)
        t = (t + 1) / 2  # [-1,1] → [0,1]
        img = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if rgb2bgr:
            img = img[:, :, ::-1]
        return img

    def show(self, img_tensor):
        """
        用 matplotlib 显示图片。

        参数:
            img_tensor: shape (3, H, W) 或 (1, 3, H, W)
        """
        from matplotlib import pyplot as plt
        img_np = self.tensor_to_img(img_tensor, rgb2bgr=False)
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()

    def save(self, img_tensor, path="output.png"):
        """
        保存图片。

        参数:
            img_tensor: shape (3, H, W) 或 (1, 3, H, W)
            path: 保存路径
        """
        import cv2
        img_np = self.tensor_to_img(img_tensor, rgb2bgr=True)
        cv2.imwrite(path, img_np)
        print(f"已保存: {path}")


# ================================================================
# 使用示例
# ================================================================
if __name__ == "__main__":
    # 初始化
    model = CLIP2GAN(device="cuda")

    # --- 1. 文本编码 ---
    texts = ["a cat", "a dog", "a bird"]
    text_feat = model.encode_text(texts)
    print(f"文本特征 shape: {text_feat.shape}")  # (3, 512)

    # --- 2. 随机生成图片 ---
    z = torch.randn(1, 512)
    img = model.z_to_image(z, truncated=True)
    print(f"生成图片 shape: {img.shape}")  # (1, 3, 1024, 1024)

    # --- 3. 评估生成图片与文本的相似度 ---
    img_feat = model.encode_image(model.preprocess_img(img[0]))
    sim = model.similarity(text_feat, img_feat)
    print(f"相似度: {sim}")  # (3, 1)

    # --- 4. 保存图片 ---
    model.save(img[0], "test_clip2gan.png")
