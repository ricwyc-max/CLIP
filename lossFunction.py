__author__ = 'Eric'

# ============================================================================
#
#                           损失函数模块
#
# 本文件包含 CLIP2GAN 项目中用到的所有损失函数：
#   1. L_rec   — 重建损失（衡量生成图片与目标图片的像素级差异）
#   2. L_D     — 判别器损失（原始 GAN 损失 + 梯度惩罚）
#   3. L_G     — 生成器对抗损失（让生成图片骗过判别器）
#   4. L_lpips — 感知损失（让生成图片关注细节）
#   5. L_div   — 多样性损失（鼓励不同特征产生不同图像）
#
# 数学公式：
#   L_rec   = ‖x - x'‖₂²                                             （L2 范数平方）
#   L_D     = -{E[log D(x)] + E[log(1 - D(x'))]}                    （原始 GAN 判别器损失）
#           + λ × E[(||∇D(x')||₂ - 1)²]                             （梯度惩罚 GP）
#   L_G     = E[log(1 - D(x'))]                                      （生成器损失）
#   L_lpips = ∑_l (1/H_l W_l) ∑_{h,w} ‖w_l ⊙ (y_l - y'_l)‖₂²   （LPIPS 感知损失）
#
# 训练流程：
#   1. 训练判别器 D: 最小化 L_D（给真实图高分，给假图低分）
#   2. 训练生成器 G: 最小化 L_G（让判别器给假图高分）
#   3. 训练生成器 G: 最小化 L_rec（让生成图逼近真实图）
#
# ============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#
#                           1. 重建损失 L_rec
#
# ============================================================================

def L_rec(x, x_recon):
    """
    重建损失（Reconstruction Loss, L2 范数平方）

    公式: L_rec = ‖x - x'‖₂²

    每个样本计算所有像素差值的 L2 范数平方，再对 batch 取均值。

    参数:
        x:       目标图片张量, shape (B, C, H, W)，值域 [-1, 1]
        x_recon: 生成图片张量, shape (B, C, H, W)，值域 [-1, 1]

    返回:
        loss_rec: 标量张量（可反向传播）
    """
    loss_rec = (x - x_recon).pow(2).flatten(1).sum(dim=1).mean()
    return loss_rec


# ============================================================================
#
#                           2. 判别器损失 L_D
#
# ============================================================================

def L_D(D, real_imgs, fake_imgs, lambda_gp=10.0):
    """
    判别器损失（Discriminator Loss, 原始 GAN + 梯度惩罚）

    公式:
        L_D = -{E[log D(x)] + E[log(1 - D(x'))]} + λ × E[(||∇D(x')||₂ - 1)²]

    各项含义:
        - E[log D(x)]       : 真实图片被判为真的概率，越大越好（loss 中取负）
        - E[log(1 - D(x'))] : 假图被判为假的概率，越大越好（loss 中取负）
        - 梯度惩罚           : 约束判别器梯度范数接近 1，保持训练稳定

    参数:
        D:         判别器模型，输出经 sigmoid，值域 (0, 1]
        real_imgs: 真实图片, shape (B, 3, H, W), 值域 [-1, 1]
        fake_imgs: 生成图片, shape (B, 3, H, W), 值域 [-1, 1]
        lambda_gp: 梯度惩罚系数，默认 10.0

    返回:
        loss: 标量张量（可反向传播）
    """
    real_out = D(real_imgs)["out"]
    fake_out = D(fake_imgs)["out"]

    # clamp 防止 log(0) = -inf
    real_out = torch.clamp(real_out, 1e-7, 1.0)
    fake_out = torch.clamp(fake_out, 1e-7, 1.0)

    loss_gan = -(torch.log(real_out).mean() + torch.log(1 - fake_out).mean())

    # 梯度惩罚
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=real_imgs.device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    interp_out = D(interpolated)["out"]

    gradients = torch.autograd.grad(
        outputs=interp_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interp_out),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    loss_gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    loss = loss_gan + loss_gp
    return loss


# ============================================================================
#
#                           3. 生成器对抗损失 L_G
#
# ============================================================================

def L_G(D, fake_imgs):
    """
    生成器对抗损失（Generator Adversarial Loss）

    公式: L_G = E[log(1 - D(x'))]
    - 当 D(G(z)) 接近 1（判别器被骗），log(1-D) → -∞，loss 小
    - 当 D(G(z)) 接近 0（判别器识别出假图），log(1-D) → 0，loss 大

    注意：不能用 torch.no_grad()，否则内层上下文会覆盖外层 enable_grad()，
    导致 loss_G 没有 grad_fn，梯度无法回传到 Bridge MLP。

    参数:
        D:         判别器模型，输出经 sigmoid，值域 (0, 1]
        fake_imgs: 生成器生成的图片张量, shape (B, 3, H, W)，需保留梯度

    返回:
        loss_G: 标量张量（可反向传播）
    """
    fake_out = D(fake_imgs)["out"]
    fake_out = torch.clamp(fake_out, 1e-7, 1.0)
    loss_G = torch.log(1 - fake_out).mean()
    return loss_G


# ============================================================================
#
#                           4. LPIPS 感知损失 (AlexNet)
#
# ============================================================================

class LPIPS_AlexNet(nn.Module):
    """
    基于 AlexNet 的 LPIPS (Learned Perceptual Image Patch Similarity) 损失

    公式: L_LPIPS = ∑_l (1/H_l W_l) ∑_{h,w} ‖w_l ⊙ (y_l^{hw} - y'_l^{hw})‖₂²

    流程:
        1. 用预训练 AlexNet 提取多层特征
        2. 每层做 unit normalize（沿通道维度减均值除标准差）
        3. 计算差值，乘以可学习的逐通道权重 w_l
        4. 对空间维度求均值，对层求和

    w_l 说明:
        - w_l 是逐通道的缩放权重，shape 为 (C_l,)
        - 初始化为全 1（等权），可在下游任务中选择冻结或微调
        - 如果冻结（train_w=False），等价于 unit-normalized L2 距离
        - 如果微调（train_w=True），让网络学习不同通道的感知重要性
    """

    # AlexNet 中用于提取特征的 ReLU 层索引
    LAYER_IDS = [1, 4, 7, 9, 11]

    def __init__(self, alexnet_weights_path=None, device="cpu", train_w=False):
        """
        参数:
            alexnet_weights_path: AlexNet 预训练权重路径，默认 AlexNet/model/alexnet_imagenet1k.pth
            device:               设备
            train_w:              w_l 是否可训练（默认 False，冻结）
        """
        super().__init__()

        if alexnet_weights_path is None:
            alexnet_weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "AlexNet", "model", "alexnet_imagenet1k.pth"
            )

        # 加载预训练 AlexNet
        from torchvision.models import alexnet
        net = alexnet()
        state_dict = torch.load(alexnet_weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()

        # 冻结 AlexNet 所有参数
        for p in net.parameters():
            p.requires_grad = False

        self.features = net.features

        # 各层通道数: [64, 192, 384, 256, 256]
        channels = [64, 192, 384, 256, 256]

        # 逐通道缩放权重 w_l，每层一个 (C_l,) 参数
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.ones(c), requires_grad=train_w) for c in channels]
        )

        # ImageNet 归一化参数（AlexNet 输入需要归一化）
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.to(device)

    def _normalize_input(self, x):
        """将 [-1,1] 图片归一化为 ImageNet 标准输入"""
        x = (x + 1) / 2  # [-1,1] → [0,1]
        x = (x - self.mean) / self.std
        return x

    def _extract_features(self, x):
        """逐层提取 AlexNet 特征"""
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.LAYER_IDS:
                feats.append(x)
        return feats

    def forward(self, x, x_recon):
        """
        计算 LPIPS 损失

        参数:
            x:       目标图片, shape (B, 3, H, W), 值域 [-1, 1]
            x_recon: 生成图片, shape (B, 3, H, W), 值域 [-1, 1]

        返回:
            loss: 标量张量（可反向传播）
        """
        # 归一化到 ImageNet 输入范围
        x_norm = self._normalize_input(x)
        x_recon_norm = self._normalize_input(x_recon)

        # 提取多层特征
        feats_x = self._extract_features(x_norm)
        feats_xr = self._extract_features(x_recon_norm)

        # 逐层计算 LPIPS
        loss = 0.0
        for i, (fx, fxr, w) in enumerate(zip(feats_x, feats_xr, self.weights)):
            # Unit normalize: 沿通道维度做 L2 归一化
            fx_norm = F.normalize(fx, p=2, dim=1)
            fxr_norm = F.normalize(fxr, p=2, dim=1)

            # 通道加权差值: w_l ⊙ (y_l - y'_l), w 广播到 (B, C, H, W)
            diff = (fx_norm - fxr_norm) * w.view(1, -1, 1, 1)

            # L2 范数平方，对通道求和，对空间求均值
            # ‖w ⊙ diff‖₂² = sum_c (w_c * diff_c)²
            layer_loss = (diff ** 2).sum(dim=1).mean(dim=[1, 2])  # (B,)

            loss = loss + layer_loss.mean()

        return loss


# ============================================================================
#
#                           5. 多样性损失 L_div
#
# ============================================================================

def L_div(feat_orig, feat_noisy, img_orig, img_noisy):
    """
    多样性损失（Diversity Loss）

    公式: L_div = d_f(f_img, f_img1) / d_I(x', x'_1)

    鼓励不同特征输入产生不同的生成图像。
    当两个特征的扰动很大但生成图像几乎相同时，分母小、loss 大，惩罚模式坍缩。

    参数:
        feat_orig:  原始 CLIP 图像特征, shape (B, D)
        feat_noisy: 加噪后的 CLIP 图像特征, shape (B, D)
        img_orig:   从 feat_orig 生成的图片, shape (B, 3, H, W), 值域 [-1, 1]
        img_noisy:  从 feat_noisy 生成的图片, shape (B, 3, H, W), 值域 [-1, 1]

    返回:
        loss_div: 标量张量（可反向传播）
    """
    # 特征空间 L1 距离: d_f = mean(|f - f1|)
    d_f = torch.abs(feat_orig - feat_noisy).mean()

    # 图像空间 L1 距离: d_I = mean(|x - x1|)
    d_I = torch.abs(img_orig - img_noisy).mean()

    # 防止分母为 0
    d_I = torch.clamp(d_I, min=1e-7)

    loss_div = d_f / d_I
    return loss_div
