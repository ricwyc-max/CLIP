__author__ = 'Eric'

# ============================================================================
#
#                           损失函数模块
#
# 本文件包含 CLIP2GAN 项目中用到的所有损失函数：
#   1. L_rec   — 重建损失（衡量生成图片与目标图片的像素级差异）
#   2. L_D     — WGAN-GP 判别器损失（标准 GAN 损失 + 梯度惩罚）
#   3. L_G     — 生成器对抗损失（让生成图片骗过判别器）
#
# 数学公式：
#   L_rec  = MSE(x, x_recon)                         （均方误差）
#   L_D    = E[D(fake)] - E[D(real)]                 （WGAN Wasserstein 损失）
#          + λ × E[(||∇D(x')||₂ - 1)²]              （梯度惩罚 GP）
#   L_G    = -E[D(G(z))]                             （生成器 Wasserstein 损失）
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
    重建损失（Reconstruction Loss, MSE 形式）

    衡量生成图片 x_recon 与目标图片 x 之间的像素级差异。
    使用 MSE（均方误差），每个像素独立贡献梯度，量级稳定。

    公式: L_rec = mean((x - x_recon)²)

    参数:
        x:       目标图片张量, shape (B, C, H, W)，值域 [-1, 1]
        x_recon: 生成图片张量, shape (B, C, H, W)，值域 [-1, 1]

    返回:
        loss_rec: 标量张量（可反向传播）
    """
    # MSE: 每个像素独立贡献梯度，梯度 = 2*(x - x_recon)/N，量级稳定
    loss_rec = F.mse_loss(x, x_recon)
    return loss_rec


# ============================================================================
#
#                           2. 判别器损失 L_D
#
# ============================================================================

def L_D(D, real_imgs, fake_imgs, lambda_gp=10.0):
    """
    WGAN-GP 判别器损失（Discriminator Loss with Gradient Penalty）

    由两部分组成：
    1. Wasserstein 损失：让判别器 D 给真实图片高分，给生成图片低分
    2. 梯度惩罚 (GP)：约束判别器的梯度范数接近 1，防止训练不稳定

    数学公式:
        L_D = E[D(fake)] - E[D(real)] + λ × E[(||∇D(x')||₂ - 1)²]

    各项含义:
        - E[D(real)]  : 真实图片的判别分数，越大越好
        - E[D(fake)]  : 生成图片的判别分数，越小越好
        - 梯度惩罚     : 防止判别器梯度爆炸或消失，保持训练稳定

    参数:
        D:         判别器模型 (nn.Module)
                   - 输入: 图片张量 (B, 3, H, W)
                   - 输出: dict，包含 "out" 键，值为 (B, 1) 的判别分数
                   - 项目中使用 core.models.discriminator.Discriminator

        real_imgs: 真实图片张量, shape (B, 3, H, W)
                   - 必须 requires_grad=True（计算梯度惩罚需要）
                   - 值域: [-1, 1]（StyleGAN 的标准输出范围）

        fake_imgs: 生成器生成的图片张量, shape (B, 3, H, W)
                   - 由生成器 G(z) 产生，z 是随机噪声
                   - 训练判别器时需要 .detach()，不回传梯度到生成器

        lambda_gp: 梯度惩罚系数 λ，默认 10.0
                   - 越大：梯度惩罚越强，判别器越"温和"，训练越稳定但可能收敛慢
                   - 越小：梯度惩罚越弱，判别器可能太强，导致训练不稳定
                   - 常用值: 1, 5, 10（WGAN-GP 原论文推荐 10）

    返回:
        loss: 标量张量（可反向传播）
              - 包含 loss_gan + loss_gp
              - 训练目标：最小化此值

    使用示例:
        # 训练判别器
        z = torch.randn(batch_size, 512)
        fake_imgs = generator(z).detach()  # detach！不回传梯度到生成器
        loss_D = L_D(discriminator, real_imgs, fake_imgs, lambda_gp=10.0)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
    """

    # ========================================================================
    # 第一部分：Wasserstein 损失
    # ========================================================================
    # 公式: E[D(fake)] - E[D(real)]
    # 最小化此值即让 D(real) 大、D(fake) 小
    # D 输出无界（不使用 sigmoid），需要 Discriminator(activate=False)

    real_out = D(real_imgs)["out"]
    fake_out = D(fake_imgs)["out"]

    loss_gan = (fake_out - real_out).mean()

    # ========================================================================
    # 第二部分：梯度惩罚 (Gradient Penalty, GP)
    # ========================================================================
    # 公式: λ × E[(||∇D(x')||₂ - 1)²]
    #
    # 为什么需要梯度惩罚？
    #   - WGAN 要求判别器是 1-Lipschitz 函数（梯度范数 ≤ 1）
    #   - 梯度惩罚通过软约束，让判别器的梯度范数接近 1
    #   - 这样可以防止判别器太强（梯度爆炸）或太弱（梯度消失）
    #   - 使训练更稳定，生成质量更高
    #
    # 具体做法：
    #   1. 在真实图片和生成图片之间做随机插值
    #   2. 计算判别器对插值图片的梯度
    #   3. 梯度范数偏离 1 的程度作为惩罚项

    # --- 步骤 1: 生成插值图片 ---
    # alpha 是 [0, 1] 之间的随机数，shape (B, 1, 1, 1) 以便广播
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=real_imgs.device)

    # 线性插值: x_interp = α × real + (1-α) × fake
    # requires_grad_(True)：告诉 PyTorch 需要对这个张量计算梯度
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    # --- 步骤 2: 计算判别器对插值图片的输出 ---
    interp_out = D(interpolated)["out"]

    # --- 步骤 3: 计算梯度 ∇D(x_interp) ---
    # torch.autograd.grad 计算 interp_out 对 interpolated 的梯度
    gradients = torch.autograd.grad(
        outputs=interp_out,           # 输出：判别器的判别分数
        inputs=interpolated,          # 输入：插值图片（对它求梯度）
        grad_outputs=torch.ones_like(interp_out),  # ∂L/∂out = 1，即直接对 out 求导
        create_graph=True,            # 必须为 True！
                                      # 因为我们需要对"梯度"再求导（二阶梯度）
                                      # 如果 False，梯度惩罚项无法反向传播
        retain_graph=True,            # 保留计算图，因为 loss_gan 也要用
    )[0]  # grad 返回 tuple，取第一个元素

    # --- 步骤 4: 计算梯度的 L2 范数 ---
    # gradients shape: (B, C, H, W) → 展平为 (B, C*H*W)
    gradients = gradients.view(gradients.size(0), -1)

    # 计算每张图片梯度的 L2 范数: ||∇D(x_interp)||₂
    # gradient_norm shape: (B,)
    gradient_norm = gradients.norm(2, dim=1)

    # --- 步骤 5: 梯度惩罚项 ---
    # 公式: λ × E[(||∇D||₂ - 1)²]
    # 当 ||∇D||₂ = 1 时，惩罚为 0（理想状态）
    # 当 ||∇D||₂ 偏离 1 时，惩罚增大，迫使梯度回到 1 附近
    loss_gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    # ========================================================================
    # 总损失 = Wasserstein 损失 + 梯度惩罚
    # ========================================================================
    loss = loss_gan + loss_gp
    return loss


# ============================================================================
#
#                           3. 生成器对抗损失 L_G
#
# ============================================================================

def L_G(D, fake_imgs):
    """
    生成器对抗损失（Generator Adversarial Loss, Wasserstein 形式）

    与 L_D 配合使用 WGAN-GP，判别器输出无界（activate=False）。

    公式: L_G = -E[D(G(z))]
    - 当 D(G(z)) 大（判别器被骗），loss 小（负数）
    - 当 D(G(z)) 小（判别器识别出假图），loss 大（正数）

    注意：不能用 torch.no_grad()，否则内层上下文会覆盖外层 enable_grad()，
    导致 loss_G 没有 grad_fn，梯度无法回传到 Bridge MLP。
    D 的参数不会被更新，因为 training.py 中只调 optimizer_brig.step()。

    参数:
        D:         判别器模型，输出无界值（activate=False）
        fake_imgs: 生成器生成的图片张量, shape (B, 3, H, W)，需保留梯度

    返回:
        loss_G: 标量张量（可反向传播）
    """
    fake_out = D(fake_imgs)["out"]
    loss_G = -fake_out.mean()
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
