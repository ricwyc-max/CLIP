__author__ = 'Eric'

# ============================================================================
#
#                           损失函数模块
#
# 本文件包含 CLIP2GAN 项目中用到的所有损失函数：
#   1. L_rec   — 重建损失（衡量生成图片与目标图片的像素级差异）
#   2. L_D     — WGAN-GP 判别器损失（标准 GAN 损失 + 梯度惩罚）
#
# 数学公式：
#   L_rec  = ||x - x_recon||₂           （L2 范数）
#   L_D    = -[E[log D(x)] + E[log(1 - D(x'))]] + λ × E[(||∇D(x')||₂ - 1)²]
#            ├── 标准 GAN 损失 ──┤     ├──────── 梯度惩罚 (GP) ─────────┤
#
# ============================================================================

import numpy as np
import torch
import torch.nn.functional as F


def L_rec(x, x_recon):
    """
    重建损失（Reconstruction Loss）

    衡量生成图片 x_recon 与目标图片 x 之间的像素级差异。
    使用 L2 范数（欧几里得距离），值越小说明两张图越接近。

    公式: L_rec = mean(||x - x_recon||₂)

    参数:
        x:       目标图片张量, shape (B, C, H, W)
                 - B: 批次大小
                 - C: 通道数（RGB = 3）
                 - H, W: 图片高宽
                 - 值域: [-1, 1] 或 [0, 1]，取决于预处理方式

        x_recon: 生成/重建的图片张量, shape (B, C, H, W)
                 - 必须和 x 的 shape 完全一致
                 - 值域应与 x 一致

    返回:
        loss_rec: 标量张量（可反向传播）
                  - 值越大，说明两张图差异越大
                  - 训练目标：最小化此值，让生成图逼近目标图

    使用示例:
        loss = L_rec(real_images, generated_images)
        loss.backward()  # 反向传播，更新生成器参数

    注意:
        - 这个函数只计算损失值，不包含 .mean() 的返回（原代码有）
        - L2 损失对大误差更敏感（平方项），但可能导致模糊
        - 如果想要更清晰的结果，可以考虑 L1 损失：torch.abs(x - x_recon).mean()
    """
    # torch.norm(x, p=2) 计算 L2 范数，即 sqrt(sum(x_i²))
    # .mean() 对 batch 维度取平均，得到标量
    loss_rec = torch.norm(x - x_recon, p=2).mean()
    return loss_rec


def L_D(D, real_imgs, fake_imgs, lambda_gp=10.0):
    """
    WGAN-GP 判别器损失（Discriminator Loss with Gradient Penalty）

    由两部分组成：
    1. 标准 GAN 损失：让判别器 D 给真实图片高分，给生成图片低分
    2. 梯度惩罚 (GP)：约束判别器的梯度范数接近 1，防止训练不稳定

    数学公式:
        L_D = -[E[log D(x)] + E[log(1 - D(x'))]] + λ × E[(||∇D(x')||₂ - 1)²]

    各项含义:
        - E[log D(x)]      : 真实图片的判别分数，D(x) 应该接近 1，log(1) = 0
        - E[log(1 - D(x'))] : 生成图片的判别分数，D(x') 应该接近 0，log(1-0) = 0
        - 梯度惩罚           : 防止判别器梯度爆炸或消失，保持训练稳定

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
    # 第一部分：标准 GAN 损失
    # ========================================================================
    # 公式: -[E[log D(x)] + E[log(1 - D(x')))]
    #
    # 直觉理解:
    #   - D(x) 是判别器给真实图片的分数（0~1 之间）
    #   - D(x') 是判别器给生成图片的分数（0~1 之间）
    #   - 我们希望 D(x) → 1（真实图片得高分）
    #   - 我们希望 D(x') → 0（生成图片得低分）
    #   - log(D(x)) 最大值在 D(x)=1 时取到（log(1)=0）
    #   - log(1-D(x')) 最大值在 D(x')=0 时取到（log(1)=0）

    # D(real) 应该接近 1（真实图片得分高）
    # real_out shape: (B, 1)
    real_out = D(real_imgs)["out"]

    # D(fake) 应该接近 0（生成图片得分低）
    # fake_out shape: (B, 1)
    fake_out = D(fake_imgs)["out"]

    # 标准 GAN 损失
    # + 1e-8 防止 log(0) 导致数值溢出（log(0) = -inf）
    # .mean() 对 batch 取平均
    loss_gan = -(torch.log(real_out + 1e-8) + torch.log(1 - fake_out + 1e-8)).mean()

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
    # 总损失 = 标准 GAN 损失 + 梯度惩罚
    # ========================================================================
    loss = loss_gan + loss_gp
    return loss

