__author__ = 'Eric'


# ============================================================================
#
#                           CLIP2GAN训练模块
#
# 整体架构：
#     原始图片 (3, 1024, 1024) → CLIP Image Encoder → img_feat (512维)
#                                                 ↓
#                                         [Bridge MLP]
#                                                 ↓
#                                     W+ latent code (23×512维)
#                                                 ↓
#                                         MobileStyleGAN → 生成图片 (3, 1024, 1024)
#
# ============================================================================



#=================================导包=======================================
import os
import sys

# MobileStyleGAN 代码路径（Discriminator 在里面）
_MOBILE_STYLEGAN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "StyleGAN", "mobileStyleGAN", "MobileStyleGAN.pytorch"
)
if _MOBILE_STYLEGAN_ROOT not in sys.path:
    sys.path.insert(0, _MOBILE_STYLEGAN_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，防止 tkinter 报错
import matplotlib.pyplot as plt
import LoadDatasets
from CLIP2GAN import CLIP2GAN
from bridgeNetwork import Bridge_MLP
from core.models.discriminator import Discriminator
import lossFunction as LF
# ============================================================================



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLIP ViT-B-32 的 Normalize 参数（open_clip 默认值）
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# ============================================================================
#
#                               统一参数配置（消融实验）
#
# ============================================================================
exp_name = "exp5"           # 实验文件夹名称
save_interval = 1           # 每 N 轮保存一次图像（完整 batch 网格图）
img_save_interval = 50      # 每 N 张图片保存一次单张生成图
use_D = True                # 是否使用判别器
use_lpips = True            # 是否使用 LPIPS 感知损失
use_div = True              # 是否使用多样性损失
rec_mode = "l2"             # L_rec 模式: L2 范数平方（原文）
lam_L_rec = 1               # L_rec 权重（基准，系数=1，量级 ~94000）
lam_lpips = 150             # λ_lpips（LPIPS ~0.15，×150 → ~22.5）
lam_G = 600                 # λ_G（L_G ~0.04，×600 → ~24）
lam_div = 200               # λ_div（L_div ~0.1，×200 → ~20）
epoches = 1                 # 训练轮数
batch_size = 2              # 批次大小
lr = 0.0001                 # Bridge MLP 学习率
lr_D = 0.0001              # 判别器学习率（比 Bridge 低，防止 D 太强）


# ============================================================================
#
#                               初始化基础网络架构
#
# ============================================================================
def loadModel(use_D=True, use_lpips=True):
    """
    加载模型函数，方便后续做消融实验
    :param use_D:    是否使用判别器
    :param use_lpips:是否使用 LPIPS 感知损失
    :return:
    """
    # 初始化CLIP以及GAN网络
    CLIPandGAN = CLIP2GAN(device="cuda")
    birdgeNetwork = Bridge_MLP().to(device)

    if use_D == True:
        D = Discriminator(size=1024, channels_in=3).to(device)
    else:
        D = None

    if use_lpips:
        lpips_fn = LF.LPIPS_AlexNet(device=device)
    else:
        lpips_fn = None

    optimizer_brig = torch.optim.Adam(birdgeNetwork.parameters(), lr=lr)
    if use_D == True:
        optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    else:
        optimizer_D = None
    return CLIPandGAN, birdgeNetwork, D, lpips_fn, optimizer_brig, optimizer_D


# ============================================================================
#
#                               训练循环搭建
#
# ============================================================================
def training(epoches, CLIPandGAN, birdgeNetwork, optimizer_brig,
             D=None, optimizer_D=None, use_D=True,
             lpips_fn=None, use_lpips=True,
             exp_name="exp1", save_interval=5, img_save_interval=1000):
    """
    训练循环，支持消融实验、图像保存、损失记录、模型保存

    :param epoches:             训练轮数
    :param CLIPandGAN:          CLIP与GAN网络实例化
    :param birdgeNetwork:       桥接网络
    :param optimizer_brig:      桥接网络优化器
    :param D:                   判别器（use_D=False 时为 None）
    :param optimizer_D:         判别器优化器（use_D=False 时为 None）
    :param use_D:               是否使用判别器
    :param lpips_fn:            LPIPS 损失函数（use_lpips=False 时为 None）
    :param use_lpips:           是否使用 LPIPS 感知损失
    :param exp_name:            实验名称，用于创建结果文件夹
    :param save_interval:       每隔多少轮保存一次图像（batch 网格图）
    :param img_save_interval:   每隔多少张图片保存一次单张生成图
    """

    # ========================================================================
    # 创建实验目录
    # ========================================================================
    exp_dir = os.path.join("results", exp_name)
    img_dir = os.path.join(exp_dir, "images")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ========================================================================
    # 加载数据集（ToTensor: PIL → tensor (3, 1024, 1024)，值域 [0,1]）
    # CLIP 编码时再通过 preprocess_img 做 Resize(224) + Normalize
    # L_rec 和 L_D 直接用 1024x1024 原图
    # ========================================================================
    dataset = LoadDatasets.MyCustomDataset(
        r'.\dataset\CelebAMask-HQ\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img',
        transform=transforms.ToTensor()  # PIL → tensor, [0,1]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ========================================================================
    # 记录损失用的列表
    # ========================================================================
    loss_history = {
        "epoch": [],
        "L_D": [],
        "L_G": [],
        "L_rec": [],
        "L_lpips": [],
        "L_div": [],
        "L_total": []
    }
    best_loss = float('inf')

    print(f"[训练] 开始训练，共 {epoches} 轮，实验目录: {exp_dir}")

    # CLIP Normalize 参数放到 device 上（只需一次）
    clip_mean = CLIP_MEAN.to(device)
    clip_std = CLIP_STD.to(device)

    # ========================================================================
    # 训练主循环
    # ========================================================================
    total_batches = len(dataloader)
    total_imgs = len(dataset)

    total_imgs_done = 0  # 跨 epoch 的全局图片计数

    for epoch in range(1, epoches + 1):
        epoch_L_D = 0.0
        epoch_L_G = 0.0
        epoch_L_rec = 0.0
        epoch_L_lpips = 0.0
        epoch_L_div = 0.0
        epoch_L_total = 0.0
        batch_count = 0

        for real_imgs in dataloader:
            # real_imgs: (B, 3, 1024, 1024), 值域 [0,1] (ToTensor 输出)
            real_imgs = real_imgs.to(device)

            # real_imgs_clip: (B, 3, 224, 224), CLIP 预处理（与 openclip 一致）
            # Resize(224): 短边缩放到 224，保持比例 → CenterCrop(224) → Normalize
            _, _, h, w = real_imgs.shape
            scale = 224 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            real_imgs_clip = F.interpolate(real_imgs, size=(new_h, new_w), mode='bicubic', align_corners=False)
            top = (new_h - 224) // 2
            left = (new_w - 224) // 2
            real_imgs_clip = real_imgs_clip[:, :, top:top+224, left:left+224]
            real_imgs_clip = (real_imgs_clip - clip_mean) / clip_std

            # real_imgs_scaled: (B, 3, 1024, 1024), 值域 [-1,1]
            # 用于 L_rec 和 L_D，与 StyleGAN 输出值域一致
            real_imgs_scaled = real_imgs * 2 - 1

            # CLIP 编码只需一次，Step 1 和 Step 2 共用
            img_feat = CLIPandGAN.encode_image(real_imgs_clip)

            # ====================================
            # 第1步：训练判别器 D
            # ====================================
            with torch.no_grad():
                style_vector = birdgeNetwork(img_feat)
                fake_imgs = CLIPandGAN.synthesis_net(style_vector.to(device))["img"].clamp(-1, 1)

            if use_D and D is not None and optimizer_D is not None:
                loss_D = LF.L_D(D, real_imgs_scaled, fake_imgs)
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                epoch_L_D += loss_D.item()
            else:
                loss_D = torch.tensor(0.0)

            # ====================================
            # 第2步：训练生成器 Bridge
            # ====================================
            with torch.enable_grad():
                style_vector = birdgeNetwork(img_feat)
                fake_imgs = CLIPandGAN.synthesis_net(style_vector.to(device))["img"].clamp(-1, 1)

                if use_D and D is not None:
                    loss_G = LF.L_G(D, fake_imgs)
                else:
                    loss_G = torch.tensor(0.0, device=device)

                loss_rec = LF.L_rec(real_imgs_scaled, fake_imgs, mode=rec_mode)

                if use_lpips and lpips_fn is not None:
                    loss_lpips = lpips_fn(real_imgs_scaled, fake_imgs)
                else:
                    loss_lpips = torch.tensor(0.0, device=device)

                # 多样性损失：对 img_feat 加噪声，比较两组生成图的差异
                if use_div:
                    img_feat_noisy = img_feat + torch.randn_like(img_feat)
                    with torch.no_grad():
                        style_noisy = birdgeNetwork(img_feat_noisy)
                        fake_imgs_noisy = CLIPandGAN.synthesis_net(style_noisy.to(device))["img"].clamp(-1, 1)
                    loss_div = LF.L_div(img_feat, img_feat_noisy, fake_imgs, fake_imgs_noisy)
                else:
                    loss_div = torch.tensor(0.0, device=device)

                loss_total = loss_rec * lam_L_rec + loss_lpips * lam_lpips + loss_G * lam_G + loss_div * lam_div

            batch_count += 1
            optimizer_brig.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(birdgeNetwork.parameters(), max_norm=1.0)
            optimizer_brig.step()

            epoch_L_G += loss_G.item()
            epoch_L_rec += loss_rec.item()
            epoch_L_lpips += loss_lpips.item()
            epoch_L_div += loss_div.item()
            epoch_L_total += loss_total.item()

            # 每个 batch 打印一次日志
            imgs_done = batch_count * batch_size
            print(f"  [Epoch {epoch}] Batch {batch_count}/{total_batches} | "
                  f"Imgs {imgs_done}/{total_imgs} | "
                  f"L_D={loss_D.item():.4f}  L_G={loss_G.item():.4f}  "
                  f"L_rec={loss_rec.item():.4f}  L_lpips={loss_lpips.item():.4f}  "
                  f"L_div={loss_div.item():.4f}  L_total={loss_total.item():.4f}")

            # 每 img_save_interval 张图片保存一张生成图和对应的原图
            total_imgs_done += batch_size
            if total_imgs_done % img_save_interval < batch_size:
                single_real = real_imgs[0]  # [0,1]
                single_fake = (fake_imgs[0].detach().clamp(-1, 1) + 1) / 2  # 转 [0,1]
                save_image(single_real, os.path.join(img_dir, f"img_{total_imgs_done}_real.png"))
                save_image(single_fake, os.path.join(img_dir, f"img_{total_imgs_done}_fake.png"))
                print(f"  >> 原图与生成图已保存 ({total_imgs_done} imgs)")

        # ====================================================================
        # 每轮结束：计算平均损失
        # ====================================================================
        avg_L_D = epoch_L_D / batch_count
        avg_L_G = epoch_L_G / batch_count
        avg_L_rec = epoch_L_rec / batch_count
        avg_L_lpips = epoch_L_lpips / batch_count
        avg_L_div = epoch_L_div / batch_count
        avg_L_total = epoch_L_total / batch_count

        loss_history["epoch"].append(epoch)
        loss_history["L_D"].append(avg_L_D)
        loss_history["L_G"].append(avg_L_G)
        loss_history["L_rec"].append(avg_L_rec)
        loss_history["L_lpips"].append(avg_L_lpips)
        loss_history["L_div"].append(avg_L_div)
        loss_history["L_total"].append(avg_L_total)

        print(f"[Epoch {epoch}/{epoches}] "
              f"L_D={avg_L_D:.4f}  L_G={avg_L_G:.4f}  "
              f"L_rec={avg_L_rec:.4f}  L_lpips={avg_L_lpips:.4f}  "
              f"L_div={avg_L_div:.4f}  L_total={avg_L_total:.4f}")

        # ====================================================================
        # 保存最佳模型（按 L_total）
        # ====================================================================
        if avg_L_total < best_loss:
            best_loss = avg_L_total
            torch.save(birdgeNetwork.state_dict(),
                       os.path.join(ckpt_dir, "best_bridge.pth"))
            print(f"  >> 最佳模型已保存 (loss={best_loss:.4f})")

        # ====================================================================
        # 每 save_interval 轮保存图像
        # ====================================================================
        if epoch % save_interval == 0:
            # 取最后一个 batch 的图做可视化
            # real_imgs: (B, 3, 1024, 1024), 值域 [0,1]
            # fake_imgs: (B, 3, 1024, 1024), 值域 [-1,1]
            real_grid = make_grid(real_imgs, nrow=4, padding=2)  # 已经是 [0,1]
            fake_grid = make_grid((fake_imgs.detach().clamp(-1, 1) + 1) / 2, nrow=4, padding=2)

            save_image(real_grid, os.path.join(img_dir, f"epoch_{epoch}_real.png"))
            save_image(fake_grid, os.path.join(img_dir, f"epoch_{epoch}_fake.png"))
            print(f"  >> 图像已保存到 {img_dir}")

    # ========================================================================
    # 训练结束：保存最后一轮权重
    # ========================================================================
    torch.save(birdgeNetwork.state_dict(),
               os.path.join(ckpt_dir, "last_bridge.pth"))
    print(f"[训练完成] 最终权重已保存到 {ckpt_dir}/last_bridge.pth")

    # ========================================================================
    # 保存损失 CSV
    # ========================================================================
    df = pd.DataFrame(loss_history)
    csv_path = os.path.join(exp_dir, "losses.csv")
    df.to_csv(csv_path, index=False)
    print(f"[训练完成] 损失记录已保存到 {csv_path}")

    # ========================================================================
    # 绘制损失曲线
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Training Loss - {exp_name}", fontsize=14)

    axes[0, 0].plot(loss_history["epoch"], loss_history["L_D"], 'b-')
    axes[0, 0].set_title("L_D (Discriminator)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(loss_history["epoch"], loss_history["L_G"], 'r-')
    axes[0, 1].set_title("L_G (Generator)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[0, 2].plot(loss_history["epoch"], loss_history["L_rec"], 'g-')
    axes[0, 2].set_title("L_rec (Reconstruction MSE)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")

    axes[1, 0].plot(loss_history["epoch"], loss_history["L_lpips"], 'c-')
    axes[1, 0].set_title("L_lpips (Perceptual)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(loss_history["epoch"], loss_history["L_total"], 'm-')
    axes[1, 1].set_title("L_total")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")

    axes[1, 2].plot(loss_history["epoch"], loss_history["L_div"], 'y-')
    axes[1, 2].set_title("L_div (Diversity)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")

    plt.tight_layout()
    curve_path = os.path.join(exp_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[训练完成] 损失曲线已保存到 {curve_path}")


# ============================================================================
#
#                               启动训练
#
# ============================================================================
if __name__ == "__main__":
    CLIPandGAN, birdgeNetwork, D, lpips_fn, optimizer_brig, optimizer_D = loadModel(use_D=use_D, use_lpips=use_lpips)
    training(
        epoches=epoches,
        CLIPandGAN=CLIPandGAN,
        birdgeNetwork=birdgeNetwork,
        optimizer_brig=optimizer_brig,
        D=D,
        optimizer_D=optimizer_D,
        use_D=use_D,
        lpips_fn=lpips_fn,
        use_lpips=use_lpips,
        exp_name=exp_name,
        save_interval=save_interval,
        img_save_interval=img_save_interval
    )
