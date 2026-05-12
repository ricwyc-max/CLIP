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

# ========== 强制修复模型加载问题 ==========
# 1. 禁用离线模式（允许加载模型）
if "HF_HUB_OFFLINE" in os.environ:
    print(f"检测到离线模式，已临时禁用: {os.environ['HF_HUB_OFFLINE']}")
    del os.environ["HF_HUB_OFFLINE"]

# 2. 设置缓存目录为项目本地 model 目录
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CLIP', 'mobileCLIP', 'model')
print(f"HF缓存目录: {os.environ['HF_HOME']}")

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
exp_name = "exp6"           # 实验文件夹名称
save_interval = 1           # 每 N 轮保存一次图像（完整 batch 网格图）
img_save_interval = 1000      # 每 N 张图片保存一次单张生成图
use_lpips = True            # 是否使用 LPIPS 感知损失
use_div = True              # 是否使用多样性损失
rec_mode = "mse"            # L_rec 模式
batch_size = 8              # 批次大小
accum_steps = 4             # 梯度累积步数（等效 batch_size = 4*8 = 32）

# ============================================================================
#                        三阶段渐进训练配置
# ============================================================================
total_epochs = 50          # 总训练轮数
stage1_ratio = 0.10         # 阶段一占比 35%（ep1-35）
stage2_ratio = 0.4         # 阶段二占比 40%（ep36-75）
stage3_ratio = 0.5         # 阶段三占比 25%（ep76-100）
warmup_epochs = 5           # 阶段切换时的平滑过渡轮数eopchs

# 截断热身（仅 Stage1 前半段）
truncation_psi_start = 0.5  # 截断起始值（0.5 = 强截断，保证初始生成质量）
truncation_ratio = 0.5      # 截断热身占 Stage1 的比例（前 50% 的 Stage1 内完成热身）

# 阶段一：稳建基础人脸（纯重建，不开判别器）
stage1 = {
    "lam_L_rec": 1.0, "lam_lpips": 0, "lam_G": 0, "lam_div": 0,
    "lam_clip": 5, "lam_reg": 0.005,
    "lr": 1e-4, "lr_D": 0
}

# 阶段二：引入结构细节（加入 LPIPS 和对抗损失）
stage2 = {
    "lam_L_rec": 1, "lam_lpips": 1, "lam_G": 1, "lam_div": 0,
    "lam_clip": 1, "lam_reg": 0.01,
    "lr": 5e-5, "lr_D": 2e-5
}

# 阶段三：精细打磨 + 多样性注入（全部激活）
stage3 = {
    "lam_L_rec":1, "lam_lpips": 1, "lam_G": 1, "lam_div": 1,
    "lam_clip": 1, "lam_reg": 0.01,
    "lr": 2e-5, "lr_D": 1e-5
}


# ============================================================================
#
#                               阶段切换函数
#
# ============================================================================

def get_stage_config(epoch):
    """根据当前 epoch 返回所属阶段的权重配置"""
    s1_end = int(total_epochs * stage1_ratio)
    s2_end = s1_end + int(total_epochs * stage2_ratio)
    if epoch <= s1_end:
        return stage1.copy(), "Stage1", 1, s1_end
    elif epoch <= s2_end:
        return stage2.copy(), "Stage2", s1_end + 1, s2_end
    else:
        return stage3.copy(), "Stage3", s2_end + 1, total_epochs


def get_current_weights(epoch):
    """
    获取当前 epoch 的权重，支持阶段切换时的线性插值过渡。

    warmup_epochs=5 时：在每个阶段的前 5 个 epoch，
    从上一阶段的权重线性过渡到当前阶段的权重。
    """
    cfg, stage_name, stage_start, stage_end = get_stage_config(epoch)

    # 计算在当前阶段内的进度
    progress_in_stage = epoch - stage_start

    # 如果不是阶段开头且有上一阶段，做平滑过渡
    if progress_in_stage < warmup_epochs and stage_name != "Stage1":
        # 获取上一阶段的配置
        if stage_name == "Stage2":
            prev_cfg = stage1.copy()
        else:
            prev_cfg = stage2.copy()

        t = progress_in_stage / warmup_epochs  # 0.0 → 1.0
        for k in cfg:
            if k in prev_cfg:
                cfg[k] = prev_cfg[k] + (cfg[k] - prev_cfg[k]) * t

    return cfg, stage_name


# ============================================================================
#
#                               初始化基础网络架构
#
# ============================================================================
def loadModel(use_lpips=True):
    """
    加载模型函数

    :param use_lpips:是否使用 LPIPS 感知损失
    :return:
    """
    # 初始化CLIP以及GAN网络
    CLIPandGAN = CLIP2GAN(device="cuda")
    birdgeNetwork = Bridge_MLP().to(device)

    # 判别器始终初始化（阶段二、三需要）
    D = Discriminator(size=1024, channels_in=3).to(device)
    D.stddev_group = 1  # 对整个 batch 算 std，适配任意 batch_size

    if use_lpips:
        lpips_fn = LF.LPIPS_AlexNet(device=device)
    else:
        lpips_fn = None

    optimizer_brig = torch.optim.Adam(birdgeNetwork.parameters(), lr=stage1["lr"])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=stage1["lr_D"])
    return CLIPandGAN, birdgeNetwork, D, lpips_fn, optimizer_brig, optimizer_D


# ============================================================================
#
#                               训练循环搭建
#
# ============================================================================
def training(CLIPandGAN, birdgeNetwork, optimizer_brig,
             D=None, optimizer_D=None,
             lpips_fn=None, use_lpips=True,
             exp_name="exp1", save_interval=5, img_save_interval=1000):
    """
    三阶段渐进训练循环

    :param CLIPandGAN:          CLIP与GAN网络实例化
    :param birdgeNetwork:       桥接网络
    :param optimizer_brig:      桥接网络优化器
    :param D:                   判别器
    :param optimizer_D:         判别器优化器
    :param lpips_fn:            LPIPS 损失函数
    :param use_lpips:           是否使用 LPIPS 感知损失
    :param exp_name:            实验名称
    :param save_interval:       每隔多少轮保存一次图像
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
        "stage": [],
        "L_D": [],
        "L_G": [],
        "L_rec": [],
        "L_lpips": [],
        "L_div": [],
        "L_clip": [],
        "L_reg": [],
        "L_total": []
    }
    best_loss = float('inf')

    print(f"[训练] 开始三阶段渐进训练，共 {total_epochs} 轮，实验目录: {exp_dir}")
    print(f"[训练] 梯度累积步数: {accum_steps}，等效 batch_size: {batch_size * accum_steps}")

    # CLIP Normalize 参数放到 device 上（只需一次）
    clip_mean = CLIP_MEAN.to(device)
    clip_std = CLIP_STD.to(device)

    # 截断热身：style_mean 用于截断 trick
    style_mean = CLIPandGAN.style_mean.to(device)  # (1, 512)

    # 预计算常量（数据集固定 1024×1024，这些值每 batch 相同）
    _scale = 224 / 1024
    _new_h, _new_w = int(1024 * _scale), int(1024 * _scale)
    _top = (_new_h - 224) // 2
    _left = (_new_w - 224) // 2

    # 阶段边界（只算一次）
    s1_end = int(total_epochs * stage1_ratio)
    truncation_end = int(s1_end * truncation_ratio)

    # ========================================================================
    # 训练主循环
    # ========================================================================
    total_batches = len(dataloader)
    total_imgs = len(dataset)

    total_imgs_done = 0  # 跨 epoch 的全局图片计数

    for epoch in range(1, total_epochs + 1):

        # ====================================================================
        # 阶段切换：获取当前权重和学习率
        # ====================================================================
        stage_cfg, stage_name = get_current_weights(epoch)

        # 更新学习率
        for pg in optimizer_brig.param_groups:
            pg['lr'] = stage_cfg["lr"]
        if optimizer_D is not None:
            for pg in optimizer_D.param_groups:
                pg['lr'] = stage_cfg["lr_D"]

        # Stage2 起启用判别器（权重从 Stage1 值渐增，lam_G 从 0 开始）
        use_D = stage_name != "Stage1"

        # 截断热身：仅在 Stage1 前半段生效，psi 从 0.5 线性增加到 1.0
        if epoch <= truncation_end and stage_name == "Stage1":
            psi = truncation_psi_start + (1.0 - truncation_psi_start) * (epoch / truncation_end)
        else:
            psi = 1.0
        print(f"\n[Epoch {epoch}/{total_epochs}] {stage_name} "
              f"(ep{stage_cfg.get('_range', '?')}) | "
              f"lr={stage_cfg['lr']:.1e} lr_D={stage_cfg['lr_D']:.1e} | "
              f"use_D={use_D} | psi={psi:.3f}")
        print(f"  权重: L_rec={stage_cfg['lam_L_rec']} L_lpips={stage_cfg['lam_lpips']} "
              f"L_G={stage_cfg['lam_G']} L_div={stage_cfg['lam_div']} "
              f"L_clip={stage_cfg['lam_clip']} L_reg={stage_cfg['lam_reg']}")
        epoch_L_D = 0.0
        epoch_L_G = 0.0
        epoch_L_rec = 0.0
        epoch_L_lpips = 0.0
        epoch_L_div = 0.0
        epoch_L_clip = 0.0
        epoch_L_reg = 0.0
        epoch_L_total = 0.0
        batch_count = 0

        for real_imgs in dataloader:
            # real_imgs: (B, 3, 1024, 1024), 值域 [0,1] (ToTensor 输出)
            real_imgs = real_imgs.to(device)

            # real_imgs_clip: (B, 3, 224, 224), CLIP 预处理（与 openclip 一致）
            # Resize(224): 短边缩放到 224，保持比例 → CenterCrop(224) → Normalize
            real_imgs_clip = F.interpolate(real_imgs, size=(_new_h, _new_w), mode='bicubic', align_corners=False)
            real_imgs_clip = real_imgs_clip[:, :, _top:_top+224, _left:_left+224]
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
                # 截断 trick: style = mean + psi * (style - mean)
                if psi < 1.0:
                    style_vector = style_mean + psi * (style_vector - style_mean)
                fake_imgs = CLIPandGAN.synthesis_net(style_vector.to(device))["img"].clamp(-1, 1)

            can_use_D = use_D and D is not None

            if can_use_D:

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
                # 截断 trick: style = mean + psi * (style - mean)
                if psi < 1.0:
                    style_vector = style_mean + psi * (style_vector - style_mean)
                fake_imgs = CLIPandGAN.synthesis_net(style_vector.to(device))["img"].clamp(-1, 1)

                if can_use_D:
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
                        # 截断 trick 也应用于多样性分支
                        if psi < 1.0:
                            style_noisy = style_mean + psi * (style_noisy - style_mean)
                        fake_imgs_noisy = CLIPandGAN.synthesis_net(style_noisy.to(device))["img"].clamp(-1, 1)
                    loss_div = LF.L_div(img_feat, img_feat_noisy, fake_imgs, fake_imgs_noisy)
                else:
                    loss_div = torch.tensor(0.0, device=device)

                # CLIP 余弦相似度损失：生成图 CLIP 特征 vs 原图 CLIP 特征
                # 使用与 real_imgs 相同的批量化预处理（数据集固定 1024×1024）
                fake_imgs_01 = (fake_imgs.clamp(-1, 1) + 1) / 2
                fake_imgs_clip = F.interpolate(fake_imgs_01, size=(_new_h, _new_w), mode='bicubic', align_corners=False)
                fake_imgs_clip = fake_imgs_clip[:, :, _top:_top+224, _left:_left+224]
                fake_imgs_clip = (fake_imgs_clip - clip_mean) / clip_std
                fake_feat = CLIPandGAN.encode_image(fake_imgs_clip)
                loss_clip = LF.L_clip(img_feat, fake_feat)

                # L1 正则：约束 style 向量稀疏
                loss_reg = LF.L_reg(style_vector)

                loss_total = (loss_rec * stage_cfg["lam_L_rec"]
                              + loss_lpips * stage_cfg["lam_lpips"]
                              + loss_G * stage_cfg["lam_G"]
                              + loss_div * stage_cfg["lam_div"]
                              + loss_clip * stage_cfg["lam_clip"]
                              + loss_reg * stage_cfg["lam_reg"])

            batch_count += 1

            # 梯度累积：除以累积步数，等效大 batch
            loss_total_scaled = loss_total / accum_steps
            loss_total_scaled.backward()

            # 累积到 accum_steps 步或最后一个 batch 时执行 optimizer step
            # （D 在 Step1 每 batch 已更新，这里只更新 Bridge）
            if (batch_count % accum_steps == 0) or (batch_count == total_batches):
                optimizer_brig.step()
                optimizer_brig.zero_grad()
                print(f"    >> 梯度更新 (batch {batch_count})")

            # 缓存 .item() 值，避免重复 CUDA 同步
            l_d_val = loss_D.item()
            l_g_val = loss_G.item()
            l_rec_val = loss_rec.item()
            l_lpips_val = loss_lpips.item()
            l_div_val = loss_div.item()
            l_clip_val = loss_clip.item()
            l_reg_val = loss_reg.item()
            l_total_val = loss_total.item()

            epoch_L_G += l_g_val
            epoch_L_rec += l_rec_val
            epoch_L_lpips += l_lpips_val
            epoch_L_div += l_div_val
            epoch_L_clip += l_clip_val
            epoch_L_reg += l_reg_val
            epoch_L_total += l_total_val

            # 每个 batch 打印一次日志
            imgs_done = batch_count * batch_size
            print(f"  [Epoch {epoch} | {stage_name} | psi={psi:.3f}] Batch {batch_count}/{total_batches} | "
                  f"Imgs {imgs_done}/{total_imgs} | "
                  f"L_D={l_d_val:.4f}  L_G={l_g_val:.4f}  "
                  f"L_rec={l_rec_val:.4f}  L_lpips={l_lpips_val:.4f}  "
                  f"L_div={l_div_val:.4f}  L_clip={l_clip_val:.4f}  "
                  f"L_reg={l_reg_val:.4f}  L_total={l_total_val:.4f}")

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
        avg_L_clip = epoch_L_clip / batch_count
        avg_L_reg = epoch_L_reg / batch_count
        avg_L_total = epoch_L_total / batch_count

        loss_history["epoch"].append(epoch)
        loss_history["stage"].append(stage_name)
        loss_history["L_D"].append(avg_L_D)
        loss_history["L_G"].append(avg_L_G)
        loss_history["L_rec"].append(avg_L_rec)
        loss_history["L_lpips"].append(avg_L_lpips)
        loss_history["L_div"].append(avg_L_div)
        loss_history["L_clip"].append(avg_L_clip)
        loss_history["L_reg"].append(avg_L_reg)
        loss_history["L_total"].append(avg_L_total)

        print(f"[Epoch {epoch}/{total_epochs}] {stage_name} "
              f"L_D={avg_L_D:.4f}  L_G={avg_L_G:.4f}  "
              f"L_rec={avg_L_rec:.4f}  L_lpips={avg_L_lpips:.4f}  "
              f"L_div={avg_L_div:.4f}  L_clip={avg_L_clip:.4f}  "
              f"L_reg={avg_L_reg:.4f}  L_total={avg_L_total:.4f}")

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
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
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

    axes[1, 1].plot(loss_history["epoch"], loss_history["L_div"], 'y-')
    axes[1, 1].set_title("L_div (Diversity)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")

    axes[1, 2].plot(loss_history["epoch"], loss_history["L_clip"], 'k-')
    axes[1, 2].set_title("L_clip (CLIP Cosine)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")

    axes[2, 0].plot(loss_history["epoch"], loss_history["L_reg"], 'tab:orange')
    axes[2, 0].set_title("L_reg (L1 Regularization)")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Loss")

    axes[2, 1].plot(loss_history["epoch"], loss_history["L_total"], 'm-')
    axes[2, 1].set_title("L_total")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Loss")

    axes[2, 2].axis('off')

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
    CLIPandGAN, birdgeNetwork, D, lpips_fn, optimizer_brig, optimizer_D = loadModel(use_lpips=use_lpips)
    training(
        CLIPandGAN=CLIPandGAN,
        birdgeNetwork=birdgeNetwork,
        optimizer_brig=optimizer_brig,
        D=D,
        optimizer_D=optimizer_D,
        lpips_fn=lpips_fn,
        use_lpips=use_lpips,
        exp_name=exp_name,
        save_interval=save_interval,
        img_save_interval=img_save_interval
    )
