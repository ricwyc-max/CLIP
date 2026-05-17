"""
CLIP2GAN 测试模块 — PyCharm 直接运行版

在 PyCharm 中直接右键 → Run 'testing' 即可。
如需修改配置，直接在下方【用户配置区】改全局变量。

支持两种测试模式：
    1. 图像重生成：加载真实图片 → CLIP 编码 → Bridge MLP → StyleGAN 生成
    2. 文生图：    输入文本 → CLIP 编码 → Bridge MLP → StyleGAN 生成
"""

__author__ = 'Eric'

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================== 路径设置 ================================
_MOBILE_STYLEGAN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "StyleGAN", "mobileStyleGAN", "MobileStyleGAN.pytorch"
)
if _MOBILE_STYLEGAN_ROOT not in sys.path:
    sys.path.insert(0, _MOBILE_STYLEGAN_ROOT)

os.environ['HF_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'CLIP', 'mobileCLIP', 'model'
)
os.environ['HF_HUB_OFFLINE'] = "1"



import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

from CLIP2GAN import CLIP2GAN
from bridgeNetwork import Bridge_MLP


# ============================================================================
#                          ★ 用户配置区 ★
#  在 PyCharm 里直接修改下面的变量，然后 Run 即可
# ============================================================================

# Bridge MLP 权重路径（必填，改为自己的权重路径）
BRIDGE_WEIGHT_PATH = r".\archive\checkpoints\last_bridge.pth"

# 测试模式: "recon"=图像重生成, "txt2img"=文生图, "all"=两项都跑
TEST_MODE = "all"

# 测试图片路径（图像重生成用）— None 则使用 CLIP/mobileCLIP/docs/ 下的默认图
# 可填: None, 单张路径 "path/to/img.jpg", 或多张 ["a.jpg", "b.jpg"]
TEST_IMAGES = [r".\test_results\img_1081000_real.png",r".\test_results\img_1082000_real.png"]

# 测试文本（文生图用）— None 则使用内置默认文本
TEST_TEXTS = None

# 结果保存目录
SAVE_DIR = "test_results"

# 截断系数 (0.0~1.0)，1.0=无截断
TRUNCATION_PSI = 1.0

# 特征噪声强度（模仿 training 多样性损失，每次生成加不同噪声）
# 0.0=关闭噪声（确定性生成），建议 0.05~0.2
RECON_NOISE_SCALE = 0.0    # 图像重生成
TXT2IMG_NOISE_SCALE = 0.005  # 文生图


# ============================================================================
#                        以下为代码实现，无需修改
# ============================================================================

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================== 辅助函数 ===========================

def load_test_model(bridge_weight_path):
    """加载 CLIP2GAN + Bridge MLP，eval 模式"""
    print(f"[Testing] 设备: {device}")
    print(f"[Testing] 加载 CLIP2GAN...")
    clip2gan = CLIP2GAN(device=device)
    print(f"[Testing] 加载 Bridge MLP 权重: {bridge_weight_path}")
    bridge = Bridge_MLP().to(device)
    state_dict = torch.load(bridge_weight_path, map_location=device, weights_only=True)
    bridge.load_state_dict(state_dict)
    bridge.eval()
    return clip2gan, bridge


def load_real_image(image_path):
    """加载图片 → (1,3,1024,1024) [0,1] 和 [-1,1]"""
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((1024, 1024), Image.BICUBIC)
    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    img_01 = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_norm = img_01 * 2 - 1
    return img_01.to(device), img_norm.to(device)


def clip_preprocess(imgs_01):
    """将 [0,1] 图像预处理为 CLIP 输入 (B,3,224,224)"""
    _scale = 224 / 1024
    _new_h, _new_w = int(1024 * _scale), int(1024 * _scale)
    _top = (_new_h - 224) // 2
    _left = (_new_w - 224) // 2
    x = F.interpolate(imgs_01, size=(_new_h, _new_w), mode='bicubic', align_corners=False)
    x = x[:, :, _top:_top + 224, _left:_left + 224]
    x = (x - CLIP_MEAN.to(imgs_01.device)) / CLIP_STD.to(imgs_01.device)
    return x


@torch.no_grad()
def generate_from_feat(clip2gan, bridge, feat, truncation_psi=1.0, noise_scale=0.1):
    """
    CLIP 特征 → Bridge MLP → W+ → StyleGAN → 图片

    参数:
        noise_scale: 对 feat 加噪的强度，模仿 training 中的多样性损失。
                     每次调用加不同噪声，保证同文本生成不同结果。
                     设为 0 则关闭噪声。
    """
    if noise_scale > 0:
        feat = feat + torch.randn_like(feat) * noise_scale
    styles = bridge(feat)
    if truncation_psi < 1.0:
        style_mean = clip2gan.style_mean.to(styles.device)
        styles = style_mean + truncation_psi * (styles - style_mean)
    fake_imgs = clip2gan.synthesis_net(styles)["img"].clamp(-1, 1)
    return fake_imgs, styles


def tensor_to_numpy(tensor):
    """将 (C,H,W) tensor [-1,1] 转为 numpy (H,W,3) [0,255] uint8"""
    t = tensor.detach().clamp(-1, 1)
    t = (t + 1) / 2
    img = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


# ========================== 可视化 ==========================

def show_images(images, titles=None, figsize=(16, 6), suptitle=None):
    """
    用 matplotlib 弹窗显示多张图片。
    images: list of (H,W,3) uint8
    """
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


# ===================== 图像重生成测试 =====================

def test_reconstruction(clip2gan, bridge, image_path, truncation_psi=1.0, save_dir="test_results"):
    """测试单张图片重生成"""
    img_01, img_norm = load_real_image(image_path)

    # CLIP 编码
    img_clip_in = clip_preprocess(img_01)
    img_feat = clip2gan.encode_image(img_clip_in)

    # 生成
    fake_imgs, styles = generate_from_feat(clip2gan, bridge, img_feat,
                                            truncation_psi=truncation_psi,
                                            noise_scale=RECON_NOISE_SCALE)
    fake_01 = (fake_imgs + 1) / 2

    # 评估
    fake_clip_in = clip_preprocess(fake_01)
    fake_feat = clip2gan.encode_image(fake_clip_in)
    cos_sim = (img_feat * fake_feat).sum(dim=-1).item()
    l_rec = F.mse_loss(img_norm, fake_imgs).item()

    # 转 numpy 显示
    img_orig = tensor_to_numpy(img_01[0])
    img_fake = tensor_to_numpy(fake_imgs[0])

    # ---- 弹窗显示 ----
    show_images(
        [img_orig, img_fake],
        titles=[f"Original", f"Reconstructed  cos_sim={cos_sim:.4f}"],
        suptitle=f"Reconstruction: {os.path.basename(image_path)}"
    )

    # ---- 保存 ----
    from torchvision.utils import save_image
    os.makedirs(save_dir, exist_ok=True)
    comparison = torch.cat([img_01, fake_01], dim=0)
    grid = make_grid(comparison, nrow=2, padding=4)
    save_path = os.path.join(save_dir, f"recon_{os.path.basename(image_path)}")
    save_image(grid, save_path)

    # ---- 控制台输出 ----
    basename = os.path.basename(image_path)
    print(f"\n{'=' * 50}")
    print(f"图像重生成: {basename}")
    print(f"{'=' * 50}")
    print(f"  L_rec (MSE):         {l_rec:.6f}")
    print(f"  CLIP cos_sim:        {cos_sim:.4f}")
    print(f"  W+ style: mean={styles.mean().item():.4f}, std={styles.std().item():.4f}")
    print(f"  对比图已保存: {save_path}")

    return {"L_rec": l_rec, "cos_sim": cos_sim}


# ========================= 文生图测试 =========================

def test_text_to_image(clip2gan, bridge, texts, truncation_psi=1.0, save_dir="test_results"):
    """测试文生图"""
    text_feat = clip2gan.encode_text(texts)
    fake_imgs, styles = generate_from_feat(clip2gan, bridge, text_feat,
                                            truncation_psi=truncation_psi,
                                            noise_scale=TXT2IMG_NOISE_SCALE)

    # 图文相似度
    fake_01 = (fake_imgs + 1) / 2
    fake_clip_in = clip_preprocess(fake_01)
    fake_feat = clip2gan.encode_image(fake_clip_in)
    sim_matrix = fake_feat @ text_feat.T  # (N_text, N_text)
    diag_sim = sim_matrix.diag().cpu().tolist()

    # ---- 控制台输出 ----
    print(f"\n{'=' * 50}")
    print(f"文生图测试 ({len(texts)} 条文本)")
    print(f"{'=' * 50}")
    for i, (t, sim) in enumerate(zip(texts, diag_sim)):
        print(f"  [{i}] \"{t}\"")
        print(f"       自匹配 cos_sim: {sim:.4f}")

    print(f"\n  图文余弦相似度矩阵 ({len(texts)}x{len(texts)}):")
    header = "".join(f"{t[:10]:>12s}" for t in texts)
    print(f"  {'':16s}{header}")
    for i in range(len(texts)):
        row = "  ".join(f"{sim_matrix[i, j].item():10.4f}" for j in range(len(texts)))
        print(f"  {texts[i][:14]:14s}{row}")

    print(f"\n  W+ style: mean={styles.mean().item():.4f}, std={styles.std().item():.4f}")

    # ---- 保存 ----
    from torchvision.utils import save_image
    os.makedirs(save_dir, exist_ok=True)
    grid = make_grid(fake_01, nrow=4, padding=4)
    grid_path = os.path.join(save_dir, "txt2img_grid.png")
    save_image(grid, grid_path)
    print(f"  生成图网格已保存: {grid_path}")

    for i, t in enumerate(texts):
        path = os.path.join(save_dir, f"txt2img_{i}.png")
        save_image(fake_01[i], path)

    # ---- 弹窗显示 ----
    imgs_np = [tensor_to_numpy(fake_imgs[i]) for i in range(len(texts))]
    # 最多显示 6 张，分两行
    n_show = min(len(texts), 6)

    import matplotlib.pyplot as plt
    n_cols = min(3, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_show > 1 else [axes]
    fig.suptitle("Text-to-Image Results", fontsize=14)
    for i in range(n_show):
        axes[i].imshow(imgs_np[i])
        axes[i].axis("off")
        axes[i].set_title(f"[{i}] {texts[i][:20]}", fontsize=10)
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    return fake_imgs, sim_matrix


# ======================== 主入口 ========================

def main():
    print(f"\n{'#' * 60}")
    print(f"#   CLIP2GAN 测试套件")
    print(f"#   权重: {BRIDGE_WEIGHT_PATH}")
    print(f"#   设备: {device}")
    print(f"#   模式: {TEST_MODE}")
    print(f"{'#' * 60}\n")

    # 校验权重路径
    if not os.path.exists(BRIDGE_WEIGHT_PATH):
        print(f"[错误] 权重文件不存在: {BRIDGE_WEIGHT_PATH}")
        print(f"请修改 testing.py 顶部的 BRIDGE_WEIGHT_PATH 为你的权重路径。")
        input("\n按回车键退出...")
        return

    # 加载模型
    clip2gan, bridge = load_test_model(BRIDGE_WEIGHT_PATH)

    # 确定保存目录
    weight_dir_name = os.path.basename(os.path.dirname(BRIDGE_WEIGHT_PATH))
    save_dir = os.path.join(SAVE_DIR, f"test_{weight_dir_name}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- 测试1: 图像重生成 ----
    if TEST_MODE in ("all", "recon"):
        print(f"\n{'=' * 60}")
        print(f"  测试 1/2: 图像重生成")
        print(f"{'=' * 60}")

        if TEST_IMAGES:
            # 支持单张路径字符串和路径列表
            if isinstance(TEST_IMAGES, str):
                image_paths = [TEST_IMAGES]
            else:
                image_paths = list(TEST_IMAGES)
        else:
            # 使用默认测试图片
            test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLIP", "mobileCLIP", "docs")
            candidates = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if candidates:
                image_paths = [os.path.join(test_dir, candidates[0])]
                print(f"  使用测试图片: {image_paths[0]}")
            else:
                print(f"  [警告] 未找到测试图片，跳过重生成测试")
                image_paths = []

        for img_path in image_paths:
            test_reconstruction(clip2gan, bridge, img_path,
                                truncation_psi=TRUNCATION_PSI, save_dir=save_dir)

    # ---- 测试2: 文生图 ----
    if TEST_MODE in ("all", "txt2img"):
        print(f"\n{'=' * 60}")
        print(f"  测试 2/2: 文生图")
        print(f"{'=' * 60}")

        texts = TEST_TEXTS or [
            "a black man",
            "a white yound woman with a blonde hair and a green eyes",
            "a make up woman with blonde curly hair red lips and a pair of dimple",
            "a yound beautiful woman with black straight hair and a blue eyes",
            "The old man is in his eighties. His face is covered with deep wrinkles, especially around the eyes and mouth. Crow's feet spread from the corners of his eyes, and frown lines are etched across his forehead. His skin is loose and sagging on his cheeks and neck, with dark age spots scattered here and there. His hair is completely white, thin and dry, combed flat against his scalp. His eyes are sunken and framed by heavy, droopy eyelids. His nose is large and slightly bulbous at the tip. His lips are thin and pale, almost disappearing into the wrinkles around his mouth. His hands are bony, with prominent veins and knobby knuckles.",
            "She is a stunningly beautiful young woman. She has a delicate oval face with smooth, fair skin. Her big, expressive eyes are as bright as stars, framed with long, curly eyelashes. She has a high-bridged nose and naturally red, full lips that form a perfect cupid's bow. Her shiny, dark brown hair falls in soft waves over her shoulders. When she smiles, two sweet dimples appear on her cheeks, making her look both charming and approachable."
        ]
        test_text_to_image(clip2gan, bridge, texts,
                           truncation_psi=TRUNCATION_PSI, save_dir=save_dir)

    print(f"\n{'=' * 60}")
    print(f"  所有测试完成！结果保存目录: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    input("\n按回车键退出...")  # 防止 PyCharm 终端闪退
