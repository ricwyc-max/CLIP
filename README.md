# 项目名称：基于 CLIP 引导的文本生成图像大模型
[![查看项目看板](https://img.shields.io/badge/github-Project-blue?logo=github)](https://github.com/ricwyc-max/CLIP)
---

> **课程**：生成式人工智能  
> **状态**：进行中 / 训练流程已跑通  
> **指导老师**：陈培垠  
> **指导老师邮箱**：pychen@hhu.edu.cn  
> **大作业成员邮箱**：134074852@qq.com  



## 1. 项目简介

本项目旨在构建一个**文本到图像Text-to-Image**的生成模型。在文生图任务中，核心挑战在于如何让模型“理解”自然语言描述，并生成与之语义对齐的高保真图像。

为此，我们利用 OpenAI 提出的 **CLIP（Contrastive Language-Image Pre-training）** 模型作为核心语义桥接器。CLIP 通过对比学习，将文本和图像映射到同一个联合嵌入空间（Joint Embedding Space），使得“一只戴帽子的猫”的文本特征与对应的图像特征在空间中的距离拉近。

在本项目中，CLIP 将承担以下关键角色：
1.  **条件引导**：指导生成模型（如 Diffusion GAN 或 Transformer）产生符合文本描述的图像内容。
2.  **语义损失计算**：计算生成图像与目标文本之间的 CLIP 相似度，作为损失函数的一部分。

## 2. 背景动机

传统的文生图模型（如早期的 GANs）常因文本编码器表达能力不足，导致生成图像出现“语义漂移”。CLIP 在大规模图文对（4亿组）上的预训练特性，使其具备强大的零样本（Zero-shot）泛化能力。通过引入 CLIP，我们希望解决：

-   **细粒度对齐**：精确还原文本中的属性（颜色、形状）、数量及关系。
-   **丰富多样性**：利用 CLIP 的泛化能力，生成训练数据中未出现的组合概念。

## 3. 模型架构（概览）

*（待补充：此处将放入整体架构图）*
1. **对比试验**: 
- **使用基本DCGAN网络进行图像生成、CLIP进行引导(CAFE-GAN, RATLIP等)**
  - 具体方案：

- **使用Style-GAN网络进行图像生成、CLIP进行引导（CLIP2GAN）**
  - 具体方案：

- **使用Stable Diffusion网络进行图像生成、CLIP进行引导（ DALL·E 2 、DiffusionCLIP ）**
  - 具体方案：

- **使用Diffusion GAN网络进行图像生成、CLIP进行引导（UFOGen 、clip2latent ）**
  - 具体方案：

- **使用Transformer网络进行图像生成、CLIP进行引导（DALL·E）**
  - 具体方案：

- **使用mobileStyleGAN网络进行图像生成、mobileCLIP网络进行图像引导（CLIP2GAN）**
  - 具体方案：
1、使用预训练mobileStyleGAN进行图像生成，预训练mobileStyleGAN权重获取参考地址：https://github.com/bes-dev/MobileStyleGAN.pytorch  
2、使用预训练mobileCLIP进行文本编码，预训练mobileCLIP权重获取参考地址：https://github.com/mlfoundations/open_clip  
3、自建 Bridge MLP 隐空间向量语义转换层  
4、进行不同损失函数的消融实验  
  - 模型输出展示：  

| 图片 | 说明 |
|------|------|
| ![图片1](./CLIP/mobileCLIP/docs/dog.jpg "测试图片") | mobileCLIP测试图像 |
| ![图片2](./CLIP/mobileCLIP/docs/test.png "展示结果") | mobileCLIP展示结果 |  

| 图片 | mobilestyleGAN随机输出结果 |
|------|------|
| <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test.png" width="300">  |  <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test1.png" width="300">  |
| <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test2.png" width="300">  |  <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test3.png" width="300">  |
| <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test4.png" width="300">  |  <img src="./StyleGAN/mobileStyleGAN/MobileStyleGAN.pytorch/testimg/test5.png" width="300">  |

  - 训练设备与超参数：

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU(8GBs) |
| 框架 | PyTorch 2.7.1 + CUDA 11.8 |
| CLIP 模型 | ViT-B-32（laion2b_s34b_b79k），512维输出，冻结 |
| 生成器 | MobileStyleGAN（FFHQ v2），style_dim=512，输出 1024×1024，冻结 |
| 判别器 | MobileStyleGAN Discriminator（1024×1024，3通道） |
| 数据集 | CelebA-HQ（30000 张 1024×1024 人脸图像） |

  - 损失函数与超参数配置（exp5）：

$$L_{total} = \lambda_{rec} L_{rec} + \lambda_{lpips} L_{lpips} + \lambda_G L_G + \lambda_{div} L_{div} + \lambda_{clip} L_{clip} + \lambda_{reg} L_{reg}$$

| 损失 | 公式 | 权重 | 说明 |
|------|------|------|------|
| L_rec | `MSE(x, x')` | 0.001 | 像素级重建损失 |
| L_lpips | AlexNet LPIPS | 150 | 感知细节损失 |
| L_D | `-E[log D(x)] - E[log(1-D(x'))] + λ·GP` | — | WGAN-GP 判别器损失（单独优化） |
| L_G | `E[log(1-D(x'))]` | 600 | 生成器对抗损失 |
| L_div | `d_f / d_I` | 200 | 特征扰动/图像差异，防模式坍缩 |
| L_clip | `1 - cos(clip_real, clip_fake)` | 50 | CLIP 语义对齐损失 |
| L_reg | `mean(\|style\|)` | 0.01 | Bridge MLP 输出 L1 稀疏约束 |

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `batch_size` | 2 | 批次大小 |
| `epoches` | 1 | 训练轮数 |
| `lr` | 1e-4 | Bridge MLP 学习率（Adam） |
| `lr_D` | 1e-4 | 判别器学习率（Adam） |
| `max_norm` | 1.0 | 梯度裁剪阈值 |

  - 模型架构：



**核心流程（CLIP2GAN 方案）**：
```
原始图片 (3, 1024, 1024)
    → CLIP Image Encoder → img_embedding (512维)
    → [Bridge MLP] → W+ latent code (23×512维)
    → MobileStyleGAN → 生成图片 (3, 1024, 1024)
```
1.  **图像编码**：输入图片 → CLIP ViT-B-32 → 512维图像特征（冻结，不训练）
2.  **特征转换**：图像特征 → Bridge MLP（12层全连接） → W+ latent code（23×512维，**可训练**）
3.  **图像生成**：W+ latent code → MobileStyleGAN SynthesisNetwork → 1024×1024图片（冻结，不训练）
4.  **损失函数**（对齐 CLIP2GAN 原文公式 $L_{min} = L_{rec} + \lambda_{lpips}L_{lpips} + \lambda_G L_G + \lambda_{div}L_{div}$）：
    - **L_rec**：重建损失（MSE），衡量生成图与原图的像素级差异
    - **L_lpips**：感知损失（AlexNet LPIPS），关注生成图细节质量
    - **L_D**：WGAN-GP 判别器损失（标准GAN损失 + 梯度惩罚）
    - **L_G**：生成器对抗损失（non-saturating log）
    - **L_div**：多样性损失，防止模式坍缩
    - **L_clip**：CLIP 余弦相似度损失，语义对齐
    - **L_reg**：L1 正则，约束 Bridge MLP 输出稀疏
5.  **训练策略**：Step1 训练判别器 D，Step2 训练 Bridge MLP（L_rec + L_lpips + L_G + L_div + L_clip + L_reg），CLIP 和 MobileStyleGAN 全程冻结
6.  **消融实验**：对不同损失函数组合进行消融实验（exp1 ~ exp5）

**基线模型参考**：
-   （例如：DALL·E 2， Stable Diffusion， 或 简单的 Conditional GAN + CLIP）
- CLIP = 判别式模型（理解/匹配）
- DALL·E / Stable Diffusion / Style-GAN = 生成式模型（创造/生成）
- 用 CLIP 作为“导师”来训练一个生成式模型

## 4. 实验计划

-   **数据集**：
CelebAMask-HQ(https://github.com/switchablenorms/CelebAMask-HQ)
CelebA-HQ(git clone https://www.modelscope.cn/datasets/OmniData/CelebA-HQ.git) (https://huggingface.co/datasets/iamivan11/CelebA-HQ-zip)
-   **评估指标**：FID（图像质量）、CLIP-Score（语义一致性）、人工评估
-   **对比实验**：CLIP 引导的不同质量生成器对比
-   **消融实验**：无 CLIP 引导的生成器 vs. 有 CLIP 引导的生成器

## 5. 当前进度

-   [x] 文献调研：理解 CLIP 的对比学习原理及在文生图中的应用
-   [x] 环境配置与 CLIP 预训练权重加载
-   [x] 搭建生成器骨架（MobileStyleGAN）
-   [x] 封装 CLIP2GAN 统一接口（CLIP + MobileStyleGAN）
-   [x] 实现损失函数模块（L_rec、L_D、L_G、L_lpips、L_div、L_clip、L_reg）
-   [x] Bridge MLP 改为输出 W+ latent code（23×512维）
-   [x] 搭建训练流程（消融实验配置、图像保存、损失记录、模型保存）
-   [x] 消融实验 exp1 ~ exp5，对齐 CLIP2GAN 原文损失公式
-   [ ] 训练与调参
-   [ ] 评估指标实现（FID、CLIP-Score）
-   [ ] 文本到图像推理

## 6. 如何运行

**环境安装**
```bash
git clone [仓库地址]
cd CLIP
pip install -r requirements.txt
```

**数据集准备**

将 CelebA-HQ 图片放入 `dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/` 目录。

**训练**
```bash
# 修改 training.py 顶部的统一参数配置（exp_name、batch_size、epoches 等）
python training.py
```

训练产物输出到 `results/{exp_name}/`：
- `images/` — 每轮 batch 网格图 + 每 1000 张的单张生成图
- `checkpoints/` — best_bridge.pth（最佳权重）、last_bridge.pth（最终权重）
- `losses.csv` — 每轮损失记录
- `loss_curve.png` — 损失曲线图

**推理测试**
```bash
python CLIP2GAN.py
```

## Requirements
---

主要依赖：
<details>
  <summary>📂 点击展开/收起：具体依赖（太长已折叠）</summary>  

```text
Package                   Version
------------------------- ------------
addict                    2.4.0
aiohappyeyeballs          2.6.1
aiohttp                   3.13.5
aiosignal                 1.4.0
annotated-doc             0.0.4
anyio                     4.12.1
arrow                     1.4.0
async-timeout             5.0.1
attrs                     26.1.0
beautifulsoup4            4.14.3
boto3                     1.42.97
botocore                  1.42.97
bravado                   11.1.0
bravado-core              6.3.1
cattrs                    25.3.0
certifi                   2026.4.22
charset-normalizer        3.4.7
click                     8.1.8
colorama                  0.4.6
coremltools               9.0
exceptiongroup            1.3.1
filelock                  3.19.1
fqdn                      1.5.1
frozenlist                1.8.0
fsspec                    2025.10.0
ftfy                      6.3.1
future                    1.0.0
gdown                     5.2.2
gitdb                     4.0.12
GitPython                 3.1.49
h11                       0.16.0
hf-xet                    1.4.3
httpcore                  1.0.9
httpx                     0.28.1
huggingface_hub           1.8.0
idna                      3.13
importlib_resources       6.5.2
isoduration               20.11.0
Jinja2                    3.1.6
jmespath                  1.1.0
jsonpointer               3.0.0
jsonref                   1.1.0
jsonschema                4.25.1
jsonschema-specifications 2025.9.1
kornia                    0.8.2
kornia_rs                 0.1.10
lark                      1.3.1
lightning-utilities       0.15.2
markdown-it-py            3.0.0
MarkupSafe                3.0.2
mdurl                     0.1.2
monotonic                 1.6
mpmath                    1.3.0
msgpack                   1.1.2
multidict                 6.7.1
neptune-client            1.14.0.post2
networkx                  3.2.1
ninja                     1.13.0
numpy                     2.0.2
oauthlib                  3.3.1
open_clip_torch           3.3.0
opencv-python             4.13.0.92
packaging                 26.2
pandas                    2.3.3
pillow                    11.3.0
pip                       26.0.1
piq                       0.8.0
propcache                 0.4.1
protobuf                  6.33.6
psutil                    7.2.2
pyaml                     26.2.1
Pygments                  2.20.0
PyJWT                     2.12.1
PySocks                   1.7.1
python-dateutil           2.9.0.post0
pytorch-fid               0.3.0
pytorch-lightning         2.6.0
pytorch_wavelets          1.3.0
pytz                      2026.1.post1
PyWavelets                1.6.0
PyYAML                    6.0.3
referencing               0.36.2
regex                     2026.1.15
requests                  2.32.5
requests-oauthlib         2.0.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rfc3987-syntax            1.1.0
rich                      15.0.0
rpds-py                   0.27.1
s3transfer                0.16.1
safetensors               0.7.0
scipy                     1.13.1
setuptools                80.9.0
shellingham               1.5.4
simplejson                4.1.1
six                       1.17.0
smmap                     5.0.3
soupsieve                 2.8.3
swagger-spec-validator    3.0.4
sympy                     1.14.0
timm                      1.0.26
torch                     2.7.1+cu118
torchaudio                2.7.1+cu118
torchmetrics              1.8.2
torchvision               0.22.1+cu118
tqdm                      4.67.3
typer                     0.23.2
typing_extensions         4.15.0
tzdata                    2026.2
uri-template              1.3.0
urllib3                   1.26.20
wcwidth                   0.6.0
webcolors                 24.11.1
websocket-client          1.9.0
wheel                     0.47.0
yarl                      1.22.0
zipp                      3.23.1
```
</details> 

详细版本见 `requirements.txt`。

## 7. 团队分工
---

-   成员 吴尧承：负责 CLIP 嵌入提取与损失函数模块
-   成员 刘健韬：负责生成器架构设计与训练循环
-   成员 吴尧承、刘健韬：负责数据集处理与评估指标实现

## 8. 项目架构
---
```text
CLIP
├── CLIP2GAN.py         CLIP + MobileStyleGAN 统一封装类
├── bridgeNetwork.py    Bridge MLP 桥接网络（12层全连接，512→23×512 W+ latent code）
├── lossFunction.py     损失函数模块（L_rec、L_D、L_G、L_lpips、L_div、L_clip、L_reg）
├── training.py         训练模块（消融实验、图像保存、损失记录）
├── LoadDatasets.py     数据集加载模块
├── testing.py          测试模块
├── CLIP/               CLIP 模型与测试
│   └── mobileCLIP/
│       ├── model/      CLIP 预训练权重
│       └── openclip.py CLIP 测试程序
├── StyleGAN/           MobileStyleGAN 代码与权重
│   └── mobileStyleGAN/MobileStyleGAN.pytorch/
│       ├── core/       模型定义（MappingNetwork、SynthesisNetwork、Discriminator）
│       └── model/      预训练权重 (.ckpt)
├── dataset/            CelebA-HQ 数据集
├── results/            训练产物（images、checkpoints、losses.csv、loss_curve.png）
└── 参考文献/           相关论文
```

## Citing
---
```text
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```
```text
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```
```text
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```
```text
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```
```text
@misc{belousov2021mobilestylegan,
      title={MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis},
      author={Sergei Belousov},
      year={2021},
      eprint={2104.04767},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{BELOUSOV2021100115,
      title = {MobileStyleGAN.pytorch: PyTorch-based toolkit to compress StyleGAN2 model},
      journal = {Software Impacts},
      year = {2021},
      issn = {2665-9638},
      doi = {https://doi.org/10.1016/j.simpa.2021.100115},
      url = {https://www.sciencedirect.com/science/article/pii/S2665963821000452},
      author = {Sergei Belousov},
}
```