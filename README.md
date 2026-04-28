# 项目名称：基于 CLIP 引导的文本生成图像大模型

> **课程**：生成式人工智能
> **状态**：进行中 / 基础架构搭建

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
- **使用基本GAN网络进行图像生成、CLIP进行引导(CAFE-GAN, RATLIP等)**
  - 具体方案：

- **使用Style-GAN网络进行图像生成、CLIP进行引导（CLIP2GAN）**
  - 具体方案：

- **使用Stable Diffusion网络进行图像生成、CLIP进行引导（ DALL·E 2 、DiffusionCLIP ）**
  - 具体方案：

- **使用Diffusion GAN网络进行图像生成、CLIP进行引导（UFOGen 、clip2latent ）**
  - 具体方案：

- **使用Transformer网络进行图像生成、CLIP进行引导（DALL·E）**
  - 具体方案：

- **使用mobileStyleGAN网络进行图像生成、mobileCLIP网络进行图像引导**
  - 具体方案：
1、使用预训练mobileStyleGAN进行图像生成，预训练mobileStyleGAN权重获取参考地址：https://github.com/bes-dev/MobileStyleGAN.pytorch  
2、使用预训练mobileCLIP进行文本编码，预训练mobileCLIP权重获取参考地址：https://github.com/mlfoundations/open_clip  
3、自建 Bridge MLP 隐空间向量语义转换层  
4、进行不同损失函数的消融实验  

  - 模型架构：



**核心流程**：
1.  **文本编码**：输入 Prompt \( \rightarrow \) CLIP Text Encoder \( \rightarrow \) 文本嵌入 \( T \)
2.  **图像生成**：随机噪声 \( z \) + 文本嵌入 \( T \) \( \rightarrow \) 生成器（Diffusion/Transformer） \( \rightarrow \) 生成图像 \( I_{gen} \)
3.  **语义对齐**：CLIP Image Encoder 提取 \( I_{gen} \) 的特征 \( V \)，与文本特征 \( T \) 计算 **CLIP-Score**，作为语义损失回传。
4. **消融实验**：对不同的编码器和生成器组合进行消融实验，同时对不同损失函数进行消融实验，了解损失函数对于模型的具体影响。
5. **尝试完成文本-图像编辑**：尝试测试文本编辑图像功能，了解其功能边界。

**基线模型参考**：
-   （例如：DALL·E 2， Stable Diffusion， 或 简单的 Conditional GAN + CLIP）
- CLIP = 判别式模型（理解/匹配）
- DALL·E / Stable Diffusion / Style-GAN = 生成式模型（创造/生成）
- 用 CLIP 作为“导师”来训练一个生成式模型

## 4. 实验计划

-   **数据集**：（待补充：如 MS-COCO， CUB-200， 或自定义数据集）
-   **评估指标**：FID（图像质量）、CLIP-Score（语义一致性）、人工评估
-   **对比实验**：CLIP 引导的不同质量生成器对比
-   **消融实验**：无 CLIP 引导的生成器 vs. 有 CLIP 引导的生成器

## 5. 当前进度

-   [ ] 文献调研：理解 CLIP 的对比学习原理及在文生图中的应用
-   [ ] 环境配置与 CLIP 预训练权重加载
-   [ ] 搭建生成器骨架
-   [ ] 实现 CLIP 引导的损失函数
-   [ ] 训练与调试
-   [ ] 探索更多的生成器和编码器类型

## 6. 如何运行

*（待补充：环境安装、数据准备、训练命令、推理示例）*

```bash
git clone [你们的仓库地址]
cd [项目目录]
pip install -r requirements.txt
python train.py --config configs/clip_guidance.yaml
```

## Requirements
---

主要依赖：
- 等待完善


其他依赖（根据对比实验按需安装）：
- 等待完善


详细版本见 `requirements.txt`。

## 7. 团队分工
---

-   成员 吴尧承：负责 CLIP 嵌入提取与损失函数模块
-   成员 刘健韬：负责生成器架构设计与训练循环
-   成员 吴尧承、刘健韬：负责数据集处理与评估指标实现



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