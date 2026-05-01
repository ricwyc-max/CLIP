__author__ = 'Eric'

#配置代理
import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# 必须在导入 open_clip 之前设置（配置HF的地址，模型直接下载到./model位置)
os.environ['HF_HOME'] = './model'  # 改成你的目标路径

#导包
import torch
from PIL import Image
import open_clip

#1、载入模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#2、载入图像和文字编码
image = preprocess(Image.open("docs/dog.jpg")).unsqueeze(0)
texts = ["a diagram", "a dog", "a cat"]
text = tokenizer(texts)

#3、进行推理计算
with torch.no_grad(), torch.autocast("cuda"):
    # 使用CLIP模型对图像进行编码，提取图像的特征向量
    image_features = model.encode_image(image)
    # 使用CLIP模型对文本进行编码，提取文本的特征向量
    text_features = model.encode_text(text)

    # 对图像特征向量进行L2归一化，除以它的模长，使其单位长度为1
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # 对文本特征向量进行L2归一化，除以它的模长，使其单位长度为1
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算图像特征与所有文本特征的相似度（缩放100倍后），再进行softmax得到概率分布
    # image_features @ text_features.T 是矩阵乘法，计算余弦相似度（因为已经归一化）
    # 乘以100.0是温度参数，控制softmax输出的分布尖锐程度
    # softmax(dim=-1) 沿最后一维进行softmax，得到每个文本类别的概率
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

# 4、美化输出
print("\n" + "="*50)
print("图像分类结果:")
print("="*50)
for i, (label, prob) in enumerate(zip(texts, text_probs[0])):
    percentage = prob.item() * 100
    bar = "█" * int(percentage / 2)  # 简单的进度条
    print(f"{i+1}. {label:25s} : {percentage:6.2f}% {bar}")
print("="*50)

# 预测结果
pred_idx = text_probs[0].argmax().item()
print(f"\n预测结果: {texts[pred_idx]}")
print(f"置信度: {text_probs[0][pred_idx].item()*100:.2f}%")