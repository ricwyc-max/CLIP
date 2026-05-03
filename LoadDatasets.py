__author__ = 'Eric'

# ============================================================================
#
#                           数据集加载模块
#
#                         加载数据集并进行初步预处理
#
# ============================================================================


#=================================导包=======================================
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
# ============================================================================

class MyCustomDataset(Dataset):
    """
    数据集加载处理类
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 直接获取所有图片文件路径
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image  # 只返回图片，没有标签