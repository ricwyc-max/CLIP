__author__ = 'Eric'

# ============================================================================
#
#                           [Bridge MLP]网络模块
#
# 12-layer fully connected layer with each layer post-connected to the activation layer Leaky ReLU
#
#
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bridge_MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super(Bridge_MLP, self).__init__()

        # 定义层（在 __init__ 中）
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.fc5 = nn.Linear(hidden_dim,hidden_dim)
        self.fc6 = nn.Linear(hidden_dim,hidden_dim)
        self.fc7 = nn.Linear(hidden_dim,hidden_dim)
        self.fc8 = nn.Linear(hidden_dim,hidden_dim)
        self.fc9 = nn.Linear(hidden_dim,hidden_dim)
        self.fc10 = nn.Linear(hidden_dim,hidden_dim)
        self.fc11 = nn.Linear(hidden_dim,hidden_dim)
        self.fc12 = nn.Linear(hidden_dim,output_dim)

        #激活函数
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # 12层全连接 + LeakyReLU
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.fc2(x)
        x = self.leakyRelu(x)
        x = self.fc3(x)
        x = self.leakyRelu(x)
        x = self.fc4(x)
        x = self.leakyRelu(x)
        x = self.fc5(x)
        x = self.leakyRelu(x)
        x = self.fc6(x)
        x = self.leakyRelu(x)
        x = self.fc7(x)
        x = self.leakyRelu(x)
        x = self.fc8(x)
        x = self.leakyRelu(x)
        x = self.fc9(x)
        x = self.leakyRelu(x)
        x = self.fc10(x)
        x = self.leakyRelu(x)
        x = self.fc11(x)
        x = self.leakyRelu(x)
        x = self.fc12(x)

        # PixelNorm：约束输出范数，保持与 MobileStyleGAN W 空间分布一致
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x

if __name__ == "__main__":
    # 使用
    model = Bridge_MLP()
    print(model)
