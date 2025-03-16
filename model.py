import torch.nn as nn
import torch.nn.functional as F

class MNISTModel_(nn.Module):
    def __init__(self):
        super(MNISTModel_, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层到隐藏层
        self.ac1 = nn.ReLU()                 # 激活函数 ReLU
        self.fc2 = nn.Linear(512, 256)     # 隐藏层到隐藏层
        self.fc3 = nn.Linear(256, 10)      # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = self.ac1(self.fc1(x))  # 激活函数 ReLU
        x = self.ac1(self.fc2(x))  # 激活函数 ReLU
        x = self.fc3(x)          # 输出层
        return x


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道数为1，输出通道数为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道数为32，输出通道数为64，卷积核大小为3x3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层，池化核大小为2x2

        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入大小为64*7*7，输出大小为128
        self.fc2 = nn.Linear(128, 10)  # 输入大小为128，输出大小为10（对应10个类别）

    def forward(self, x):
        # 卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积 + ReLU + 池化

        # 展平
        x = x.view(-1, 64 * 7 * 7)  # 展平

        # 全连接层 + 激活函数
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU
        x = self.fc2(x)  # 第二层全连接（输出层）
        return x