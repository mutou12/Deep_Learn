import os
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 下载 MNIST 数据集
data_path = './data'
mnist_dataset = MNIST(root=data_path, train=True, download=True)
mnist_dataset = MNIST(root=data_path, train=False, download=True)

# 创建一个字典来存储每个数字的图像
digit_images = {i: None for i in range(10)}

# 遍历数据集，找到每个数字的一个示例
for img, label in mnist_dataset:
    if digit_images[label] is None:
        digit_images[label] = img
    if all(digit_images.values()):
        break

# 显示每个数字的图像
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(digit_images[i], cmap='gray')
    ax.set_title(f'Digit: {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()