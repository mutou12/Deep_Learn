from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # 加载 MNIST 数据集
        self.data = MNIST(root=self.root, train=self.train, download=True)
        self.x = [1,3,4,5]
        self.y = [2,4,5,6]

    def __len__(self):
        return len(self.data)
        # return len(self.x)

    def __getitem__(self, index):
        img, label = self.data[index]

        # 如果有 transform，则应用 transform
        if self.transform:
            img = self.transform(img)

        return img, label

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将像素值从 [0, 255] 缩放到 [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
])

# 创建自定义数据集
train_dataset = MNISTDataset(root='./data', train=True, transform=transform)
test_dataset = MNISTDataset(root='./data', train=False, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 示例：打印一些数据
if __name__ == "__main__":
    for images, labels in train_loader:
        print(f"Images batch shape: {images.size()}")
        print(f"Labels batch shape: {labels.size()}")
        break