import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import model
import os

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 111
        output = model(data)
        loss = criterion(output, target)
        loss.backward()   # 111
        optimizer.step()  # 111
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    transform = dataset.transforms.Compose([
        dataset.transforms.ToTensor(),
        dataset.transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = dataset.MNISTDataset("./data", train=True, transform=transform)
    test_dataset = dataset.MNISTDataset("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型
    model = model.MNISTModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和测试模型
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

        # 保存模型
        model_path = f'./model/mnist_model_epoch_cnn_{epoch}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')