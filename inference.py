import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import dataset
import model

def load_model(model_path, device):
    model_ = model.MNISTModel().to(device)
    model_.load_state_dict(torch.load(model_path, map_location=device))
    model_.eval()
    return model_

def infer(model, device, data_loader):
    model.eval()
    images, true_labels, pred_labels = [], [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            images.extend(data.cpu())
            true_labels.extend(target.cpu())
            pred_labels.extend(pred.cpu())
            if len(images) >= 10:  # 只取前10个样本进行展示
                break
    return images, true_labels, pred_labels

def plot_images(images, true_labels, pred_labels):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        img = images[i].view(28, 28).numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_labels[i].item()}, Pred: {pred_labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    transform = dataset.transforms.Compose([
        dataset.transforms.ToTensor(),
        dataset.transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = dataset.MNISTDataset("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载模型
    model_path = 'model/mnist_model_epoch_10.pth'  # 请确保模型路径正确
    model = load_model(model_path, device)

    # 推理
    images, true_labels, pred_labels = infer(model, device, test_loader)

    # 显示结果
    plot_images(images, true_labels, pred_labels)