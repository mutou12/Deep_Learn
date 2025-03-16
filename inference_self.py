import torch
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import model

def load_model(model_path, device):
    model_ = model.MNISTModel().to(device)
    model_.load_state_dict(torch.load(model_path, map_location=device))
    model_.eval()
    return model_

def infer_single_image(model, device, image_path):
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图像是灰度图
        transforms.Resize((28, 28)),  # 调整图像大小为 28x28
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
    ])

    # 加载图像
    image = Image.open(image_path)
    image = ImageOps.invert(image)  # 反转颜色
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备

    # 显示送入模型的图片
    plt.imshow(image.cpu().squeeze().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    plt.show()

    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)

    return pred.item()

def plot_image(image_path, true_label, pred_label):
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.title(f'Pred: {pred_label}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = 'model/mnist_model_epoch_cnn_10.pth'  # 请确保模型路径正确
    model = load_model(model_path, device)

    # 手写数字图片路径
    image_path = 'image.png'  # 请确保图片路径正确
    true_label = 8  # 真实标签（如果已知）

    # 推理
    pred_label = infer_single_image(model, device, image_path)

    # 显示结果
    plot_image(image_path, true_label, pred_label)