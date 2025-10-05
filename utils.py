import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# 修改get_imagenet_data函数，使其只加载分类和标签
def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # 加载ImageNet分类索引
    try:
        class_idx = json.load(open("./data/imagenet_class_index.json"))
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        print(f"Loaded ImageNet class labels, total classes: {len(idx2label)}")
    except FileNotFoundError:
        # 如果找不到文件，使用默认的分类映射
        print("Warning: imagenet_class_index.json not found, using predefined class labels")
        idx2label = ['unknown'] * 1000  # 假设1000个分类
    
    return idx2label, MEAN, STD

# 新增函数：加载input文件夹中的图片
def load_images_from_input_folder(folder_path="./input"):
    """加载input文件夹中的所有图片并返回处理后的图片和默认标签"""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")
    
    # 定义图片转换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    images = []
    filenames = []
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in image_extensions:
            file_path = os.path.join(folder_path, file)
            try:
                # 打开并处理图片
                img = Image.open(file_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
                images.append(img_tensor)
                filenames.append(file)
                print(f"Loaded image: {file}")
            except Exception as e:
                print(f"Error loading image {file}: {e}")
    
    if not images:
        raise ValueError(f"No valid images found in {folder_path}")
    
    # 创建默认标签（这里使用0作为默认标签）
    labels = torch.zeros(len(images), dtype=torch.long)
    
    # 将所有图片合并为一个tensor
    images_tensor = torch.cat(images, dim=0)
    
    return images_tensor, labels, filenames

# 新增函数：保存对抗样本结果到outputs文件夹
def save_adversarial_results(original_images, adv_images, filenames, output_folder="./outputs"):
    """保存原始图片、对抗样本和扰动到outputs文件夹"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 定义反归一化变换（用于显示）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/STD[0], 1/STD[1], 1/STD[2]]),
        transforms.Normalize(mean=[-MEAN[0], -MEAN[1], -MEAN[2]], std=[1., 1., 1.]),
    ])
    
    for i, (original, adv, filename) in enumerate(zip(original_images, adv_images, filenames)):
        # 创建一个新的图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图片
        original = inv_transform(original)
        original = torch.clamp(original, 0, 1)
        axes[0].imshow(np.transpose(original.cpu().numpy(), (1, 2, 0)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 显示对抗样本
        adv = inv_transform(adv)
        adv = torch.clamp(adv, 0, 1)
        axes[1].imshow(np.transpose(adv.cpu().numpy(), (1, 2, 0)))
        axes[1].set_title('Adversarial Example')
        axes[1].axis('off')
        
        # 显示扰动
        perturbation = torch.abs(adv - original)
        perturbation = perturbation / perturbation.max()  # 归一化以便更好地显示
        axes[2].imshow(np.transpose(perturbation.cpu().numpy(), (1, 2, 0)))
        axes[2].set_title('Perturbation')
        axes[2].axis('off')
        
        # 保存图片
        output_path = os.path.join(output_folder, f"adversarial_{filename}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved adversarial result to: {output_path}")

# 保留原有的其他函数
def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2


@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return (100 * float(correct) / total)
