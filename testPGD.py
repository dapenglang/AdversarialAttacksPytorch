import sys
import os

import torch
import torch.nn as nn

# 设置正确的路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import models, transforms
from utils import get_imagenet_data, get_pred, load_images_from_input_folder, save_adversarial_results

# 尝试导入PGD攻击方法
try:
    # 优先从本地torchattacks模块导入
    from torchattacks import PGD
except ImportError:
    # 如果失败，尝试从父目录的torchattacks导入
    print("Warning: Using torchattacks from parent directory")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchattacks import PGD

# 主函数
def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载ImageNet分类和标签
    idx2label, MEAN, STD = get_imagenet_data()
    
    # 加载input文件夹中的图片
    try:
        images, labels, filenames = load_images_from_input_folder("./input")
        print(f"[Data loaded] Total images: {len(filenames)}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # 加载预训练模型
    try:
        model = models.resnet18(pretrained=True).to(device).eval()
        print("[Model loaded]")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建PGD攻击实例
    atk = PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    atk.set_normalization_used(mean=MEAN, std=STD)
    print(f"Attack method: {atk}")
    
    # 对每张图片执行攻击
    print("[Starting adversarial attack...]")
    adv_images = []
    
    for i in range(len(images)):
        img = images[i:i+1]  # 获取单张图片
        label = labels[i:i+1]
        filename = filenames[i]
        
        print(f"Attacking image {i+1}/{len(images)}: {filename}")
        
        # 执行攻击
        adv_img = atk(img, label)
        adv_images.append(adv_img)
        
        # 获取预测结果
        original_pred = get_pred(model, img, device)
        adv_pred = get_pred(model, adv_img, device)
        
        # 打印结果
        original_class = idx2label[original_pred.item()] if original_pred.item() < len(idx2label) else 'unknown'
        adv_class = idx2label[adv_pred.item()] if adv_pred.item() < len(idx2label) else 'unknown'
        
        print(f"Original prediction: {original_pred.item()} ({original_class})")
        print(f"Adversarial prediction: {adv_pred.item()} ({adv_class})")
        print(f"Attack successful: {original_pred.item() != adv_pred.item()}")
        print("------------------------")
    
    # 将所有对抗样本合并为一个tensor
    adv_images_tensor = torch.cat(adv_images, dim=0)
    
    # 保存结果到outputs文件夹
    print("[Saving results to outputs folder...]")
    save_adversarial_results(images, adv_images_tensor, filenames, "./outputs")
    print("[All results saved successfully!]")

if __name__ == "__main__":
    main()