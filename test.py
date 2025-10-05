import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import models, transforms
from utils import get_imagenet_data, get_pred, load_images_from_input_folder

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
    
    # 创建指定的十种攻击方法实例
    attack_methods = create_attack_methods(model, MEAN, STD)
    if not attack_methods:
        print("Failed to create any attack methods")
        return
    
    # 确保outputs文件夹存在
    os.makedirs("./outputs", exist_ok=True)
    
    # 对每张图片执行指定的攻击方法
    print("[Starting adversarial attacks...]")
    for img_idx, (img, label, filename) in enumerate(zip(images, labels, filenames)):
        img_name = os.path.splitext(filename)[0]  # 去掉文件扩展名
        print(f"\nProcessing image {img_idx+1}/{len(filenames)}: {filename}")
        
        # 获取原始预测结果
        original_pred = get_pred(model, img.unsqueeze(0), device)
        original_class = idx2label[original_pred.item()] if original_pred.item() < len(idx2label) else 'unknown'
        print(f"Original prediction: {original_pred.item()} ({original_class})")
        
        # 对该图片执行每种攻击方法
        for attack_name, atk in attack_methods.items():
            try:
                print(f"\nApplying {attack_name} attack...")
                
                # 执行攻击
                adv_img = atk(img.unsqueeze(0), label.unsqueeze(0))
                
                # 获取对抗样本的预测结果
                adv_pred = get_pred(model, adv_img, device)
                adv_class = idx2label[adv_pred.item()] if adv_pred.item() < len(idx2label) else 'unknown'
                
                # 打印攻击结果
                print(f"{attack_name} prediction: {adv_pred.item()} ({adv_class})")
                print(f"Attack successful: {original_pred.item() != adv_pred.item()}")
                
                # 保存结果图片
                save_attack_result(img, adv_img, img_name, attack_name, MEAN, STD)
                
            except Exception as e:
                print(f"Error applying {attack_name}: {e}")
                continue
    
    print("\n[All attacks completed and results saved!]")

# 创建指定的十种攻击方法实例
def create_attack_methods(model, MEAN, STD):
    attack_methods = {}
    attack_classes = {}
    
    # 尝试导入所有需要的攻击方法
    try:
        from torchattacks import BIM, CW, DeepFool, FGSM, JSMA, MIFGSM, OnePixel, PGD, TIFGSM, TPGD
        attack_classes = {
            "BIM": BIM,
            "CW": CW,
            "DeepFool": DeepFool,
            "FGSM": FGSM,
            "JSMA": JSMA,
            "MIFGSM": MIFGSM,
            "OnePixel": OnePixel,
            "PGD": PGD,
            "TIFGSM": TIFGSM,
            "TPGD": TPGD
        }
        print("Successfully imported all specified attack methods")
    except ImportError as e:
        print(f"Error importing attack methods: {e}")
        # 尝试单独导入可用的攻击方法
        try_imports = [
            ("BIM", "BIM"),
            ("CW", "CW"),
            ("DeepFool", "DeepFool"),
            ("FGSM", "FGSM"),
            ("JSMA", "JSMA"),
            ("MIFGSM", "MIFGSM"),
            ("OnePixel", "OnePixel"),
            ("PGD", "PGD"),
            ("TIFGSM", "TIFGSM"),
            ("TPGD", "TPGD")
        ]
        
        for display_name, import_name in try_imports:
            try:
                exec(f"from torchattacks import {import_name}")
                attack_classes[display_name] = locals()[import_name]
                print(f"Successfully imported {display_name}")
            except ImportError:
                print(f"Failed to import {display_name}")
                continue
    
    # 定义攻击参数
    attack_params = {
        "BIM": lambda: attack_classes["BIM"](model, eps=8/255, alpha=2/255, steps=10),
        "CW": lambda: attack_classes["CW"](model, c=1, kappa=0, steps=50),
        "DeepFool": lambda: attack_classes["DeepFool"](model),
        "FGSM": lambda: attack_classes["FGSM"](model, eps=8/255),
        "JSMA": lambda: attack_classes["JSMA"](model, theta=0.1, gamma=0.1, batch_size=1),  # 减小batch_size以降低内存使用
        "MIFGSM": lambda: attack_classes["MIFGSM"](model, eps=8/255, alpha=2/255, steps=10),
        "OnePixel": lambda: attack_classes["OnePixel"](model, pixels=7, steps=100),
        "PGD": lambda: attack_classes["PGD"](model, eps=8/255, alpha=2/255, steps=10, random_start=True),
        "TIFGSM": lambda: attack_classes["TIFGSM"](model, eps=8/255, alpha=2/255, steps=10),
        "TPGD": lambda: attack_classes["TPGD"](model, eps=8/255, alpha=2/255, steps=10)
    }
    
    # 创建攻击实例
    for attack_name, create_func in attack_params.items():
        if attack_name in attack_classes:
            try:
                atk = create_func()
                # 设置归一化参数
                if hasattr(atk, 'set_normalization_used'):
                    atk.set_normalization_used(mean=MEAN, std=STD)
                attack_methods[attack_name] = atk
                print(f"Created {attack_name} attack method")
            except Exception as e:
                print(f"Failed to create {attack_name}: {e}")
                continue
    
    return attack_methods

# 保存攻击结果图片
def save_attack_result(original_img, adv_img, img_name, attack_name, MEAN, STD):
    """保存原始图片、对抗样本和扰动到outputs文件夹，使用图片名_攻击方法名的格式"""
    try:
        # 定义反归一化变换（用于显示）
        inv_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/STD[0], 1/STD[1], 1/STD[2]]),
            transforms.Normalize(mean=[-MEAN[0], -MEAN[1], -MEAN[2]], std=[1., 1., 1.]),
        ])
        
        # 创建一个新的图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 处理原始图片
        original = inv_transform(original_img)
        original = torch.clamp(original, 0, 1)
        axes[0].imshow(np.transpose(original.cpu().numpy(), (1, 2, 0)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 处理对抗样本
        adv = inv_transform(adv_img.squeeze(0))
        adv = torch.clamp(adv, 0, 1)
        axes[1].imshow(np.transpose(adv.cpu().numpy(), (1, 2, 0)))
        axes[1].set_title(f'{attack_name} Adversarial')
        axes[1].axis('off')
        
        # 处理扰动
        perturbation = torch.abs(adv - original)
        perturbation = perturbation / perturbation.max()
        axes[2].imshow(np.transpose(perturbation.cpu().numpy(), (1, 2, 0)))
        axes[2].set_title(f'{attack_name} Perturbation')
        axes[2].axis('off')
        
        # 保存图片，使用图片名_攻击方法名的格式
        output_filename = f"{img_name}_{attack_name}.png"
        output_path = os.path.join("./outputs", output_filename)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved result to: {output_path}")
    except Exception as e:
        print(f"Error saving result for {img_name}_{attack_name}: {e}")

if __name__ == "__main__":
    main()