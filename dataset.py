import torch
from torchvision import datasets, transforms


# ============ EMNIST 官方方向修复函数（必须保留） ============
# 警告：这个函数只用于处理原始 EMNIST 数据集，不要用于用户输入！
def fix_emnist_orientation(img: torch.Tensor):
    """
    EMNIST 原始数据默认是旋转 + 翻转过的，需要复原成直立状态。
    输入: img tensor, shape (1, H, W)
    """
    # 逆时针旋转 90 度
    img = torch.rot90(img, k=1, dims=[1, 2])
    # 水平翻转
    img = torch.flip(img, dims=[2])
    return img


# ============ 数据加载函数 ============
def load_emnist(augment=False):
    """
    augment=True  → 训练集（开启数据增强：旋转、平移、缩放）
    augment=False → 测试集（仅基础处理）
    """

    if augment:
        # ✅ 训练集：增加数据增强，模拟手写的不规则性
        transform_train = transforms.Compose([
            # 随机旋转 -15 到 15 度
            transforms.RandomRotation(15, fill=0),
            # 随机平移 10% 和缩放 10%
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),

            transforms.ToTensor(),
            # 最后修正原始数据的方向
            transforms.Lambda(fix_emnist_orientation),
        ])
    else:
        # ❌ 测试集：不增强
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(fix_emnist_orientation),
        ])

    # 测试集 transform（始终保持干净）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(fix_emnist_orientation),
    ])

    # --- 加载训练集 ---
    train_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform_train
    )

    # --- 加载测试集 ---
    test_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
        transform=transform_test
    )

    return train_dataset, test_dataset