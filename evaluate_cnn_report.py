import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_emnist
from models.my_cnn import MyCNN

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 加载数据
_, test_dataset = load_emnist(augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载 CNN 模型
model = MyCNN(num_classes=47).to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算 accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"[CNN] Test Accuracy: {acc:.4f}")

# 生成分类报告
report = classification_report(all_labels, all_preds)
print(report)

# 保存到 txt
with open("cnn_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    cbar=True,
    xticklabels=False,
    yticklabels=False
)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_cnn.png", dpi=200)
plt.close()

print("Saved cnn_classification_report.txt and confusion_matrix_cnn.png")
