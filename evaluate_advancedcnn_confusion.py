# evaluate_advancedcnn_confusion.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_emnist
from models.advanced_cnn import AdvancedCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 加载测试集
    _, test_dataset = load_emnist(augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. 加载 AdvancedCNN 模型
    model = AdvancedCNN(num_classes=47).to(device)
    model.load_state_dict(torch.load("advancedcnn_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # 3. 推理所有测试样本
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Accuracy
    acc = (all_preds == all_labels).mean()
    print(f"AdvancedCNN Test Accuracy: {acc:.4f}")

    # 5. Classification Report
    report = classification_report(all_labels, all_preds)
    print(report)

    with open("advancedcnn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 6. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        cmap="Blues",
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title("AdvancedCNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig("confusion_matrix_advancedcnn.png", dpi=200)
    plt.close()

    print("Saved confusion_matrix_advancedcnn.png and advancedcnn_classification_report.txt")


if __name__ == "__main__":
    main()
