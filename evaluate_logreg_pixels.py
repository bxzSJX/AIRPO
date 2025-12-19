# evaluate_logreg_pixels.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from dataset import load_emnist

def main():
    # 1. 载入测试集
    _, test_dataset = load_emnist(augment=False)

    print("Preparing pixel features for test set...")
    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1) / 255.0
    y_test = test_dataset.targets.numpy()

    # 2. 加载 Logistic Regression 像素模型
    clf = joblib.load("logreg_model.pkl")

    # 3. 预测
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[Pixel Logistic Regression] Test Accuracy: {acc:.4f}")

    # 4. classification report（precision / recall / f1）
    report = classification_report(y_test, y_pred)
    print(report)

    # 保存文本
    with open("logreg_pixels_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 5. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm,
                cmap="Blues",
                cbar=True,
                xticklabels=False,
                yticklabels=False)
    plt.title("Logistic Regression (Pixels) Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_logreg.png", dpi=200)
    plt.close()

    print("Saved confusion_matrix_logreg.png and logreg_pixels_classification_report.txt")

if __name__ == "__main__":
    main()
