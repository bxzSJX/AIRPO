# generate_model_comparison.py
import matplotlib.pyplot as plt
import re
import os

def read_acc(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    m = re.search(r"Accuracy:\s*([0-9.]+)", first)
    if m:
        return float(m.group(1))
    else:
        return float(first)  # 对 CNN / AdvancedCNN 的 txt 支持

# 读取 4 个模型的准确率
acc_lr = read_acc("logreg_pixels_classification_report.txt")
acc_hog = read_acc("logreg_hog_classification_report.txt")
acc_cnn = read_acc("cnn_test_accuracy.txt")
acc_adv = read_acc("advancedcnn_test_accuracy.txt")

names = ["LR (Pixels)", "HOG + LR", "CNN", "Advanced CNN"]
values = [acc_lr, acc_hog, acc_cnn, acc_adv]

plt.figure(figsize=(8, 5))
plt.bar(names, values, color=["gray", "green", "blue", "orange"])
plt.ylim(0.5, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Comparison (Accuracy)")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
plt.close()

print("Saved model_comparison.png")
