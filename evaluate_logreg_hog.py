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
from models.logreg_hog import extract_hog_features

EMNIST_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

def main():
    _, test_dataset = load_emnist(augment=False)

    print("Extracting HOG features for test set...")
    X_test = extract_hog_features(test_dataset.data)
    y_test = test_dataset.targets.numpy()

    clf = joblib.load("logreg_hog.pkl")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[HOG + LR] Test Accuracy: {acc:.4f}")

    # classification report（precision / recall / f1）
    report = classification_report(y_test, y_pred)
    print(report)
    with open("logreg_hog_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm,
                cmap="Blues",
                cbar=True,
                xticklabels=False,
                yticklabels=False)
    plt.title("HOG + Logistic Regression Confusion Matrix (EMNIST Balanced)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_hog.png", dpi=200)
    plt.close()

    print("Saved confusion_matrix_hog.png and logreg_hog_classification_report.txt")

if __name__ == "__main__":
    main()
