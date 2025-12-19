# models/logreg_hog.py

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from skimage.feature import hog  # HOG 特征提取

def extract_hog_features(images):
    """
    images: torch tensor, shape (N, 28, 28)
    return numpy array: (N, feature_dim)
    """
    hog_features = []
    for img in images:
        img_np = img.numpy()
        feat = hog(
            img_np,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        hog_features.append(feat)

    return np.array(hog_features)


def train_logreg_hog(train_dataset, test_dataset):
    """
    使用 HOG 特征训练 Logistic Regression
    """

    print("Extracting HOG features (train)...")
    X_train = extract_hog_features(train_dataset.data)
    y_train = train_dataset.targets.numpy()

    print("Extracting HOG features (test)...")
    X_test = extract_hog_features(test_dataset.data)
    y_test = test_dataset.targets.numpy()

    print("Training Logistic Regression (HOG)...")

    clf = LogisticRegression(
        max_iter=300,
        verbose=1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"HOG + Logistic Regression Accuracy: {acc:.4f}")

    # 保存模型给 GUI 用
    joblib.dump(clf, "logreg_hog.pkl")

    return clf, X_test, y_test, y_pred
