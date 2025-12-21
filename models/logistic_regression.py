# models/logistic_regression.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_logistic_regression(train_dataset, test_dataset):
    X_train = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1)
    y_test = test_dataset.targets.numpy()

    logreg = LogisticRegression(max_iter=200, verbose=1)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", acc)

    joblib.dump(logreg, "logreg_model.pkl")

    return logreg
