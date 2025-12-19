from dataset import load_emnist
from models.logistic_regression import train_logistic_regression

train_dataset, test_dataset = load_emnist()
train_logistic_regression(train_dataset, test_dataset)
