from dataset import load_emnist
from models.logreg_hog import train_logreg_hog

def main():
    train_dataset, test_dataset = load_emnist(augment=False)
    train_logreg_hog(train_dataset, test_dataset)

if __name__ == "__main__":
    main()
