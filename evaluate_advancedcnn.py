# evaluate_advancedcnn.py
import torch
from torch.utils.data import DataLoader
from dataset import load_emnist
from models.advanced_cnn import AdvancedCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    _, test_dataset = load_emnist()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = AdvancedCNN().to(device)
    model.load_state_dict(torch.load("advancedcnn_model.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print("AdvancedCNN Test Accuracy:", acc)

    # ⭐ 保存 accuracy
    with open("advancedcnn_test_accuracy.txt", "w") as f:
        f.write(str(acc))

if __name__ == "__main__":
    main()
