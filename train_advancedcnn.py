import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import load_emnist
from models.advanced_cnn import AdvancedCNN
import matplotlib.pyplot as plt
import os


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out.data, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ 重点：开启 augment=True
    train_dataset, test_dataset = load_emnist(augment=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = AdvancedCNN(num_classes=47).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 12 # 建议训练 10 轮以上
    train_acc_list = []
    test_acc_list = []
    loss_list = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss_list.append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "advancedcnn_model.pth")
    print("✅ 模型已保存为 advancedcnn_model.pth")

    # 保存准确率记录
    with open("advancedcnn_test_accuracy.txt", "w") as f:
        f.write(str(test_acc_list[-1]))


if __name__ == "__main__":
    main()