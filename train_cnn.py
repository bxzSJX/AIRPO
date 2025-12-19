import torch
from torch.utils.data import DataLoader
from models.my_cnn import MyCNN
from dataset import load_emnist
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, test_dataset = load_emnist(augment=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = MyCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_acc_list = []
test_acc_list = []
loss_list = []

def evaluate(model):
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

for epoch in range(10):
    model.train()
    total = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    test_acc = evaluate(model)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    loss_list.append(loss.item())

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

torch.save(model.state_dict(), "cnn_model.pth")

# ---- Learning Curve ----
plt.figure(figsize=(10,4))
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("learning_curve.png")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(loss_list, label="Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
plt.show()
