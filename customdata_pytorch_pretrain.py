import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ DATA TRANSFORMS ------------------
train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ------------------ DATASET ------------------
data_dir = 'antsnbees'  # Replace with your dataset path
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
}

train_loader = DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
val_loader = DataLoader(image_datasets['val'], batch_size=32, shuffle=False)

# ------------------ MODEL ------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 2 classes (ants & bees)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# ------------------ LOSS & OPTIMIZER ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------ TRAINING ------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# ------------------ VALIDATION ------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {(correct/total)*100:.2f}%")

# ------------------ SAVE MODEL ------------------
torch.save(model.state_dict(), "resnet18_model.pth")
print("Model saved as resnet18_model.pth")
