import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------------- DATA ----------------
data_dir = "gesture1/asl_dataset"  # Folder with 36 classes (0-9, a-z)
IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(dataset.classes)
print(f"Detected {num_classes} classes.")

train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------- MODEL ----------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 8

# ---------------- TRAINING ----------------
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# ---------------- TEST ----------------
model.eval()
correct, total = 0,0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs,1)
        correct += (predicted==labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {(correct/total)*100:.2f}%")

import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Number of random samples to check
num_samples = 5

for _ in range(num_samples):
    idx = random.randint(0, len(dataset)-1)  # pick a random index
    img, label = dataset[idx]
    
    # Print info
    print(f"Index: {idx}, Image shape: {img.shape}, Label: {label} ({dataset.classes[label]})")
    
    # Show the image (convert tensor to PIL for visualization)
    plt.imshow(TF.to_pil_image(img))
    plt.title(f"Label: {dataset.classes[label]}")
    plt.axis('off')
    plt.show()


# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "resnet18_handsign_color.pth")
print("✅ Model saved as resnet18_handsign_color.pth")
