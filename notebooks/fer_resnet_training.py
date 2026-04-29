import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =====================
# 0. SET WORKING DIRECTORY
# =====================

os.chdir("/Users/parandcurly/Desktop/computer vision")
print("Current folder:", os.getcwd())

# =====================
# 1. TRANSFORMS
# =====================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# 2. LOAD DATASET
# =====================

train_dataset = ImageFolder("archive/train", transform=train_transform)
test_dataset  = ImageFolder("archive/test", transform=val_test_transform)

print("Classes:", train_dataset.classes)

# =====================
# 3. TRAIN / VALIDATION SPLIT
# =====================

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# remove augmentation from validation
val_dataset.dataset.transform = val_test_transform

# =====================
# 4. DATALOADERS (FIXED)
# =====================

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# =====================
# 5. MODEL (CPU SAFE)
# =====================

device = torch.device("cpu")
print("Using device:", device)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 7)
model = model.to(device)

# =====================
# 6. LOSS + OPTIMIZER
# =====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================
# 7. TRAINING LOOP
# =====================

num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_correct, train_total = 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ===== VALIDATION =====
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

print("\n Training finished!")

