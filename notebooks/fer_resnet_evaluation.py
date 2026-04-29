import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18, ResNet18_Weights



# =====================
# 1. Set working directory (check current folder)
# =====================

os.chdir(os.path.dirname(os.path.abspath(__file__))) //PATH ASSOLUTO
print("Current folder:", os.getcwd())

# =====================
# 2. Load the best model
# =====================

# Set device to CPU (or GPU if available)
device = torch.device("cpu")  # Change to "cuda" if you have GPU
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 classes (emotions)

# Load the best model from the saved .pth file
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()  # Set to evaluation mode

# =====================
# 3. Prepare the test dataset and dataloader
# =====================

test_dataset = ImageFolder("archive/test", transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
]))

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# =====================
# 4. Collect predictions and labels
# =====================

all_preds = []
all_labels = []

with torch.no_grad():  # Turn off gradients for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()  # Get predicted labels

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())  # Convert to CPU for metrics

# =====================
# 5. Classification Report
# =====================

# Print classification report (precision, recall, F1-score)
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# =====================
# 6. Confusion Matrix
# =====================

# Generate and plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=test_dataset.classes, 
            yticklabels=test_dataset.classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================
# 7. Test Accuracy (optional)
# =====================

# Calculate test accuracy
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")

import os
os.chdir("/Users/parandcurly/Desktop/computer vision")
