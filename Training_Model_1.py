import os
import shutil
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image, UnidentifiedImageError
import warnings

# Configuration
SOURCE_DIR = "/content/drive/MyDrive/HackOrbit/model1_dataset/5_Classes"   #path to our dataset
CSV_PATH = "/content/drive/MyDrive/HackOrbit/model1_dataset/New 5 labels.csv"    #path to our .csv file
OUTPUT_DIR = "/content/drive/MyDrive/HackOrbit"    #path to save the splitted dataset
NUM_CLASSES = 6
BATCH_SIZE = 16
EPOCHS = 35
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# Create output dirs
for split in ['train', 'val', 'test']:
    for cls in range(NUM_CLASSES):
        Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Collect valid images and labels
image_paths = []
for cls in range(NUM_CLASSES):
    class_dir = os.path.join(SOURCE_DIR, str(cls))
    for file in os.listdir(class_dir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(class_dir, file)
            try:
                with Image.open(full_path) as img:
                    img.verify()  # Verify image integrity
                image_paths.append((full_path, cls))
            except (UnidentifiedImageError, OSError) as e:
                warnings.warn(f"⚠️ Skipping corrupted image: {full_path} | {e}")

# Split dataset
train_val, test = train_test_split(image_paths, test_size=0.1, stratify=[i[1] for i in image_paths], random_state=SEED)
train, val = train_test_split(train_val, test_size=0.2, stratify=[i[1] for i in train_val], random_state=SEED)

# Copy files to respective folders
def copy_files(file_list, split):
    for path, label in file_list:
        dest = f"{OUTPUT_DIR}/{split}/{label}/{os.path.basename(path)}"
        try:
            shutil.copy2(path, dest)
        except Exception as e:
            warnings.warn(f"⚠️ Failed to copy image: {path} | {e}")

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

# Define SafeImageFolder
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            img_path = self.imgs[index][0]
            warnings.warn(f"⚠️ Skipping unreadable image at index {index}: {img_path} | {e}")
            dummy_img = torch.zeros(3, 64, 64)
            dummy_label = -1
            return dummy_img, dummy_label

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load datasets with SafeImageFolder
train_ds = SafeImageFolder(f"{OUTPUT_DIR}/train", transform=transform)
val_ds = SafeImageFolder(f"{OUTPUT_DIR}/val", transform=transform)
test_ds = SafeImageFolder(f"{OUTPUT_DIR}/test", transform=transform)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
device = torch.device("cpu")
model = model.to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_dl:
        # Skip dummy labels (-1)
        mask = labels != -1
        if not mask.any():
            continue
        imgs, labels = imgs[mask], labels[mask]
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss / len(train_dl):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        mask = labels != -1
        if not mask.any():
            continue
        imgs, labels = imgs[mask], labels[mask]
        imgs = imgs.to(device)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.cpu().tolist())

# Print report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# Save the model
torch.save(model.state_dict(), "/content/drive/MyDrive/HackOrbit/traffic_sign_modelnew091.pth")
print("✅ Model saved as 'traffic_sign_modelnew091.pth'")
