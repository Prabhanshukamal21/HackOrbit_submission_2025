import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DATASET_DIR = "/content/drive/MyDrive/HackOrbit/model 2 Dataset/3_Classes"  # Dataset path
OUTPUT_DIR = "/content/drive/MyDrive/HackOrbit"                             # save the splitted Dataset path
MODEL_PATH = "/content/drive/MyDrive/HackOrbit/Classifier_modelnew2.pth"    # path to save the model
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 30
IMG_SIZE = 200
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 1. SPLIT DATA =====================
def prepare_dataset():
    print("\U0001F4C2 Preparing dataset...")
    image_paths = []
    for cls in range(NUM_CLASSES):
        class_dir = os.path.join(DATASET_DIR, str(cls))
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((os.path.join(class_dir, file), cls))

    train_val, test = train_test_split(image_paths, test_size=0.1, stratify=[i[1] for i in image_paths], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.2, stratify=[i[1] for i in train_val], random_state=SEED)

    for split, data in zip(['train', 'val', 'test'], [train, val, test]):
        for cls in range(NUM_CLASSES):
            Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)
        for src, label in data:
            dest = f"{OUTPUT_DIR}/{split}/{label}/{os.path.basename(src)}"
            shutil.copy2(src, dest)
    print("✅ Dataset split complete.")

# ===================== 2. TRAIN MODEL =====================
def train_model():
    print("\U0001F3CB\ufe0f Training model...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(f"{OUTPUT_DIR}/train", transform=transform)
    val_ds = datasets.ImageFolder(f"{OUTPUT_DIR}/val", transform=transform)
    test_ds = datasets.ImageFolder(f"{OUTPUT_DIR}/test", transform=transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_dl):.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    prepare_dataset()
    train_model()
