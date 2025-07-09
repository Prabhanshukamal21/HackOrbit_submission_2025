import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "/content/drive/MyDrive/HackOrbit/Classifier_modelnew2.pth"  # Trained Model 2
IMAGE_PATH = "/content/drive/MyDrive/HackOrbit/val/1/538.jpeg"                # Test image path
IMG_SIZE = 200
CLASS_NAMES = ['Valid', 'Invalid', 'Damaged']

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Preprocess Image ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# === Predict ===
with torch.no_grad():
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, 1)
    predicted_label = CLASS_NAMES[pred.item()]

# === Draw Bounding Box and Label ===
def draw_prediction(image, label):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    box = [10, 10, width - 10, height - 10]  # Full image box with padding
    draw.rectangle(box, outline="red", width=4)
    font = ImageFont.load_default()
    draw.text((15, 15), label, fill="red", font=font)
    return image

# === Show Result ===
result_img = draw_prediction(image.copy(), predicted_label)
plt.imshow(result_img)
plt.axis('off')
plt.title(f"Predicted: {predicted_label}")
plt.show()
