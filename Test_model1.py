import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# === Configuration ===
NUM_CLASSES = 6
MODEL_PATH = "/content/drive/MyDrive/HackOrbit/traffic_sign_modelnew091.pth"
CLASS_LABELS = {
    0: "Speed limit (100km/h)",
    1: "Road Work",
    2: "Go straight or right",
    3: "Traffic Signals Ahead",
    4: "Stop",
    5: "Invalid Sign"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_and_draw(image_path):
    # Load and transform image
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        label = CLASS_LABELS[predicted_class.item()]

    # Draw bounding box (entire image assumed as sign)
    draw = ImageDraw.Draw(original_image)
    W, H = original_image.size
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline="red", width=4)

    # Add label text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), label, fill="white", font=font)

    # Show image with prediction
    plt.imshow(original_image)
    plt.axis("off")
    plt.title(f"Predicted: {label}")
    plt.show()

    # Save image if needed
    original_image.save("predicted_with_box.png")
    print(f"✅ Saved: predicted_with_box.png")

# === Example Usage ===
test_image_path = "/content/drive/MyDrive/HackOrbit/val/2/36_1040_1577671990.5089772.png"  # Change this
predict_and_draw(test_image_path)
