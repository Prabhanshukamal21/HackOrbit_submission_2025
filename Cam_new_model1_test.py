import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# === Configuration ===
NUM_CLASSES = 6
MODEL_PATH = r"F:\TSRS\traffic_sign_modelnew091.pth"  # Update if needed
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

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Utility: Prediction from Frame ===
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return CLASS_LABELS[predicted.item()]

# === Real-Time Detection ===
cap = cv2.VideoCapture(0)  # Use 0 for default camera

print("📷 Starting webcam... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Predict traffic sign
    try:
        label = predict_frame(frame)
    except Exception as e:
        label = "Prediction Error"
        print(f"⚠️ {e}")

    # Draw full-frame bounding box (placeholder for detection)
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Traffic Sign Detection", frame)

    # Press 's' to save the frame
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("captured_prediction.png", frame)
        print("✅ Frame saved as captured_prediction.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting.")
