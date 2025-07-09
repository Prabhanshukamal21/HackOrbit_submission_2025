import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np

# ======= CONFIG =======
NUM_CLASSES = 3
MODEL_PATH = r"F:\TSRS\Classifier_modelnew2.pth"  # Update path
IMG_SIZE = 200
CLASS_LABELS = {
    0: "Valid Sign",
    1: "Invalid Sign",
    2: "Damaged Sign"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= LOAD MODEL =======
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ======= TRANSFORMS =======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ======= PREDICT FROM FRAME =======
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = CLASS_LABELS[predicted.item()]
    return label

# ======= START WEBCAM =======
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit or 's' to save a frame.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to read from webcam.")
        break

    try:
        label = predict_frame(frame)
    except Exception as e:
        label = "Prediction Error"
        print(f"⚠️ Error: {e}")

    # Draw bounding box (full frame)
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 2)
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Traffic Sign Condition Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("captured_model2_output.png", frame)
        print("✅ Saved as captured_model2_output.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited.")
