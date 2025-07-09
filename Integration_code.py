import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np

# ========== CONFIG ==========
MODEL1_PATH = r"F:\TSRS\traffic_sign_modelnew091.pth"  # 6-class model
MODEL2_PATH = r"F:\TSRS\Classifier_modelnew2.pth"  # 3-class condition model

IMG_SIZE1 = 64
IMG_SIZE2 = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
CLASS_LABELS_1 = {
    0: "Speed limit (100km/h)",
    1: "Road Work",
    2: "Go straight or right",
    3: "Traffic Signals Ahead",
    4: "Stop",
    5: "Invalid Sign"
}
CLASS_LABELS_2 = {
    0: "Valid Sign",
    1: "Invalid Sign",
    2: "Damaged Sign"
}

# ========== LOAD MODELS ==========
# Model 1: Sign Classification
model1 = models.resnet18(pretrained=False)
model1.fc = nn.Linear(model1.fc.in_features, 6)
model1.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
model1 = model1.to(DEVICE).eval()

# Model 2: Condition Classification
model2 = models.resnet18(pretrained=True)
model2.fc = nn.Linear(model2.fc.in_features, 3)
model2.load_state_dict(torch.load(MODEL2_PATH, map_location=DEVICE))
model2 = model2.to(DEVICE).eval()

# ========== TRANSFORMS ==========
transform1 = transforms.Compose([
    transforms.Resize((IMG_SIZE1, IMG_SIZE1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
transform2 = transforms.Compose([
    transforms.Resize((IMG_SIZE2, IMG_SIZE2)),
    transforms.ToTensor()
])

# ========== PREDICT FUNCTION ==========
def predict_both_models(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    input1 = transform1(pil_image).unsqueeze(0).to(DEVICE)
    input2 = transform2(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out1 = model1(input1)
        out2 = model2(input2)
        pred1 = torch.argmax(out1, dim=1).item()
        pred2 = torch.argmax(out2, dim=1).item()

    label1 = CLASS_LABELS_1[pred1]
    label2 = CLASS_LABELS_2[pred2]
    return label1, label2

# ========== REAL-TIME CAMERA ==========
cap = cv2.VideoCapture(0)
print("📷 Starting Integrated Prediction... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    try:
        sign_label, condition_label = predict_both_models(frame)
    except Exception as e:
        sign_label = "Error"
        condition_label = str(e)

    # Draw result
    h, w, _ = frame.shape
    label = f"{sign_label} - {condition_label}"
    cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 2)
    cv2.putText(frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("🚦 Sign & Condition Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("combined_prediction.png", frame)
        print("✅ Frame saved as combined_prediction.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited.")
