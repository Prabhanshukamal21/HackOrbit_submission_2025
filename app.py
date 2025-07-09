import os
import threading
import base64
import re
from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms
import pyttsx3

# === CONFIG ===
UPLOAD_FOLDER = 'static/uploads'
MODEL1_PATH = r'F:\TSRS\traffic_sign_modelnew091.pth'  # 6 classes
MODEL2_PATH = r'F:\TSRS\Classifier_modelnew2.pth'  # 3 classes
IMG_SIZE1 = 64
IMG_SIZE2 = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === CLASS LABELS ===
CLASS_LABELS_1 = {
    0: "Speed limit 100 kilometers per hour",
    1: "Road Work",
    2: "Go straight or right",
    3: "Traffic Signals Ahead",
    4: "Stop",
    5: "Not Defined Sign"
}
CLASS_LABELS_2 = {
    0: "Valid Sign",
    1: "Invalid Sign",
    2: "Damaged Sign"
}

# === LOAD MODELS ===
model1 = models.resnet18(pretrained=False)
model1.fc = nn.Linear(model1.fc.in_features, 6)
model1.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
model1 = model1.to(DEVICE).eval()

model2 = models.resnet18(pretrained=True)
model2.fc = nn.Linear(model2.fc.in_features, 3)
model2.load_state_dict(torch.load(MODEL2_PATH, map_location=DEVICE))
model2 = model2.to(DEVICE).eval()

# === TRANSFORMS ===
transform1 = transforms.Compose([
    transforms.Resize((IMG_SIZE1, IMG_SIZE1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
transform2 = transforms.Compose([
    transforms.Resize((IMG_SIZE2, IMG_SIZE2)),
    transforms.ToTensor()
])

# === VOICE FEEDBACK ===
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# === INFERENCE FUNCTION ===
def predict_models(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"❌ Error: {e}")
        return "Invalid image", "Cannot process"

    input1 = transform1(img).unsqueeze(0).to(DEVICE)
    input2 = transform2(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred1 = torch.argmax(model1(input1), dim=1).item()
        pred2 = torch.argmax(model2(input2), dim=1).item()

    label1 = CLASS_LABELS_1.get(pred1, "Unknown")
    label2 = CLASS_LABELS_2.get(pred2, "Unknown")
    speak(f"{label1} — {label2}")
    return label1, label2

# === ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(file.stream)
                img.verify()
                file.stream.seek(0)
            except Exception:
                return render_template('index.html', prediction="Invalid image uploaded")

            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label1, label2 = predict_models(filepath)
            prediction_text = f"{label1} - {label2}"
            return render_template('index.html', prediction=prediction_text, image_filename=filename)
        else:
            return render_template('index.html', prediction="Please upload a valid image file (.jpg/.png)")
    return render_template('index.html', prediction=None)

@app.route('/capture', methods=['POST'])
def capture():
    data = request.form['image_data']
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = base64.b64decode(img_str)

    filename = "webcam_capture.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    label1, label2 = predict_models(filepath)
    prediction_text = f"{label1} - {label2}"
    return render_template('index.html', prediction=prediction_text, image_filename=filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)