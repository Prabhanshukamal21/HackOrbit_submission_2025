# HackOrbit_submission_2025
# Project Description
This AI-powered system is designed to enhance traffic awareness and safety by leveraging computer vision and deep learning. It integrates two robust image classification models into a user-friendly web application built with Flask.

Key Capabilities:
Model 1: Classifies traffic signs into 6 common categories using a CNN-based architecture (ResNet18).
Model 2: Detects the condition of the sign (valid, invalid, or damaged) to assist with infrastructure monitoring and real-time decision-making.

# 🚦 Traffic Sign Recognition and Condition Classifier System Web App

This is a Flask-based web application that integrates two deep learning models to:

* Classify **traffic signs** (Model 1)
* Identify the **condition** of the traffic sign (Model 2)
* Provide **voice feedback** of the predictions
* Allow input via **image upload** or **live webcam capture**

---

## 📁 Folder Structure

```
.
├── app.py                  # Main Flask application
├── templates/
│   └── index.html          # HTML front-end (interactive and webcam-enabled)
├── static/uploads/        # Stores uploaded or captured images
├── traffic_sign_model1.pth  # Trained PyTorch model (Model 1)
├── traffic_sign_model2.pth  # Trained PyTorch model (Model 2)
└── README.md
```

---

## 🧠 Model 1: Traffic Sign Classifier

* **Architecture**: Modified ResNet18
* **Input Size**: 200x200 pixels
* **Number of Classes**: 6

### 🔢 Classes in Model 1:

| Class ID | Class Label                         |
| -------- | ----------------------------------- |
| 0        | Speed limit 100 kilometers per hour |
| 1        | Road Work                           |
| 2        | Go straight or right                |
| 3        | Traffic Signals Ahead               |
| 4        | Stop                                |
| 5        | Not Defined Sign                    |

---

## 🧠 Model 2: Traffic Sign Condition Classifier

* **Architecture**: Modified ResNet18
* **Input Size**: 200x200 pixels
* **Number of Classes**: 3

### 🔢 Classes in Model 2:

| Class ID | Class Label  |
| -------- | ------------ |
| 0        | Valid Sign   |
| 1        | Invalid Sign |
| 2        | Damaged Sign |

---

## 🧪 Features

* ✅ Image Upload for prediction
* 🎥 Live Webcam Capture and Classification
* 🔊 Text-to-Speech prediction feedback
* 🔁 Works on CPU

---

## ⚙️ How to Run

1. Ensure Python 3.10 or later is installed.
2. Install dependencies:

```bash
pip install flask torchvision torch pyttsx3 pillow
```

3. Place your `traffic_sign_model1.pth` and `traffic_sign_model2.pth` files in the project root.
4. Run the app:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

---

## 💬 Voice Feedback

* Uses `pyttsx3` to read out both predictions.
* Automatically skips if TTS loop is already running.

---

## 📷 Webcam Feature

* Opens camera in browser
* User captures image using `📸 Capture & Predict`
* Image is sent to backend, predicted, and displayed

---

## 📝 Notes

* Ensure webcam access is allowed in browser.
* Make sure `static/uploads/` is writeable.
* Run from terminal to avoid IDE threading issues.
* This project is an Prototype and only able to predict the above 0-5 classes if we have to predict more then the train tained traffic sign we have to train the model again with another dataset.

---

## 👤 Authors
* Developed as part of a traffic awareness AI system.
* For educational and hackathon purposes.

---
## These are the commits done during the Hackathone(36 hours)

# **Team -** CoDev

# Day - 1 (8th July)

**Checkpoint 1 (Commit) :-**
**Task ->** Prepare the basic project Structure and also make the working and flow chart.

**Checkpoint 2 (Commit) :-**
**Task ->** Collecting Dataset

**Checkpoint 3 (Commit) :-**
**Task ->** Data Pre-Processing

**Checkpoint 4 (Commit) :-**
**Task ->** Data pre-processing done, Dataset
 	prepared for  model-1 and working on
 	Dataset for model-2

**Checkpoint 5 (Commit) :-**
**Task ->** Train model (still going)

##### Day - 2 (9th July)

**Checkpoint 1 (Commit) :-**
**Task -> Model-1 Training Complete**

**Checkpoint 2 (Commit) :-**
**Task ->** Model-2 dataset completed and trained

**Checkpoint 3 (Commit) :-**
**Task ->** Integration of model1 and model2 is in processing

**Checkpoint 4 (Commit) :-**
**Task ->** Wraping flask application
