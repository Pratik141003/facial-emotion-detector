# Facial Emotion Detection & Landmark Tracking System

This project is a real-time facial emotion recognition and landmark detection system using deep learning (TensorFlow/Keras) and MediaPipe. It allows for detecting human emotions such as Happy, Sad, Angry, etc., directly from live webcam footage, while also overlaying facial landmarks.

---

## 📌 Features

- 🎯 **Real-Time Emotion Recognition** using webcam input
- 💡 **Deep Learning-Based CNN Model** trained on grayscale face images
- 🧠 Fine-tuning support using **MobileNetV2**
- 🧍‍♂️ Facial landmark detection using **MediaPipe FaceMesh**
- 📈 Class weighting for imbalanced datasets
- 🗂 Modular code for easy training, testing, and inference

---

## 📁 Project Structure

facial-emotion-detector/
│
├── dataset/ 
├── models/ # Trained models (train_emotion_model.h5 )
│
├── main_app.py # Run this for real-time webcam emotion detection
├── landmark_detector.py # Test only landmark detection
├── train_emotion_model.py # Train CNN model from scratch
├── fine_tune_custom_model.py # Fine-tune MobileNetV2 on your dataset
├── requirements.txt # Python dependencies
└── README.md # Project overview



---

## 🚀 How to Run

### 🔧 1. Install Requirements

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv env
env\Scripts\activate       # On Windows
source env/bin/activate    # On Linux/Mac

Install dependencies:
pip install -r requirements.txt

 Train from Scratch:
python train_emotion_model.py

 Fine-Tune with MobileNetV2:
python fine_tune_custom_model.py

Run Real-Time Emotion Detection
python main_app.py

Run Landmark Detection Only
python landmark_detector.py

Emotion Classes Used:
['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

Technologies Used:

Python 3.x
TensorFlow / Keras
OpenCV
MediaPipe
NumPy
Matplotlib
scikit-learn

