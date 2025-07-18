import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# === Load fine-tuned model ===
print("🔄 Loading emotion model...")
model = load_model("models/emotion_model.h5")
print("✅ Model loaded successfully!")

# === Emotion labels (must match your model training order) ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === Mediapipe face mesh setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
print("🎯 Mediapipe FaceMesh initialized")

# === Webcam setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

print("📸 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to read from webcam.")
        break

    # Convert frame to RGB for mediapipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Convert frame to grayscale for emotion model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray, (48, 48))
        normalized_face = resized_face.astype("float32") / 255.0
        img_array = np.expand_dims(np.expand_dims(normalized_face, -1), 0)  # Shape: (1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(img_array, verbose=0)
        emotion_index = int(np.argmax(prediction))
        emotion = emotion_labels[emotion_index]

        # Show prediction live in terminal
        print(f"🧠 Detected Emotion: {emotion}")

        # Show on screen
        cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    else:
        print("😐 No face detected")

    # Display the frame
    cv2.imshow("Facial Expression Detector", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔚 Quitting application...")
        break

cap.release()
cv2.destroyAllWindows()
