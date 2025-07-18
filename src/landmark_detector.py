import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Try different webcam indices
for cam_idx in range(0, 4):
    print(f"⏺ Trying to open camera {cam_idx}")
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        print(f"✅ Opened camera {cam_idx}")
        break
    else:
        cap.release()
        cap = None

if cap is None or not cap.isOpened():
    print("❌ Error: Could not open any webcam (checked indices 0–3).")
    sys.exit(1)

print("✅ Webcam is running. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Warning: Unable to read frame.")
        break
    # Optional: Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Draw landmarks if found
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

    # Show the frame
    cv2.imshow('Facial Landmark Detector', frame)

    # Debug: Print frame status
    # print("Showing frame...")

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Cleanup done. Bye!")
