import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

# Initialize pyttsx3
engine = pyttsx3.init()

# Load the trained model
model = tf.keras.models.load_model('triceps_model_improved.keras')

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam
cap = cv2.VideoCapture(0)

# Variable to store the previous label
previous_label = None
label_lock = threading.Lock()

def speak_label(label):
    with label_lock:
        print(f"Speaking label: {label}")  # Debugging statement
        engine.say(label)
        engine.runAndWait()

def extract_landmarks_for_prediction(results):
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks).reshape(1, -1)
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Display the pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = extract_landmarks_for_prediction(results)
        if landmarks is not None:
            prediction = model.predict(landmarks)
            label = 'Correct' if prediction[0] > 0.5 else 'Incorrect'

            # Debugging statement
            print(f"Predicted label: {label}, Previous label: {previous_label}")

            # Speak the label if it has changed
            if label != previous_label:
                previous_label = label
                threading.Thread(target=speak_label, args=(label,)).start()

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if label == 'Correct' else (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Analysis', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
