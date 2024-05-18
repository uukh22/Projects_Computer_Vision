import cv2
import mediapipe as mp
import csv
from datetime import datetime

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Open a CSV file to save the landmarks data
csv_file = open('triceps_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['label'] + [f'landmark_{i}_{j}' for i in range(33) for j in ['x', 'y', 'z', 'visibility']] + ['timestamp'])

# Variable to track if recording correct exercises
recording_correct = False

# Function to extract pose landmarks and write to CSV
def extract_landmarks(results, label):
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([label] + landmarks + [timestamp])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Display the pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow('Data Collection', frame)

    # Capture keypresses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):  # Start recording correct exercises
        recording_correct = True
    elif key & 0xFF == ord('i'):  # Stop recording correct exercises
        recording_correct = False
    elif recording_correct:  # If recording correct exercises, save data
        extract_landmarks(results, 'correct')
    elif not recording_correct:
        extract_landmarks(results, 'Incorrect')

# Release resources
cap.release()
csv_file.close()
cv2.destroyAllWindows()
