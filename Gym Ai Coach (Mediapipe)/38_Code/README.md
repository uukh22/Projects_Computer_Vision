# Exercise Analysis Platform

This platform is designed to collect exercise data, train a model, and provide real-time feedback on exercise performance using Python scripts.

## Steps to Use the Platform

### 1. Run the Script
Execute the script using Python to start the process.

### 2. Data Collection
Run the `data_collection.py` script to collect exercise data.
- The script will open the webcam and display a live feed.
- Press 'c' to start recording correct exercises.
- Press 'i' to stop recording correct exercises.
- Press 'q' to exit the program.

### 3. Train the Model
After collecting sufficient data, run the `Model_Training.py` script.
- Ensure the data file `triceps_data.csv` is created in the same directory as the script.
- This script will read the data, train a model, evaluate it, and save the trained model as `triceps_model_improved.keras`.

### 4. Real-Time Exercise Analysis
Run the `Real_Time.py` script for real-time exercise analysis using the trained model.
- The script will load the trained model and start the webcam.
- It will display live feedback on whether the exercise is performed correctly or incorrectly.
- Press 'q' to exit the program.

---

