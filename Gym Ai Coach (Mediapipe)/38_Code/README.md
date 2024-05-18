1-Run the Script by executing it using Python

2- Data Collection:
Run the script `data_collection.py`
  To run the data collection part of the script, follow these steps:
    -The script will open the webcam and display a live feed.
    -Press 'c' to start recording correct exercises.
    -Press 'i' to stop recording correct exercises.
    -Press 'q' to exit the program.

3-Train the Model:
   After collecting sufficient data, Run the script 'Model_Training.py'
   -Ensure the data file triceps_data.csv is created in the same directory as the script.
   -This part of the script will read the data, train a model, evaluate it, and save the trained model as triceps_model_improved.keras.

4-Real-Time Exercise Analysis:
 Run the script 'Real_Time.py'
  To run the real-time analysis using the trained model:
   -The script will load the trained model and start the webcam.
   -It will display live feedback on whether the exercise is performed correctly or incorrectly.
   -Press 'q' to exit the program.
