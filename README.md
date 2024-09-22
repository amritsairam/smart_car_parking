Parking Spot Detector

This project is designed to detect available parking spots in real-time using video footage. The system identifies parking spots, tracks changes in occupancy using frame differences, and marks spots as available or occupied with color-coded rectangles.

Features

Parking Spot Detection: Detects parking spots using a mask and bounding boxes.
Occupancy Status: Determines if a parking spot is occupied or free based on a machine learning model.
Real-Time Updates: Updates the status every 30 frames to ensure real-time performance.
Efficient Computation: Only the necessary frames are used for comparison to optimize performance.
Requirements

Python 3.x
OpenCV
Scikit-Image
NumPy
Scikit-Learn (for loading the pre-trained model)
A trained model for parking spot occupancy classification (provided in model.p).

How It Works

Video and Mask Input: The program uses a video file and a mask image to detect parking spots. The mask image is used to identify connected components, which represent the parking spots.
Frame Processing: For every 30 frames, the system checks for differences between the current frame and the previous one. This is used to detect if a parking spot’s status has changed.
Occupancy Detection: A machine learning model (loaded from model.p) is used to classify each parking spot as either empty or occupied.
Real-Time Display: The parking spots are marked with color-coded rectangles:
Green: Spot is available.
Red: Spot is occupied.
Available Spots Count: The total number of available parking spots is displayed on the video in real time.

File Descriptions

main.py: The main script that handles video processing, parking spot detection, and real-time updates.
utils.py: Contains utility functions for detecting parking spots (get_parking_spots_bboxes) and determining whether a spot is empty or occupied (empty_or_not).
model.p: A pre-trained machine learning model that predicts whether a parking spot is occupied or not based on the input image.
