import cv2
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from EAR_calculator import *
from imutils import face_utils
import dlib
import time
import csv
from playsound import playsound
from scipy.spatial import distance as dist
import os
from datetime import datetime

# Creating the dataset
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# All eye and mouth aspect ratio with time
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []

# Open the default camera (laptop's camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

# Initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO] Loading the predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landmarks for the left and right eye respectively
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Constants for drowsiness detection
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES = 15
MAR_THRESHOLD = 14

# Initialize counters
BLINK_COUNT = 0
FRAME_COUNT = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    rects = detector(frame, 1)

    # Loop over all the face detections and apply the predictor
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mouth = shape[mstart:mend]

        # Compute the EAR for both the eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        EAR = (leftEAR + rightEAR) / 2.0
        ear_list.append(EAR)

        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))

        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)

        # Drowsiness detection
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1

            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                # Perform drowsiness alert actions
                BLINK_COUNT += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                # Reset drowsiness alert actions
                FRAME_COUNT = 0

        if MAR > MAR_THRESHOLD:
            # Perform yawning alert actions
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
