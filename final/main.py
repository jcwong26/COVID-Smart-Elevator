# Main Program

import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise
from scipy.spatial import distance
from elevator import Elevator
from hand_gesture import run_avg, segment, convexity_def, background

global background

elevator = Elevator(5)

aWeight = 0.5

camera = cv2.VideoCapture(1) #external webcam

top, right, bottom, left = 150, 500, 600, 950

num_frames = 0

while (True):
    (grabbed, frame) = camera.read()

    #resize and flip frame
    frame = imutils.resize(frame, width = 1000)
    frame = cv2.flip(frame, 1)

    clone = frame.copy()

    (height, weight) = frame.shape[:2]

    roi = frame[top:bottom, right:left]

    #convert to grayscale and add blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        hand = segment(gray)

        if hand is not None:
            (threshold, segmented, hull) = hand

            cv2.drawContours(clone, [segmented + (right,top)], -1, (0, 0, 255))
            cv2.imshow("Threshold", threshold)

            drawing = np.zeros((threshold.shape[0], threshold.shape[1], 3), np.uint8)
            cv2.drawContours(clone, [hull + (right,top)], 0, (255,255,255), 2)

            (convex, num_fingers) = convexity_def(threshold)

            cv2.imshow("Convexity Defects Count", convex)

            
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    num_frames += 1

    cv2.imshow("Video Feed", clone)

    keypress = cv2.waitKey(1) & 0xFF
    if keypress ==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

