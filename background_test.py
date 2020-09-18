# Comparison of background subtraction methods

import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise


cap = cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2()

background = None

aWeight = 0.5

camera = cv2.VideoCapture(1) #external webcam

top, right, bottom, left = 150, 500, 600, 950

num_frames = 0

# Calculate average of background
def run_avg(image, aWeight):
    global background

    if background is None:
        background = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, background, aWeight)

def bg_sub(image):
    fgmask = fgbg.apply(image)
    return fgmask

def segment(image, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), image)

    # apply binary threshold to create b/w image from grayscale image
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)

        hull = []
        for i in range(len(cnts)):
            hull.append(cv2.convexHull(cnts[i],False))

        return (thresholded, segmented, hull, cnts)

def edge(image):
    global background

    canny_output = cv2.Canny(image, 100, 200)
    return canny_output

## Main Program
while(True):
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

    background_sub = bg_sub(roi)
    cv2.imshow("FG Mask", background_sub)

    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        hand = segment(gray)

        if hand is not None:
            (threshold, segmented, hull, contours) = hand

            cv2.drawContours(clone, [segmented + (right,top)], -1, (0, 0, 255))
            cv2.imshow("Threshold", threshold)

    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    canny = edge(gray)
    cv2.imshow("Canny",canny)

    num_frames += 1

    cv2.imshow("Video Feed", clone)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress ==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
