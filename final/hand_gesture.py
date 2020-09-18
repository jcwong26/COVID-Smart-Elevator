import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise
from scipy.spatial import distance

background = None

def run_avg(image, aWeight):
    global background

    if background is None:
        background = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, background, aWeight)

def segment(image):
    global background
    threshold = 25
    diff = cv2.absdiff(background.astype("uint8"), image)

    # apply binary threshold to create b/w image from grayscale image
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        #get largest contour area (hand)
        segmented = max(cnts, key=cv2.contourArea)

        hull = cv2.convexHull(segmented)

        return (thresholded, segmented, hull)

def convexity_def(thresh):
    contours, hier = cv2.findContours(thresh,2,1)
    cnt = contours[0]
    count = 1
    img = thresh.copy()
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return img
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0] #start, end, farthest pt, dist to farthest pt
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        #print (str(i), ': ', d)
        # if d > 20000:
        #     count += 1

        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        # cosine law
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
        # count finger if angle is less than 90 degrees
        if angle <= np.pi/2:
            count += 1
            cv2.circle(img,far,5,(0,0,255),-1)

        cv2.line(img,start,end,(190,0,0),2)
        
        # cv2.putText(img,str(i),far,cv2.FONT_HERSHEY_COMPLEX,0.85,(0,0,255),2)

    if count > 0:
        cv2.putText(img, 'Fingers: '+str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), thickness=2)          
        
    return (img, count)