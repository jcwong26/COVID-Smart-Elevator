# Comparison of three methods to detect the hand

import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise
from scipy.spatial import distance

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

def segment(image, threshold=25):
    global background
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

def circ_mask(image, hull): # (threshold img, hull)
    # get most extreme points of contour
    left = tuple(hull[hull[:,:,0].argmin()][0])
    right = tuple(hull[hull[:,:,0].argmax()][0])
    top = tuple(hull[hull[:,:,1].argmin()][0])
    bottom = tuple(hull[hull[:,:,1].argmax()][0])

    midx = (left[0]+right[0])//2
    midy = (top[0]+bottom[0])//2

    distance = pairwise.euclidean_distances([left,right,bottom,top],[[midx, midy]])[0]
    radius = int(0.80*distance)

    circular_rui = np.zeros_like(image, dtype='uint8')
    cv2.circle(circular_rui, (midx,midy), radius, (255,255,255),10)

    mask = cv2.bitwise_and(image, image, mask=circular_rui) #threshold image
  

    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    circumference = 2*np.pi*radius
    for cnt in contours:
        (x, y, w, h)=cv2.boundingRect(cnt)

        #Don't count bottom contour (wrist)
        not_wrist=(midx+(midy*0.25)) > (y+h)
        limit_pts=(circumference*0.25)>cnt.shape[0]
        if limit_pts and not_wrist:
            count+=1

    cv2.putText(mask, 'Fingers: '+str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), thickness=2)
    cv2.imshow("Circular Mask", mask)

def convexity_def(thresh):
    contours, hier = cv2.findContours(thresh,2,1)
    cnt = contours[0]
    count = 0
    img = thresh.copy()
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return img
    for i in range(defects.shape[0]):
        print (defects)
        s,e,f,d = defects[i,0] #start, end, farthest pt, dist to farthest pt
        start = tuple(cnt[s][0])
        print("start: ", start)
        end = tuple(cnt[e][0])
        print("end: ", end)
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
        
        #cv2.putText(img,str(i),far,cv2.FONT_HERSHEY_COMPLEX,0.85,(0,0,255),2)

    if count > 0:
        cv2.putText(img, 'Fingers: '+str(count+1), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), thickness=2)          
    else:
        cv2.putText(img, 'Fingers: 0', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), thickness=2)          
    return img

def convexity_def2(thresh,col_img):
    contours, hier = cv2.findContours(thresh,2,1)
    cnt = contours[0]
    count = 0

    hull = cv2.convexHull(cnt, returnPoints=True)
    if hull is None:
        return col_img
    prev = tuple(hull[0][0])
    for pt in hull:
        plot = pt[0]
        if distance.euclidean(prev, plot) < 50:
            x = (prev[0]+plot[0])/2
            y = (prev[1]+plot[1])/2
            prev = (x,y)
        else:
            cv2.circle(col_img,tuple(plot),5,(0,0,255),2)
            prev = plot
        cv2.circle(col_img,tuple(plot),5,(0,0,255),2)
    
    # cv2.putText(img, 'Fingers: '+str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), thickness=2)                 
    return col_img


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

    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        hand = segment(gray)

        if hand is not None:
            (threshold, segmented, hull) = hand

            cv2.drawContours(clone, [segmented + (right,top)], -1, (0, 0, 255))
            # cv2.imshow("Threshold", threshold)

            drawing = np.zeros((threshold.shape[0], threshold.shape[1], 3), np.uint8)
            cv2.drawContours(clone, [hull + (right,top)], 0, (255,255,255), 2)
            

            circ_mask(threshold, hull)
            convex = convexity_def2(threshold,roi)

            cv2.imshow("Convexity Defects Count", convex)

            
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    num_frames += 1

    cv2.imshow("Video Feed", clone)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress ==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
