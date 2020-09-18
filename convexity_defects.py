#Testing convexity defects on still image

import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise
 
five = cv2.imread('images/five_hand.png')

def convexity_def(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    contours, hier = cv2.findContours(thresh,2,1)
    cnt = contours[0]

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0] #start, end, farthest pt, dist to farthest pt
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        print (str(i), ': ', d)

        cv2.line(img,start,end,[190,0,0],2)
        #cv2.circle(img,far,5,[255,0,0],-1)
        cv2.putText(img,str(i),far,cv2.FONT_HERSHEY_COMPLEX,0.85,(0,0,255),2)
          

    return img

five_cnt = convexity_def(five)

cv2.imshow("5", five_cnt)
cv2.waitKey(0)
cv2.destroyAllWindows()
