import cv2

body_class = cv2.CascadeClassifier('haarcascades\haarcascade_fullbody.xml')

def body(frame):
    img = frame.copy()
    count = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    detections = body_class.detectMultiScale(gray)
    for (x,y,w,h) in detections:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),3)
        count += 1

    return (img, count)

    