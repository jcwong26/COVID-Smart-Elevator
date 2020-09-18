import cv2

body_class = cv2.CascadeClassifier('haarcascades\haarcascade_fullbody.xml')
cap = cv2.VideoCapture(1)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    detections = body_class.detectMultiScale(gray)
    for (x,y,w,h) in detections:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),3)

    cv2.imshow('Body Detection', img)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    