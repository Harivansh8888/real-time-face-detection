import time
from cv2 import cv2
face_cascade = cv2.CascadeClassifier("C:\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)


while True:
    
    _, img = video.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , 1.05, 15)

    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),4)
        
    cv2.imshow('Capturing Video', img)

    key = cv2.waitKey(1)
    if key == ord('h'):
        break


video.release()
cv2.destroyAllWindows()
