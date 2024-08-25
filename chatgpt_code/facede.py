#python code for face detection 
import cv2
import numpy as np
import imutils
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use 0 for laptop's built-in camera
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = imutils.resize(img, width=1000, height=1800)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and save them
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = img  # slice the face from the image

    # Display the output
    cv2.imshow("Camera", img)

    if cv2.waitKey(1) == 27:  # press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
