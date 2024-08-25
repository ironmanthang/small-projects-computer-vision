import cv2
#code for using mobile camera for face detection, run on pc, and then save the video
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

url = "http://10.177013.32.131:8080/" # Your url might be different, check the app "ip webcam"

vs = cv2.VideoCapture(url+"/video")

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while True:
    ret, frame = vs.read()
    if not ret:
        continue

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))  # You can adjust the size as needed

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Write the frame into the file 'output.mp4'
    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release everything after the job is finished
vs.release()
out.release()
cv2.destroyAllWindows()
