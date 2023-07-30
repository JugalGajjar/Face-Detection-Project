import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

url = "http://192.168.174.145:4747/video?640x480"
webcam = cv2.VideoCapture(url)

while webcam.isOpened() is True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        raise "Failed to read image!"

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('JG\'s Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q/q is pressed
    if key == 81 or key == 113:
        break

print('Code Completed')