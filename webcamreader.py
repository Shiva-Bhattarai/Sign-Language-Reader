import cv2 as cv
import mediapipe as mp

webcam = cv.VideoCapture(0) #to cature the video from webcam

while True:
    ret, frame = webcam.read() #ret must be used in order to open the webcam frame window

    cv.imshow('frame', frame)
    if cv.waitKey(40) == ord('q'): #ord is used so that q is pressed to quit and wait key is 40 for 40 millisecond per frame
        break

webcam.release()
cv.destroyAllWindows() #to destroy all windows