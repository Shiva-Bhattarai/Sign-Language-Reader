import numpy as np
import mediapipe as mp
import cv2 as cv
from Function import *  

import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

holy_hands = mp.solutions.hands

camera = cv.VideoCapture(0)
index_cord = []  # List to store values for pointer

prev_time = 0  # Initializing prev_time

with holy_hands.Hands(max_num_hands=1) as hands:
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv.flip(frame, 1)  # Flip the frame horizontally

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = frame.shape
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    cv.circle(frame, (x, y), 5, (255, 0, 255), cv.FILLED)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                imgH, imgW = frame.shape[:2]
                string = ''
                hand_cordinate = []
                for index, landmark in enumerate(hand_landmarks.landmark):
                    x_cordinate, y_cordinate = int(landmark.x * imgW), int(landmark.y * imgH)
                    hand_cordinate.append([index, x_cordinate, y_cordinate])
                hand_cordinate = np.array(hand_cordinate)

                string = persons_input(hand_cordinate)
                frame = get_fram(frame, hand_cordinate, string)

                if string == " D":
                    index_cord.append([15, hand_cordinate[8][1], hand_cordinate[8][2]])
                if string == " I" or string == " J":
                    index_cord.append([15, hand_cordinate[20][1], hand_cordinate[20][2]])
                for val in index_cord:
                    frame = cv.circle(frame, (val[1], val[2]), val[0], (255, 255, 255), 1)
                    val[0] = val[0] - 1
                    if val[0] <= 0:
                        index_cord.remove(val)

        # Resize the frame for better viewing (adjust the resizing factor as needed)
        resized_frame = cv.resize(frame, (int(frame.shape[1] * 1.3), int(frame.shape[0] * 1.3)))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0  # Avoid division by zero
        prev_time = curr_time
        cv.putText(resized_frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv.imshow('Hand Tracking and Sign Language Detection', resized_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv.destroyAllWindows()