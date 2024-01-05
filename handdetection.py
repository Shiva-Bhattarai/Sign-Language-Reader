import cv2
import mediapipe as mp
import time

# Accessing the default camera (Webcam)
camera = cv2.VideoCapture(0)

# Hand tracking using Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0

# Main loop for continuous video processing
while True:
    # Capturing the video frame by frame
    success, frame = camera.read()
    # Converting the color from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processing hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Iterating through detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extracting landmark position
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                # Displaying landmarks as circles on the frame
                cv2.circle(frame, (x, y), 5, (255, 0, 255), cv2.FILLED)

            # Drawing hand connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculating frames per second (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Displaying the FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Displaying the processed frame
    cv2.imshow("Hand Tracking", frame)
    # Exiting the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the camera and closing the OpenCV windows
camera.release()
cv2.destroyAllWindows() == ord('q')
