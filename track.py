import numpy as np
import cv2
import mediapipe as mp
import math


mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.holistic



cap = cv2.VideoCapture(0)


# joints = [[20,19]]


with mp_hands.Holistic(min_detection_confidence = 0.8,min_tracking_confidence =0.5) as hand:
    # print(hand)
    while cap.isOpened():
        ret, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hand.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # print(results.multi_hand_landmarks)

        

        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks,mp_hands.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(image, results.left_hand_landmarks,mp_hands.HAND_CONNECTIONS)
        

        
        cv2.imshow("Hand Gestures", image)
        
            # image_hight, image_width, _ = image.shape
            # x_coodinate = results.pose_landmarks.landmark[mp_hands.PoseLandmark.LEFT_INDEX].x * image_width
            # y_coodinate = results.pose_landmarks.landmark[mp_hands.PoseLandmark.RIGHT_INDEX].y * image_hight
            # print(x_coodinate)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()





