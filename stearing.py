import numpy as np
import cv2
import mediapipe as mp
import keyspress as key
import math
import time

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.holistic



cap = cv2.VideoCapture(0)


# joints = [[20,19]]


def angle(image,results):

    image_hight, image_width, _ = image.shape

    if results.pose_landmarks:
        x_coodinate_left = results.pose_landmarks.landmark[mp_hands.PoseLandmark.RIGHT_INDEX].x * image_width
        y_coodinate_left = results.pose_landmarks.landmark[mp_hands.PoseLandmark.RIGHT_INDEX].y * image_hight
        x_coodinate_right = results.pose_landmarks.landmark[mp_hands.PoseLandmark.LEFT_INDEX].x * image_width
        y_coodinate_right = results.pose_landmarks.landmark[mp_hands.PoseLandmark.LEFT_INDEX].y * image_hight

        x_coodinate_finger_tip = 10
        y_coodinate_finger_tip = 10
        x_coodinate_finger_pip = 10
        y_coodinate_finger_pip = 10
        x_coodinate_wrist = 0
        y_coodinate_wrist = 0



        backdistance = 180

        if results.left_hand_landmarks:
            # x,v cordinates for back fingre
            x_coodinate_finger_tip = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            y_coodinate_finger_tip = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight


            x_coodinate_finger_pip = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
            y_coodinate_finger_pip = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_hight
            # x_coodinate_wrist = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
            # y_coodinate_wrist = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_hight
            # x_coodinate_finger_tip = 10
            # y_coodinate_finger_tip = 10
            # x_coodinate_finger_pip = 10
            # y_coodinate_finger_pip = 10
            xy_tip = np.array([x_coodinate_finger_tip,y_coodinate_finger_tip])
            xy_pip = np.array([x_coodinate_finger_pip,y_coodinate_finger_pip])
            # xy_wrist = np.array([x_coodinate_wrist,y_coodinate_wrist])

            # back_radians = np.arctan2(xy_wrist[1] - xy_pip[1],xy_wrist[0]-xy_pip[0]) - np.arctan2(xy_tip[1] - xy_pip[1],xy_tip[0]-xy_pip[0])
            # back_angle = math.sqrt(((abs(xy_pip[0])-abs(xy_tip[0]))**2) - ((abs(xy_pip[1])-abs(xy_tip[1]))**2))
            backdistance = math.sqrt((xy_tip[0]-xy_pip[0])**2 + (xy_tip[1]-xy_pip[1])**2)
            # print(backdistance)

            # if back_angle > 180:
            #     back_angle = math.floor(360 -back_angle)



        # else:
        #     x_coodinate_finger_tip = 0
        #     y_coodinate_finger_tip = 0
        #     x_coodinate_finger_pip = 0
        #     y_coodinate_finger_pip = 0
        #     x_coodinate_wrist = 0
        #     y_coodinate_wrist = 0
        #     xy_tip = 0
        #     xy_tip = 0
        #     xy_wrist = 0


            # print(x_coodinate_finger_tip)
            
        a = np.array([x_coodinate_left,y_coodinate_left])
        b = np.array([x_coodinate_right,y_coodinate_right])
        c = np.array([x_coodinate_left-10000,y_coodinate_left])


        # shoulder_coordinate = np.array([x_coodinate_left_shoulder,y_coodinate_left_shoulder])
        # elbow_coordinate = np.array([x_coodinate_left_elbow,y_coodinate_left_elbow])

        

        radians = np.arctan2(c[1] - b[1],c[0]-b[0]) - np.arctan2(a[1] - b[1],a[0]-b[0])
        angle = radians*180.0/np.pi

        
        # back_radians = np.arctan2(a[1]-elbow_coordinate[1],a[0]-elbow_coordinate[0]) - np.arctan2(shoulder_coordinate[1]-elbow_coordinate[1],shoulder_coordinate[0]-elbow_coordinate[0])
        # back_angle = abs(back_radians*180.0/np.pi)

        
        time.sleep(0.1)
        if angle < -15 and backdistance < 30:
                key.ReleaseKey(key.W)
                key.ReleaseKey(key.A)
                key.PressKey(key.S)
                key.PressKey(key.D)
                print("right + BACK")

        elif angle > 15 and backdistance < 30:

            print("Left + back")
            key.ReleaseKey(key.D)
            key.ReleaseKey(key.W)
            key.PressKey(key.S)
            key.PressKey(key.A)
        
        elif angle > 15 and backdistance > 30:
            print("Left + Straight")
            key.ReleaseKey(key.S)
            key.ReleaseKey(key.D)
            key.PressKey(key.W)
            key.PressKey(key.A)

        elif angle < -15 and backdistance > 30:
            print("Right + Straight")
            key.ReleaseKey(key.S)
            key.ReleaseKey(key.A)
            key.PressKey(key.W)
            key.PressKey(key.D)

        elif angle >-15 and angle <15 and backdistance >30:
            print("Straight")
            key.ReleaseKey(key.S)
            key.ReleaseKey(key.A)
            key.ReleaseKey(key.D)
            key.PressKey(key.W)

        elif angle >-15 and angle <15 and backdistance <30:
            print("Back")
            key.ReleaseKey(key.W)
            key.ReleaseKey(key.A)
            key.ReleaseKey(key.D)
            key.PressKey(key.S)

        else:
            print("nothing")
            key.ReleaseKey(key.W)
            key.ReleaseKey(key.A)
            key.ReleaseKey(key.D)
            key.ReleaseKey(key.S)


        # print(angle)

    return image


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

        angle(image,results)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()





