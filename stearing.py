import numpy as np
import cv2
import mediapipe as mp
import keyspress as key



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
            
        a = np.array([x_coodinate_left,y_coodinate_left])
        b = np.array([x_coodinate_right,y_coodinate_right])
        c = np.array([x_coodinate_left-10000,y_coodinate_left])


        radians = np.arctan2(c[1] - b[1],c[0]-b[0]) - np.arctan2(a[1] - b[1],a[0]-b[0])
        angle = radians*180.0/np.pi


        # print(angle)
        
        if angle < -15:
            key.ReleaseKey(key.A)
            key.PressKey(key.W)
            key.PressKey(key.D)
            print("right")

        elif angle > 15:
            print("left")
            key.ReleaseKey(key.D)
            key.PressKey(key.W)
            key.PressKey(key.A)

        else:
            print("straight")
            key.ReleaseKey(key.D)
            key.ReleaseKey(key.A)
            key.PressKey(key.W)


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
        
        cv2.imshow("Hand Gestures", image)
            # print(results.pose_landmarks.landmark[20].x)
            # image_hight, image_width, _ = image.shape
            # x_coodinate = results.pose_landmarks.landmark[mp_hands.PoseLandmark.LEFT_INDEX].x * image_width
            # y_coodinate = results.pose_landmarks.landmark[mp_hands.PoseLandmark.RIGHT_INDEX].y * image_hight
            # print(x_coodinate)

        angle(image,results)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()





