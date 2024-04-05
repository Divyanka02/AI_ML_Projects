import mediapipe as mp
import cv2
import numpy as np
import screen_brightness_control as sbc
mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands          
liste_jointe =[8,4] 
def Calculate_distance(image, results, joint_list):
    for hand in results.multi_hand_landmarks:
        r = np.array([hand.landmark[liste_jointe[0]].x, hand.landmark[liste_jointe[0]].y])  # First coord
        f = np.array([hand.landmark[liste_jointe[1]].x, hand.landmark[liste_jointe[1]].y])  # Second coord
        distance = np.sqrt((f[0]-r[0])**2+(f[1]-r[1])**2)
        cv2.putText(image, str(round(distance,2)), tuple(np.multiply(f, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image, distance,r,f
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, fram = cap.read()
        image = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        print(results)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            image, dist,a,b = Calculate_distance(image, results, liste_jointe)
            dist1=dist*250
            if dist1>=100:
                dist1=100
            sbc.set_brightness(dist1, display=0)
            cv2.rectangle(image, (0, 0), (355, 73), (255, 0, 0),thickness = 4)
            cv2.line(image,tuple(np.multiply(a, [640, 480]).astype(int)),tuple(np.multiply(b, [640, 480]).astype(int)),(255,255,255),thickness = 3)
            cv2.putText(image,str(round(dist1,2)),(10, 60),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('screen-brightness-control', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break