import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands # el tespiti için mediapipe yükler (formalite)
hands = mpHands.Hands()  # create the object it uses RGB
mpDraw = mp.solutions.drawing_utils # algılanan el noktalarını çizmek için yardımcı araç

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) # mediapipe için 
    results = hands.process(imgRGB) # görüntüyü işleyerek eltespiti yapar
    # print(results.multi_hand_landmarks)
    
    # check if we have multiple hands
    if results.multi_hand_landmarks: # eğer tespit edilen eller varsa, her birinin eklme noktaları handLm olarak alınır
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms) # algılanan eklem noktalarını img üzerine çizer
    
    cv.imshow('Frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'): # 'q' tuşuna basınca çık
        break

cap.release()
cv.destroyAllWindows()