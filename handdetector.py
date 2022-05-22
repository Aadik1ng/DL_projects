import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

#i = int(input(print('Enter the ID')))
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
listx=[]
listy=[]
box=[]
area = []
lth=[]
while True:
    k=cv2.waitKey(1)& 0xFF
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                #if (id == i):
                cx, cy = int(lm.x * w), int(lm.y * h)
                listx.append(cx)
                listy.append(cy)
                #print(type(lm))
                #print(id, cx, cy)
                if id ==4:
                    x1=cx
                    y1=cy
                if id==8:
                    x2=cx
                    y2=cy
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

                    # mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
            xmax,xmin=max(listx),min(listy)
            ymax,ymin=max(listx),min(listy)
            box=xmin, ymin, xmax, ymax
            cv2.rectangle(img, (box[0]-50,box[1]-50),(box[2]+50,box[3]+50), (0, 255, 0), 2)
            listx = []
            listy = []
            area = ((box[2] - box[0]) * (box[3] - box[1]) // 100)
            if area > 500:
                leng = int(math.hypot(x2 - x1, y2 - y1))
                lth.append(leng)
                cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
                vol = np.interp(leng, [20, 300], [minVol, maxVol])
                volBar = np.interp(leng, [20, 300], [400, 150])
                volPer = np.interp(leng, [20, 300], [0, 100])
                print(int(leng), vol)
                volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 20, 255), 3)






    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    if k == ord('q'):
        break
    cv2.imshow("Image", img)
