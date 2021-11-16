import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time

f = st.file_uploader('Upload File')

tmpfile = tempfile.NamedTemporaryFile(delete=False)
tmpfile.write(f.read())

cap = cv2.VideoCapture(tmpfile.name)
stframe = st.empty()


# -------------- PoseDetection -------------
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    #cv2.imshow("Image", img)
    stframe.image(img)
    cv2.waitKey(1)
    
