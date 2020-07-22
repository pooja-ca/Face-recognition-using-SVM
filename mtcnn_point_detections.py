# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:29:24 2020

@author: Syed
"""

import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            x, y, w, h = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(frame,
                (x, y),
                (x+w, y+h),
                (0,155,255),
                2)
            cv2.circle(frame,
                       (keypoints['left_eye']),
                       2,
                       (0,155,255),
                       2)
            cv2.circle(frame,
                       (keypoints['right_eye']),
                       2,
                       (0,155,255),
                       2)
            cv2.circle(frame,
                       (keypoints['nose']),
                       2,
                       (0,155,255),
                       2)
            cv2.circle(frame,
                       (keypoints['mouth_left']),
                       2,
                       (0,155,255),
                       2)
            cv2.circle(frame,
                       (keypoints['mouth_right']),
                       2,
                       (0,155,255),
                       2)
            cv2.line(frame, keypoints['mouth_left'], keypoints['mouth_right'], (0,155,255), 2)
            cv2.imshow('Detector',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()