# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:08:11 2020

@author: Syed
"""

# importing libraries
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import load
from keras.models import load_model
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import cv2
from numpy import asarray
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

detector = MTCNN()

data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train = %d, test = %d\n' %(trainX.shape[0], testX.shape[0]))

print('Loading FaceNet...')
model2 = load_model('facenet_keras.h5')
print('FaceNet Loaded\n')

print('loading SVM...')
in_encoder = Normalizer(norm = 'l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
print('Vector Normalisation...')
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
# print(testy)
print('Encoding Output Labels...')
trainy = out_encoder.fit_transform(trainy)
testy = out_encoder.transform(testy)
print('Loading the model...')
model = SVC(kernel='linear', probability = True)
model.fit(trainX, trainy)
print('SVM Loaded\n')

print('Detecting...')

# face recognition with webcam
cap = cv2.VideoCapture(0)

while True:
    def get_embedding(model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis = 0)
        yhat = model.predict(samples)
        return yhat[0]

    ret, frame = cap.read()
    
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            x, y, w, h = person['box']
            x, y = abs(x), abs(y)
            cropped_face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame,
                          (x, y),
                          (x+w, y+h),
                          (0,155,255),
                          2)
            test_image = Image.fromarray(cropped_face)
            test_image = test_image.resize((160,160))
            test_image = asarray(test_image)
            test_image = get_embedding(model2, test_image)
            # test_image = test_image.reshape(1, -1)
            sample = expand_dims(test_image, axis = 0)
            yhat_class = model.predict(sample)
            yhat_prob = model.predict_proba(sample)
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index]*100
            # print(yhat_class)
            predict_names = out_encoder.inverse_transform(yhat_class)
            # print(predict_names[0], class_probability)
            if class_probability < 95:
                predict_names[0]="Match not found"
            cv2.putText(frame,'%s - %.2f' % (predict_names[0],class_probability), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,155,255), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()