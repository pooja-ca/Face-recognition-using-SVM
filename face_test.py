# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:37:43 2020

@author: Syed
"""

from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']

data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train = %d, test = %d' %(trainX.shape[0], testX.shape[0]))

in_encoder = Normalizer(norm = 'l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
# print(testy)
trainy = out_encoder.fit_transform(trainy)
testy = out_encoder.transform(testy)
# print(testy)

model = SVC(kernel='rbf', probability = True)
model.fit(trainX, trainy)

selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# print(random_face_class)

samples = expand_dims(random_face_emb, axis = 0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index]*100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])

plt.imshow(random_face_pixels)
title = '%s (%.3f)' %(predict_names[0], class_probability)
plt.title(title)
plt.show()
