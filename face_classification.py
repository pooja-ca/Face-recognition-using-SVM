# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:52:48 2020

@author: Syed
"""

from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train = %d, test = %d' %(trainX.shape[0], testX.shape[0]))

in_encoder = Normalizer(norm = 'l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
print(testy)
trainy = out_encoder.fit_transform(trainy)
testy = out_encoder.transform(testy)

model = SVC(kernel='rbf', probability = True)
model.fit(trainX, trainy)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

target_names = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling', 'syed-asif']

print('Accuracy: train = %d, test = %d' %(score_train*100, score_test*100))
print('Classification Report: ', classification_report(testy, yhat_test, target_names=target_names))
print('Confusion Matrix:\n ', confusion_matrix(testy, yhat_test))
