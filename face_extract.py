# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:34:04 2020

@author: Syed
"""

from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    # print(results)
    
    x1, y1, w, h = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):            
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class %s' %(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

trainX, trainy = load_dataset('data/train/')
print(trainX.shape, trainy.shape)

testX, testy = load_dataset('data/val/')
print(testX.shape, testy.shape)

np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)


        
    