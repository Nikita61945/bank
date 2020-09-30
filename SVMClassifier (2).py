# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:58:07 2020

@author: sheru
"""
import cv2 
import mahotas as mt
import numpy as np
from sklearn.svm import LinearSVC
import os
import glob
import time
import pickle

train_features = []
train_labels = []
train_path = "Training Images"
train_names = os.listdir(train_path)



def extract_features(image):
    textures = mt.features.haralick(image)
    
    ht_mean = textures.mean(axis=0)
    return ht_mean

time_start = time.time()



for i in range(0, 5):
    imageIndex = 0
    for train_name in train_names:
        cur_path = train_path + "/" + train_name
        label = "Note"
        
        for file in glob.glob(cur_path):
            
            img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.equalizeHist(img1)
            
            ret, thresh = cv2.threshold(img1, 65, 255, cv2.THRESH_BINARY)
            img1 = thresh
            
            features = extract_features(img1)
            
            label = "Note"
            train_features.append(features)
            train_labels.append(label)
            print("Iteration(", (i+1),") Image ", (imageIndex+1), " of ", len(train_names), " completed.")
            imageIndex +=1

negative_path = "Negative Images"
negative_names = os.listdir(negative_path)
imageIndex = 0
for negative_name in negative_names:
    cur_path = negative_path + "/" + negative_name
    label = "Negative"
    
    for file in glob.glob(cur_path):
        img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.equalizeHist(img1)
        ret, thresh = cv2.threshold(img1, 65, 255, cv2.THRESH_BINARY)
        img1 = thresh
        
        features = extract_features(img1)
        
        train_features.append(features)
        train_labels.append(label)
        print("Negative image ", (imageIndex+1), " of ", len(negative_names), " completed.")
        imageIndex +=1
    

print("Training features {}".format(np.array(train_features).shape))
print("Training labels: {}".format(np.array(train_labels).shape))

print("[STATUS] Creating the classifier...")
clf_svm = LinearSVC(C=1, dual=False, max_iter=10000)   #random_state=9, max_iter=5000, dual=False

print("[STATUS] Fitting data/label to model...")
clf_svm.fit(train_features, train_labels)

print("[STATUS] Training complete...")

print("-----Testing-----")
test_path = "Test Images" 
test_names = os.listdir(test_path)

i = 1
positive = 0
negative = 0
for name in test_names:
    
    cur_path = test_path + "/" + name 
    
    for file in glob.glob(cur_path):
        
        testImg = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        testImg = cv2.equalizeHist(testImg)
        ret, thresh = cv2.threshold(testImg, 65, 255, cv2.THRESH_BINARY)
        testImg = thresh
        features = extract_features(testImg)
        prediction = clf_svm.predict(features.reshape(1, -1))[0]
        print("Prediction of note ", i, ": ", prediction, "-> ", name)
        if prediction == "Negative":
            negative += 1
        else:
            positive+=1
        i += 1


time_elapsed = time.time() - time_start
print("Completed in: ", round(time_elapsed, 2), " seconds")
print("Notes: ", positive)
print("Negative: ", negative)
accuracy = (positive)/60*100
print("Accuracy: ", accuracy)
print("Saving trained classifier...")
filename = "TrainedSVMClassifier6.sav"
classifier = pickle.dump(clf_svm, open(filename, 'wb'))
print("Classifier saved!")
    
    
