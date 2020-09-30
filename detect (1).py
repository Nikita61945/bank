#!/usr/bin/python


from utils import*
from gtts import gTTS
import os
from matplotlib import pyplot as plt
import cv2
import subprocess
from gtts import gTTS
import pickle
from PIL import Image
import scipy.ndimage
import mahotas as mt 

def extract_features(image):
	textures = mt.features.haralick(image)
	
	ht_mean = textures.mean(axis=0)
	return ht_mean
max_val = 8
max_pt = -1
max_kp = 0
thresholdVal = 65
train_features = []
train_labels = []
train_path = "Training Images"
train_names = os.listdir(train_path)
orb = cv2.ORB_create(nfeatures=12000)
#scoreType=cv2.ORB_FAST_SCORE
#nfeatures=500
test_img = read_img('F:/currency-recognition-master/Training Images/R10obverse.jpg')
fromCenter = False
test_img=resize_img(test_img, 0.5)
#r = cv2.selectROI("Image",test_img, fromCenter)


imCrop = test_img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
imCrop=test_img
cv2.imshow("Image", imCrop)

cv2.waitKey(0)

filename = "TrainedSVMClassifier6.sav"
clf_svm = pickle.load(open(filename, 'rb'))
classifyImage = imCrop
#Perform image preprocessing and enhancement to extract features for classification
classifyImage=cv2.cvtColor(classifyImage, cv2.COLOR_RGB2GRAY)
classifyImage = cv2.equalizeHist(classifyImage)
ret, thresh = cv2.threshold(classifyImage, thresholdVal, 255, cv2.THRESH_BINARY)
classifyImage = thresh
#angles=1*45
#classifyImage = scipy.ndimage.rotate(classifyImage,angles)
#cv2.imshow("yo",classifyImage)
cv2.waitKey(0)
features = extract_features(classifyImage)
prediction = clf_svm.predict(features.reshape(1, -1))[0]
#img5=cv2.rotate(imCrop, cv2.ROTATE_90_CLOCKWISE)
#img5 = scipy.ndimage.rotate(imCrop,45,reshape=true)
count=0
#prediction="null"
#for i in range(0,8):
	
#	if(prediction== "Note"):
		
	   
	   
#		break
#	else:
		
#		classifyImage    = scipy.ndimage.rotate(classifyImage,45)
		
#		features = extract_features(classifyImage)
#		prediction=clf_svm.predict(features.reshape(1, -1))[0]
#		count=count+1
		
		
	
angle= count*45
original= scipy.ndimage.rotate(imCrop,angle)
original = resize_img(original, 1.2)
original1=img_to_gaussian_gray(original)
#original2=sobel_edge2(original1)
original3=sobel_edge2(original1)

#original4= fourier(original)
display('original', original3)


# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)

(kp1, des1) = orb.detectAndCompute(original3, None)

training_set = ['files/10front.jpg','files/10back.jpg','files/010obverse.jpg', 'files/010reverse.jpg','files/10front_large.jpg',
				'files/20front.jpg', 'files/20back.jpg', 'files/020obverse.jpg','files/020reverse.jpg','files/20front_large.jpg',
				'files/50front.jpg', 'files/50back.jpg','files/050obverse.jpg','files/050reverse.jpg','files/50front_large.jpg',
				'files/100back.jpg','files/100front.jpg','files/100obverse.jpg','files/100reverse.jpg','files/100front_large.jpg',
				'files/200back.jpg','files/200front.jpg','files/200obverse.jpg','files/200reverse.jpg','files/200front_large.jpg']

for i in range(0, len(training_set)):
	#train image
	train_img = cv2.imread(training_set[i])    
	tr2= img_to_gaussian_gray(train_img)
	#tr3= sobel_edge2(tr2)
	tr4= sobel_edge2(tr2)
	(kp2, des2) = orb.detectAndCompute(tr4, None)
	
	# brute force matcher
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	
	
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if prediction=="Note":
	#print(training_set[max_pt])
	#print('good matches ', max_val)
	
	train_img = cv2.imread(training_set[max_pt])
	tr2= img_to_gaussian_gray(train_img)
	#tr3= sobel_edge2(tr2)
	tr4=sobel_edge2 (tr2)
	
	
	#img3 = cv2.drawMatchesKnn(original3, kp1, tr4, max_kp, good,4)
	img3= cv2.drawMatchesKnn(original3, kp1, tr4, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	
	note = str(training_set[max_pt])[6:-3]
	print('\nDetected denomination: Rs. ', note)
	display("i",tr4)
	display("hello",img3)
	
	mytext = note
	language = 'en'
	#myobj = gTTS(text=mytext, lang=language, slow=False)
	#myobj.save("welcome.mp3")
	#os.system(" welcome.mp3")
	

else:
	print('That is not a NOTE')