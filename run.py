import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential, save_model, load_model
filepath='F:/comp7022/saved_model'
model = load_model(filepath, compile = True)

#sample = []
#sample.append('C:/Users/Nikita/Desktop/100rand.jpg')
#sample = np.array(sample)
#print(sample.shape)
#predictions = model.predict(sample)
#print(predictions)
#classes = np.argmax(predictions, axis = 4)
#print(classes)

def prepare(filepathpicture):
    IMG_SIZE=64
    img_array=cv2.imread(filepathpicture)
    new_array=cv2.resize(img_array,(64,64))
    print(img_array.shape)
    return new_array.reshape(64, 64, 3)
list = []
#img.append('C:/Users/Nikita/Desktop/100rand.jpg')
##image = 'C:/Users/Nikita/Desktop/100rand.jpg'
#print('done')
#img_array = Image.fromarray(image, 'RGB')
#print('done')
#resized_img = image.resize((64, 64))
#print('done')
#list.append(np.array(resized_img))
##print('done')
#list = np.array(list)

#predictions = model.predict(image)

#print(predictions)

newimg = cv2.imread('F:/currency-recognition-master/Test Images/TESTR50_1.jpg')
newimg_array = Image.fromarray(newimg, 'RGB')
resized_img = newimg_array.resize((320, 180,))
X_test = []
X_test.append(np.array(resized_img))
X_test =np.array(X_test)
Y_predict = model.predict_classes( X_test )
print(Y_predict)
