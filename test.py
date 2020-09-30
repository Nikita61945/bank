import cv2

import os

import numpy as np
from PIL import Image
import random

X = []

labels = []

for dirname, _, filenames in os.walk('F:/comp7022/Dataste/BN/'):
    for filename in filenames:
       # print(dirname)
       # print(filename)
        if (len(str(os.path.join(dirname, filename)).split('/')) > 2):

            test = os.path.join(dirname, filename)

            image = cv2.imread(os.path.join(dirname, filename)) #reads single image
            #image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image=cv2.GaussianBlur((image), (5, 5), 0)
            img_array = Image.fromarray(image,'RGB')
            #print(img_array)

            resized_img = img_array.resize((320, 180))
            #print(resized_img)

            X.append(np.array(resized_img))
            #print(np.array(resized_img)) --> np array

            label = str(os.path.join(dirname)).split('/')[-1] #gets label i.e 10, 20
           # print('label='+label)
            if (label == 'Training Images'):

                labels.append(0)
                print("zero")

            else:
                print("one")
                labels.append(1)

# Any results you write to the current directory are saved as output.

print( 'Number of images : %d' % len(X))

print( 'Associated labels : %d' % len( labels ))

X = np.array( X )

np.random.shuffle(X)

print( 'Shape of X : ', X.shape )

Y = np.array( labels )

Y = np.reshape( Y, ( 586, 1 ) )

print( 'Shape of Y : ', Y.shape )

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, random_state = 10, test_size = 0.2 )

print( 'X train shape : ', X_train.shape )

print( 'Y train shape : ',Y_train.shape )

from keras.utils import to_categorical

Y_train = np.reshape( Y_train, ( 468, 1 ) ) #y train shape value

print( 'Y train shape : ',Y_train.shape )

Y_train = to_categorical( Y_train, 2 )

Y_train.shape

print( 'Y train shape : ',Y_train.shape )

from keras import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, ZeroPadding2D, AveragePooling2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras import backend as K

K.clear_session()

model = Sequential()
#model.add(Conv2D(input_shape=(180, 320 ,3),padding='same',kernel_size=3,filters=16))
#model.add(LeakyReLU(0.1))
#model.add(Conv2D(padding='same',kernel_size=3,filters=32))
#model.add(LeakyReLU(0.1))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(padding='same',kernel_size=3,filters=32))
#model.add(LeakyReLU(0.1))
#model.add(Conv2D(padding='same',kernel_size=3,filters=64))
#model.add(LeakyReLU(0.1))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(256))
#model.add(Dropout(0.5))
#model.add(LeakyReLU(0.1))
#model.add(Dense(6))
#model.add(Activation('softmax'))
model.add(ZeroPadding2D(input_shape=(180, 320,3), padding=(3, 3)))
print("114")
model.add(Conv2D(32, (7, 7), strides=(1, 1)))
print("116")
model.add(BatchNormalization(axis=3))

model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(2))

model.add(Activation('softmax'))

model.summary()

model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'] )

model.fit( X_train, Y_train, batch_size = 15, epochs = 20)

#newimg = 'C:/Users/Nikita/Desktop/100rand2'
#newimg_array = Image.fromarray(newimg, 'RGB')
#resized_img = newimg_array.resize((64, 64))
#X_test = []
#X_test.append(np.array(resized_img))
Y_predict = model.predict_classes( X_test )

print( 'X test shape : ',X_test.shape)

print( 'Y predict shape : ',Y_predict.shape )

from sklearn.metrics import accuracy_score

accuracy_score( Y_test, Y_predict )

from sklearn.metrics import classification_report

print( classification_report( Y_test, Y_predict ) )

#save model
from tensorflow.keras.models import Sequential, save_model, load_model
filepath = 'F:/comp7022/saved_model'
save_model(model, filepath)

#model = load_model(filepath, compile = True)

#sample = []
#sample.append('C:/Users/Nikita/Desktop/100rand.jpg')
#sample = np.array(sample)
#print(sample.shape)
##predictions = model.predict(sample)
#print(predictions)
#classes = np.argmax(predictions, axis = 3)
#print(classes)


