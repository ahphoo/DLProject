import keras
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil

triangles = 6000 #number of triangles for training set - feel free to increase if you have a good setup
noise = 8000 #number of non-triangles for training set - feel free to increase if you have a good setup
print("Getting filenames")
x1 = os.listdir("triangle")
x1 = x1[0:triangles]
x2 = os.listdir("noise")
x2 = x2[0:noise]
print("Converting training set images to image feature arrays")
train_X = []
count = 0
for fname in x1:
    if(fname != "._.DS_Store" and fname!= ".DS_Store"):

        img= Image.open('triangle/' + fname).convert('LA')
        #I converted to greyscale and decreased image size because I'm running on a slow laptop
        #img = img.rotate(180)
        #run a second time with rotated uncommented, and unrotated commented to get leftward triangles
        img = np.array(img.resize((60,100)))
        train_X.append(img)
y1 = [1]* len(train_X) #consider triangles to be class 1
for fname in x2:
    if(fname != ".DS_Store"):

        img = Image.open('noise/' + fname).convert('LA')
        img = np.array(img.resize((60,100)))
        train_X.append(img)
y2 = [0]*(len(train_X) - len(y1)) #consider everything else to be class 2
train_X = np.array(train_X)

train_Y = y1+y2
train_Y = np.array(train_Y)


train_X = train_X.astype('float32')
train_Y= train_Y.astype('float32')
train_X = train_X / 255.
#test_X = test_X /255.

train_Y = to_categorical(train_Y)
#test_Y_one_hot = to_categorical(test_Y)
train_X, valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size = .2, random_state= 13)

batch_size = 64
epochs = 2
num_classes =2

#CNN model - credit to https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python (the first tutorial you linked)
mymodel = Sequential()
mymodel.add(Conv2D(32, kernel_size=(3,3), activation = 'linear', input_shape =(100,60,2), padding = 'same'))
mymodel.add(LeakyReLU(alpha=.1))
mymodel.add(MaxPooling2D((2,2), padding = 'same'))
mymodel.add(Dropout(.25))
mymodel.add(Conv2D(64,(3,3), activation='linear', padding = 'same'))
mymodel.add(LeakyReLU(alpha=.1))
mymodel.add(MaxPooling2D((2,2), padding = 'same'))
mymodel.add(Dropout(.25))
mymodel.add(Conv2D(128,(3,3), activation='linear', padding = 'same'))
mymodel.add(LeakyReLU(alpha=.1))
mymodel.add(MaxPooling2D((2,2), padding = 'same'))
mymodel.add(Dropout(.3))
mymodel.add(Flatten())
mymodel.add(Dense(128, activation='linear'))
mymodel.add(LeakyReLU(alpha=.1))
mymodel.add(Dropout(.3))
mymodel.add(Dense(num_classes,activation='softmax'))
mymodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
mymodel.summary()

model_train = mymodel.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1, validation_data = (valid_X, valid_label))
path = "C:/CryptoCNN/"
x3 = os.listdir(path + "actualplots/")
test_X = []
print("Using trained model to find triangles in actual crypto price histories")
for fname in x3:
    if(fname != ".DS_Store"):
        img= Image.open('actualplots/' + fname).convert('LA')
        img = np.array(img.resize((60,100)))
        #img.resize(0, 450, 4)
        test_X.append(img)

test_X = np.array(test_X)
predicted_classes = mymodel.predict(test_X)
count = 0
print("Found triangles are being placed in directory CryptoCNN/foundtriangles/")
for i in predicted_classes:
    if(i[1] > .2):
        shutil.copy2(path + "actualplots/" + x3[count], path + "foundtriangles/")
    count += 1
