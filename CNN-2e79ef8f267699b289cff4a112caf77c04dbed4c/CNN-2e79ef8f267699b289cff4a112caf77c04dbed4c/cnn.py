import skimage
#!/usr/bin/python3.6
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage import io, transform
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import shutil


chartpath=desktop = os.path.expanduser("~/Desktop/")
trainingchartspath=os.path.join(desktop,'CryptoCNN/CNNTrainingCharts/')
testdatapath=os.path.join(desktop,'CryptoCNN/CNNTestData/')
testchartspath=os.path.join(desktop,'CryptoCNN/CNNTestCharts/')
trianglechartspath=os.path.join(desktop,'CryptoCNN/charts')

# Get images from the directory /CNNTrainingCharts/ for use in model training/evaluation
def getTrainingImages(directory, start, end):
    files = os.listdir(directory)
    images = []
    values = []
    for i in range(start,end):
        if (files[i][0] != "."): # deals with .DS_store
                value = int(files[i].split('_')[1])
                images.append(transform.resize(io.imread(os.path.join(trainingchartspath,files[i]),as_grey=True),(400,600)))
                values.append(value)
    return [np.array(images),np.array(values)]

def getTestImages():
    files = os.listdir(testchartspath)
    images = [] # array of images
    names = [] # array of chart names
    for i in range(len(files)):
        if (files[i][0] != "."): # deals with .DS_store
            images.append(io.imread(os.path.join(testchartspath,files[i]),as_grey=True))
            names.append(files[i])
    return [images,names]

# Take training data and prepare it for use by the model
def processTrainingData(start, end):
    result = getTrainingImages(trainingchartspath, start, end)
    X_train = result[0]
    y_train = result[1]

    Y_train = to_categorical(y_train)
    X_train = X_train.astype('float32')
    X_train /= 255

    flattenX = []
    for i in range(len(X_train)):
        flattenX.append(X_train[i].flatten())
    X_train = np.array(flattenX)

    np.save("X_train_pre_sc",X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_sc_train = scaler.transform(X_train)

    np.save("X_train_pre_pca",X_sc_train)
    np.save("Y_train_pre_pca",Y_train)

    NCOMPONENTS = 225
    pca = PCA(n_components=NCOMPONENTS)
    pca.fit(X_sc_train)
    X_pca_train = pca.transform(X_sc_train)

    XReshape = []
    for i in range(len(X_pca_train)):
        XReshape.append(np.reshape(X_pca_train[i],(15,15)))
    X_train = np.asarray(XReshape)
    X_train = X_train.reshape(-1, 15, 15, 1)

    np.save("X_train_225",X_train)
    np.save("Y_train_225",Y_train)

# Take test data and prepare for use to evaluate the model
def processTestData(X_test_pre,Y_test_pre):
    X_test = X_test_pre
    Y_test = Y_test_pre

    Y_test = to_categorical(Y_test)

    X_test = X_test.astype('float32')
    X_test /= 255
    flattenX = []
    for i in range(len(X_test)):
        flattenX.append(X_test[i].flatten())
    X_test = np.array(flattenX)
    scaler = StandardScaler()
    X_train_pre_sc = np.load("X_train_pre_sc.npy")
    scaler.fit(X_train_pre_sc)
    X_sc_test = scaler.transform(X_test)

    NCOMPONENTS = 225
    pca = PCA(n_components=NCOMPONENTS)
    X_sc_train = np.load("X_train_pre_pca.npy")
    pca.fit(X_sc_train)
    X_pca_test = pca.transform(X_sc_test)

    XReshape = []
    for i in range(len(X_pca_test)):
        XReshape.append(np.reshape(X_pca_test[i],(15,15)))
    X_test = np.asarray(XReshape)
    X_test = X_test.reshape(-1, 15, 15, 1)

    np.save("X_test",X_test)
    np.save("Y_test",Y_test)

# Take test data prepare for prediction
def processPredictionData(X_prediction_pre):
    X_test = X_prediction_pre

    X_test = X_test.astype('float32')
    X_test /= 255
    flattenX = []
    for i in range(len(X_test)):
        flattenX.append(X_test[i].flatten())
    X_test = np.array(flattenX)
    scaler = StandardScaler()
    X_train_pre_sc = np.load("X_train_pre_sc.npy")
    scaler.fit(X_train_pre_sc)
    X_sc_test = scaler.transform(X_test)

    NCOMPONENTS = 225
    pca = PCA(n_components=NCOMPONENTS)
    X_sc_train = np.load("X_train_pre_pca.npy")
    pca.fit(X_sc_train)
    X_pca_test = pca.transform(X_sc_test)

    XReshape = []
    for i in range(len(X_pca_test)):
        XReshape.append(np.reshape(X_pca_test[i],(15,15)))
    X_test = np.asarray(XReshape)
    X_test = X_test.reshape(-1, 15, 15, 1)

    return X_test

# Train model on training data and save the model (~87% accuracy)
def trainAndSaveModel():
    X_train = np.load("X_train_225.npy")
    Y_train = np.load("Y_train_225.npy")

    batch_size = 30
    epochs = 5
    num_classes = 2

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='linear', input_shape=(15,15,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2,2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Conv2D(128,(3,3),activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128,activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model_train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    model.save('triangle_model.h5')

# Tests the model using more labeled examples (~73% accuracy)
def runModelTest():
    X_test = np.load("X_test.npy")
    Y_test = np.load("Y_test.npy")
    model = load_model('triangle_model.h5')
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Result is " + str(scores[1]*100) + "%")

# Uses the model to classify charts contained in /CNNTestCharts/, outputs result to text file
def runTestCharts(priceDataBaseSearch=False):
    testCharts = getTestImages() # [images,names]
    data = []
    associated = [] # keeps track of which chart the data is associated with
    for i in range(len(testCharts[0])):
        if (i % 100 == 0): print(str(i))
        name = testCharts[1][i]
        (height,width) = testCharts[0][i].shape
        print(testCharts[0][i].shape)
        if (width == 400): # correct size to run through CNN
            data.append(testCharts[0][i])
            associated.append(name)
        elif (width < 400): # need to scale up
            data.append(transform.resize(testCharts[0][i],(600,400)))
            associated.append(name)
        else: # (width > 400) too large, split into multiple images and analyze
            firstChartCopy = deepcopy(testCharts[0][i])
            first = testCharts[0][i][:,0:400]
            data.append(first)
            associated.append(name)
            for j in range(0,int(width/400)):
                chartCopy = deepcopy(testCharts[0][i])
                start = 400*j + width % 400
                end = start + 400
                other = testCharts[0][i][:,start:end]
                data.append(other)
                associated.append(name)

    chartData = processPredictionData(np.asarray(data))
    model = load_model('triangle_model.h5')
    predictions = list(model.predict_classes(chartData))
    results = []
    previous = ""
    for i in range(len(associated)): # go through and condense results
        if (associated[i] == previous): # chart is part of a split up larger chart
            results[len(results)-1] = min(results[len(results)-1],predictions[i])
        else: # (associated[i] != previous), chart is part of a new chart
            results.append(predictions[i])
        previous = associated[i]

    if (priceDataBaseSearch): # transfer the files to /CryptoCNN/charts
        for i in range(len(results)):
            if (results[i] == 1): # chart contained a triangle
                 shutil.move(os.path.join(desktop,'CryptoCNN/CNNTestCharts/'+testCharts[1][i]),trianglechartspath)
    else: # outputs text file
        output = ""
        for i in range(len(results)): # create string for output in text file
            output = output + testCharts[1][i] + "\t" + str(results[i]) + "\n"
        outputFile = open("output.txt","w")
        outputFile.write(output)
        outputFile.close()


runTestCharts()












