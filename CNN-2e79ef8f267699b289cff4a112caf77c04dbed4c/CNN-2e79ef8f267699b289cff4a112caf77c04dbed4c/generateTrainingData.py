# training example generation and charting function
from skimage import io, transform
import random
import numpy as np
import pandas as pd
from math import pi
from bokeh.plotting import figure, show, output_file
from bokeh.io import export_png
import os
import shutil
from copy import deepcopy

chartpath=desktop = os.path.expanduser("~/Desktop/")
chartPath=os.path.join(desktop,'CryptoCNN/charts/')
trainingdatapath=os.path.join(desktop,'CryptoCNN/CNNTrainingData/')
trainingchartspath=os.path.join(desktop,'CryptoCNN/CNNTrainingCharts/')
testdatapath=os.path.join(desktop,'CryptoCNN/CNNTestData/')
testchartspath=os.path.join(desktop,'CryptoCNN/CNNTestCharts/')
pricedatabasepath=os.path.join(desktop,'CryptoCNN/pricedatabase/')

# generate one day of financial data
# start: beginning price
# returns [date,open,close,high,low]
def generateDay(start,date):
    center = np.random.normal(.1,.75) # assuming moving forward
    deviation = .75
    numIntraDayPoints = 10
    points  = [0]*numIntraDayPoints
    points[0] = round(start + np.random.normal(center,deviation),2)
    for i in range(1,numIntraDayPoints):
        #points[i] = points[i-1] + np.random.normal(center,deviation)
        points[i] = round(start+np.random.normal(center,deviation),2)
        #while (points[i] < 1):
        #    points[i] = points[i-1] + np.random.normal(center,deviation)
    return [date,points[0],points[numIntraDayPoints-1],max(points),min(points)]

# generate multiday data
def generateTimeFrame(timeLength):
    #timeFrame = random.randint(60,360) #### MAGIC NUMBERS
    timeFrame = timeLength
    data = [0]*timeFrame
    start = random.randint(25,50) # Starting price upon which to base day 0 data
    data[0] = generateDay(start,0)
    for i in range(1,timeFrame):
        data[i] = generateDay(data[i-1][1],i) # bases opening price on prior closing
    return data


# Charting software (mostly stolen from sample website with some modifications)
def candleStick(data, nameOfFile="candlestick.png", training=True):
    df = pd.DataFrame(data, columns=["date","open","close","high","low"])
    #df["date"] = pd.to_datetime(df["date"])

    inc = df.close > df.open
    dec = df.open > df.close
    same = df.open == df.close
    #w = 12*60*60*1000 # half day in ms
    w = 1

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    #p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "Candlestick")
    p = figure(x_axis_type="linear",tools=TOOLS, plot_width=w*8*(len(df.date)), title = "Candlestick",y_range=[0,1.05*max(df.high)])
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.3

    p.segment(df.date, df.high, df.date, df.low, color="black")
    p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="red", line_color="black")
    p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="green", line_color="black")
    p.vbar(df.date[same], w, df.open[same], df.close[same], fill_color="black", line_color="black")

    p.toolbar.logo = None
    p.toolbar_location = None

    export_png(p, filename=nameOfFile)
    #os.rename("~/Desktop/CNNCrypto/"+nameOfFile, "~/Desktop/CryptoCNN/CNNTrainingCharts/")
    if (training == True):
        shutil.move(os.path.join(desktop,'CryptoCNN/misc/'+nameOfFile), trainingchartspath)
    else:
        shutil.move(os.path.join(desktop,'CryptoCNN/misc/'+nameOfFile), testchartspath)
    #output_file("candlestick.png", title="candlestick.py example")

    #show(p)  # open a browser

# finds local mins and max using a moving average
def findMinMax(data,pM=2,eps=.04):
    epsilon = eps # Controls how far point has to be above/below moving average for it to be considered a local min/max
    movingAvgLen = 16 # How many points moving average is contructed over
    highMovingAvg = [0]*(len(data)-movingAvgLen)
    lowMovingAvg = [0]*(len(data)-movingAvgLen)
    locMin = []
    locMax = []
    for i in range(0,len(highMovingAvg)):
        highSum = 0
        lowSum = 0
        for j in range(i, i + movingAvgLen):
            highSum = highSum + data[j][3]
            lowSum = lowSum + data[j][4]
        highMovingAvg[i] = highSum/(movingAvgLen)
        lowMovingAvg[i] = lowSum/(movingAvgLen)
    for i in range(int(movingAvgLen/2)-1,len(data)-int(movingAvgLen/2)):
        if (data[i][3] > (1+epsilon)*highMovingAvg[i-int(movingAvgLen/2)]):
            locMax.append(i)
        if (data[i][4] < (1-epsilon)*lowMovingAvg[i-int(movingAvgLen/2)]):
            locMin.append(i)
    return [locMin,locMax] 

# quads give indexes in form (low1,low2,high1,high2) that could form triangles
def findQuads(locMin,locMax,dataLen):
    quadsList = []
    upperDiff = int(.35*dataLen) # Upperbound on low2-low1, and high2-high1
    lowerDiff = int(.07*dataLen) # Lowerbound on low2-low1, and high2-high1
    for i in range(0,len(locMin)):
        for j in range(i+1,len(locMin)):
            for k in range(0,len(locMax)): 
                for l in range(k+1,len(locMax)):
                    (a,b,c,d) = (locMin[i],locMin[j],locMax[k],locMax[l])
                    overlap = (a < c and c < b) or (c < a and a < d) # top and bottom overlap
                    length = (b - a < upperDiff) and (d - c < upperDiff) and (b - a > lowerDiff) and (d - c > lowerDiff) # top and bottom of triangle not too long/short
                    if (overlap and length):
                        quadsList.append((locMin[i],locMin[j],locMax[k],locMax[l]))
    return quadsList

# Checks slopes of potential triangles that lowSlope > highSlope
def findValidTriangles(quadsList,data):
    validTriangles = []
    for i in range(0,len(quadsList)):
        (low1,low2,high1,high2) = quadsList[i]
        (low1Val,low2Val,high1Val,high2Val) = (data[low1][4],data[low2][4],data[high1][3],data[high2][3])
        lowSlope = (low2Val-low1Val)/(low2-low1)
        highSlope = (high2Val-high1Val)/(high2-high1)
        if (lowSlope > highSlope):
            validTriangles.append(quadsList[i])
    return validTriangles

# Check that all of the data in the triangle is contained within the boundary lines of the triangle
def checkBounds(quadsList,data):
    validTriangles = []
    error = .05 # Adjust this parameter to allow for "near miss" triangles
    for i in range(0,len(quadsList)):
        (low1,low2,high1,high2) = quadsList[i]
        (low1Val,low2Val,high1Val,high2Val) = (data[low1][4],data[low2][4],data[high1][3],data[high2][4])
        lowSlope = (low2Val-low1Val)/(low2-low1)
        highSlope = (high2Val-high1Val)/(high2-high1)
        valid = True # where triange is valid based on in-between points being in triangle
        for j in range(low1+1,low2):
            lowerBound = data[low1][4] + lowSlope*(j-low1)
            if (data[j][4] < (1-error)*lowerBound): # day goes lower than bound allows
                valid = False
        for j in range(high1+1,high2):
            upperBound = data[high1][3] + highSlope*(j-high1)
            if (data[j][3] > (1+error)*upperBound): # day goes higher than bound allows
                valid = False
        if (valid):
            validTriangles.append(quadsList[i])
    return validTriangles

# Generally classifies testing data with 80% accuaracy
def classifier(data):
    locMinAndMax = findMinMax(data) # Find local mins/max over blocks day periods
    locMin = locMinAndMax[0]
    locMax = locMinAndMax[1]
    quadsList = findQuads(locMin,locMax,len(data)) # Find potential quads of day that could form triangles
    potentialTriangles = findValidTriangles(quadsList,data) # Checks for triangle shape
    validTriangles = checkBounds(potentialTriangles,data) # Checks days in triangle for bounds requirement
    return(len(validTriangles))

# Generating examples to train/test model
"""
for i in range(0,2500):
    bound = 0 # How many counts to consider the example positive
    a = generateTimeFrame(50)
    c = classifier(a)
    result = 0
    if (c > bound): result = 1
    name = str(i)+"_"+str(result)+"_"
    candleStick(a,name+".png",True)
    df = pd.DataFrame(a,columns=["date","open","close","high","low"])
    df.to_csv(os.path.join(trainingdatapath,name+".csv"),',',index=False)
"""

# Swaps out the "Date" value from each day for an interger so the data will be compatible with candlestick function
def swapDateForIndex(data):
    old = data
    new = []
    for i in range(len(data)):
        row = old[i]
        row[0] = i
        new.append(row)
    return new

# Creates the .png file and .csv
def testDataCandlestickAndCSV(tickerSymbol,dataMatrix,csv=True):
    name = tickerSymbol+' '+dataMatrix[0][0]+'-'+dataMatrix[len(dataMatrix)-1][0]
    candleStick(swapDateForIndex(dataMatrix), nameOfFile=name+".png",training=False)
    df = pd.DataFrame(dataMatrix,columns=["date","open","close","high","low"])
    if (csv):
        df.to_csv(os.path.join(testdatapath,name+".csv"),',',index=False)

# Go through pricedatabase folder and create 50 day charts
"""
files = os.listdir(pricedatabasepath)
for i in range(len(files)):
        if (files[i][0] != "."): # deals with .DS_store
                df = pd.read_csv(os.path.join(pricedatabasepath,files[i]))
                colTitles = ["date","open","close","high","low"]
                data = ((df.drop(['time','volume'],axis=1)).reindex(columns=colTitles)).as_matrix()
                ticker = files[i].split(' ')[0]
                if (len(data) <= 50): # Don't need to split it up just put charts in CNNTestCharts
                    testDataCandlestickAndCSV(ticker,data)
                else: # Going to need to split into multiple 50 day frames
                    dataCopy = deepcopy(data)
                    dataSplit = dataCopy[0:50] # get first 50 days
                    testDataCandlestickAndCSV(ticker,dataSplit)
                    for j in range(0,int(len(data)/50)):
                        dataCopy = deepcopy(data)
                        start = 50*j + len(data) % 50
                        end = start + 50
                        dataSplit = dataCopy[start:end]
                        testDataCandlestickAndCSV(ticker,dataSplit)
"""

# Go through /CryptoCNN/CNNTestData/, create charts and put them in 
def processTestData():
    files = os.listdir(testdatapath)
    for i in range(len(files)):
        if (files[i][0] != "."): # deals with .DS_store
                df = pd.read_csv(os.path.join(testdatapath,files[i]))
                colTitles = ["date","open","close","high","low"]
                data = ((df.drop(['time','volume'],axis=1)).reindex(columns=colTitles)).as_matrix()
                ticker = files[i].split(' ')[0]
                testDataCandlestickAndCSV(ticker,data,csv=False) # Creates chart and puts it in /CNNTestCharts/


processTestData()









