from math import pi
import pandas as pd
import bokeh
import numpy as np
import numpy.random as npr
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import MSFT
import bokeh.io

def genBoundTri(mean, length, delta, vol):
    #vol is volatility, not volume, delta is the converging upper and lower boundaries
    #mean is the central value, length is the time frame of the pattern
    path = "C:/CryptoCNN/pricedatabase/ETH_USD.csv"
    df = pd.read_csv(path, header = 0, sep =',',parse_dates = True)[:length]
    low = mean - delta
    high = mean + delta
    df.open[0] = low + delta * npr.random()
    upward = -1
    if (npr.random() > .5):
        upward = 1
    df.close[0] = npr.random() * upward *vol  * df.open[0] + df.open[0]
    for i in range(1,length):
        low = mean - delta + i/length * delta
        high = mean + delta - i/length*delta
        if(df.close[i-1] < low):
            upward = upward * -1
        df.open[i] = df.close[i-1]
        df.close[i]  = (npr.randn() + upward/2) * upward * vol * (1-i/length)  * df.open[i] + df.open[0]
    return df
def genDecVar(mean, length, delta, sd):
    #sd is like volatility, but works into the calculation a bit differently, delta is the converging upper and lower boundaries
    #mean is the central value, length is the time frame of the pattern
    path = "C:/CryptoCNN/pricedatabase/ETH_USD.csv"
    df = pd.read_csv(path, header = 0, sep = ',', parse_dates = True)[:length]
    low = mean - delta
    upward = 0
    high = mean + delta
    df.open[0] = low + delta *npr.random()
    df.close[0] = min(high, max(low, low + upward + npr.randn() * 2 * delta * sd))
    upward = 0
    for i in range(1,length):
        low = mean - delta + i/length * delta
        high = mean + delta - i/length * delta
        df.open[i] = df.close[i-1]
        nextclose = min(high, max(low, df.open[i])) +upward + npr.randn() * delta * sd
        upward = 0
        if(nextclose> high):
            nextclose = high
            upward = -.13
        if(nextclose<low):
            nextclose = low
            upward = .13
        df.close[i] = nextclose
    return df
def genRandom(mean, length, vol):
    path = "C:/CryptoCNN/pricedatabase/ETH_USD.csv"
    df = pd.read_csv(path, header = 0, sep = ',', parse_dates = True)[:length]
    df.open[0] = (npr.randn() * .1 + 1) * mean
    df.close[0] = df.open[0] * (1+npr.randn()*vol)
    for i in range(1,length):

        df.open[i] = df.close[i-1]
        df.close[i] = df.open[i] * (1+npr.randn()*vol)
    return df
def writeToFile(df, dirname):
    df["date"]= pd.to_datetime(df["date"])
    inc = df.close > df.open
    dec = df.open > df.close
    w = 12*60*60*1000 # half day in ms
    p = figure(x_axis_type="datetime", plot_width=450, plot_height = 210)
    #p.xaxis.major_label_orientation = pi/4
    p.axis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None
    p.grid.grid_line_alpha=0
    #p.segment(df.date, df.high, df.date, df.low, color="black")
    p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")
    fname = dirname + str(npr.random() * 10000000) + ".png"
    bokeh.io.export_png(p, filename=fname)

min_mean = 40
meanamp = 120
count = 0
rands  = 10
var = 0
for x in range(0,1000):

    mean = npr.rand()*meanamp + min_mean
    for i in range(0,var):
        df = genBoundTri(mean, int(30  + npr.random() * 90) , (.2 + npr.random()*.2)*mean , .4 + npr.random() * .18)
        #i actually reversed the names of functions to generate symmetric triangles
        #genDecVar generates the more strict triangle shapes by having a max and min allowed price
        #genBoundTri generates more realistic looking charts by decreasing variance of prices over time
        #genRandom generates adversarial examples by just generate random price movements

        writeToFile(df, "varplots/")
        count += 1
        print(count)


    for l in range(0,rands):
        df = genRandom(mean,int( 40 + npr.random() * 80), .02 + npr.random()*.1)
        writeToFile(df, "noise/")
        count+=1
        print(count)
