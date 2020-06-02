## -*- coding: utf-8 -*-
#"""
#Created on Sun Apr 30 09:25:59 2017
#
#@author: Vangelis
#"""

##original code by Pawel Lachowicz, QuantAtRisk.com,
## adapted by Vangelis@sanzprophet.com 
# 
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, time
import json
from bs4 import BeautifulSoup
import requests
import os
#from win32com.client import Dispatch
import time as mod_time
# 
############### SET PATHS FOR CSV FILES
datapath=desktop = os.path.expanduser("~/Desktop/")
dataPath=os.path.join(desktop,'CryptoCNN/pricedatabase/') # Where to save price data
dataPathMisc = os.path.join(desktop,'CryptoCNN/misc/') # Where to save Market Cap text file

#if path does not exist create one : Desktop/Data/Cryptcompare_Crypto/
if not os.path.exists(dataPath):
    os.mkdir(dataPath)
    print('creating Directory ...'+dataPath)

 
 ################### FUNCTIONS ###########################
def timestamp2date(timestamp):
    # function converts a Unix timestamp into Gregorian date
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
 
def date2timestamp(date):
    # function coverts Gregorian date in a given format to timestamp
    return datetime.strptime(date_today, '%Y-%m-%d').timestamp()
 
def fetchCryptoOHLC(fsym, tsym):
    # function fetches a crypto price-series for fsym/tsym and stores
    # it in pandas DataFrame
 
    cols = ['date', 'time', 'open', 'high', 'low', 'close','volume']
    lst = ['time', 'open', 'high', 'low', 'close','volumeto']
 
    #timestamp_today = datetime.today().timestamp()
    
    timestamp_today =mod_time.mktime(datetime.today().timetuple()) + datetime.today().microsecond / 1e6
    

    curr_timestamp = timestamp_today
 
    for j in range(2):
        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(int(curr_timestamp)) + "&limit=2000"
        response = requests.get(url)
        
        print(response)
        
        if response.status_code== 200 :
        
            soup = BeautifulSoup(response.content, "html.parser")
            dic = json.loads(soup.prettify())
            print(len(dic['Data']))
            for i in range(1, 2001):
                tmp = []
                for e in enumerate(lst):
                    x = e[0]
                    y = dic['Data'][i][e[1]]
                    if(x == 0):
                        tmp.append(str(timestamp2date(y)))
                    tmp.append(y)
                if(np.sum(tmp[-4::]) > 0):
                    df.loc[len(df)] = np.array(tmp)
            df.index = pd.to_datetime(df.date)
            df.drop('date', axis=1, inplace=True)
            curr_timestamp = int(df.ix[0][0])
            if(j == 0):
                df0 = df.copy()
            else:
                data = pd.concat([df, df0], axis=0)
                
        else: #if error response is not 200. problem return empty df (this needs work)
            data = df
    return data
    
 
 ############################################# START PROGRAM FLOW    #################   
    
# 1. GET LIST OF CRYPTOCURRENCIES BY MARKET CAP
    
url = "https://api.coinmarketcap.com/v1/ticker/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
dic = json.loads(soup.prettify())
 
# create an empty DataFrame
df = pd.DataFrame(columns=["Ticker", "MarketCap"])
 
for i in range(len(dic)):
    if (dic[i]['symbol'] != "NANO"): # At time of modifying this code NANO is not working
        df.loc[len(df)] = [dic[i]['symbol'], dic[i]['market_cap_usd']]
 
df.sort_values(by=['MarketCap'])
# apply conversion to numeric as 'df' contains lots of 'None' string as values
df.MarketCap = pd.to_numeric(df.MarketCap)

P = df[df.MarketCap > 40e6]  #only coins above 40 mill cap (customize as you wish)
P.to_csv(os.path.join(dataPathMisc,"CryptoTickersByCap.txt"),",");
#print(P)#, end="\n\n")
 
portfolio = list(P.Ticker)
print(portfolio)

##################################################
#Set base currency
tsym='USD'

####################################################

# Go through list of coins, download prices for each and write to csv
for e in enumerate(portfolio):
  #e[0] is the index capitalization rank (1,2,3,4,6,etc)
  #e[1] is the ticker of the  coin ('BTC', 'ETH', etc )
    print(e[0], e[1])
    
    #temp fix as Cryptocompare does not have data for MIOTa and LKK (Lykke)
    if e[1].startswith("MIOTA") or e[1].startswith("LKK"):
        continue
        
    data = fetchCryptoOHLC(e[1], tsym) #i.e. fetchCryptoOHLC('ETH', 'USD')
    
    # Formatting dates for file names
    startDate = str(list(data.index)[0]).split(' ')[0].split('-')
    endDate = str(list(data.index)[len(list(data.index))-1]).split(' ')[0].split('-')
    startYear = startDate[0]
    startMonth = startDate[1]
    startDay = startDate[2]
    endYear = endDate[0]
    endMonth = endDate[1]
    endDay = endDate[2]

    fileend=e[1]+' '+startYear+startMonth+startDay+"-"+endYear+endMonth+endDay+'.csv'
    print(fileend)
    #print(data.iloc[0,0])
    #print(data.iloc[1,0])
    #if (data.iloc[0,0] == data.iloc[1,0]):
    #    print(data.count)
    #    data.drop(data.index[1])
    #    print(data.count)
    data = data.drop_duplicates()
    data.to_csv(os.path.join(dataPath,fileend),',')
    
############################################################

















