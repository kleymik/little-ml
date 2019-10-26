# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy.random import random, seed
import pdb
import yfinance as yf
import pandas as pd

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def sigmoidd(x):  return sigmoid(x) * (1 - sigmoid(x))
def sigmoidds(s, accel=1.0): return s * (1 - s) * accel
def prtArr(arr, lbl=None):
    if lbl: print('----', lbl, ' ', arr.shape)
    for rw in arr:
        for x in rw: print(f'{x:10.4f}  ', end='')
        print()
    print()

def getStockSeries(stck='AAPL'):
    # Import the yfinance. If you get module not found error
    # the run !pip install yfiannce from your Jupyter notebook

    # Get the data of the stock AAPL
    # data = yf.download('AAPL','2016-01-01','2019-08-01')
    # Plot the close price of the AAPL
    # data['Adj Close'].plot()

    # tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
    tickers_list = ['AAPL', 'IBM']
    df = pd.DataFrame(columns=tickers_list)
    # Fetch the data
    for ticker in tickers_list: df[ticker] = yf.download(ticker,'2019-01-01','2019-08-01')['Adj Close']
    df = df.resample('W').last()  # Weekly Data
    df.to_csv('./yfData.csv')
    print(df.head)
    return df

def getCheckDummyData():
    IN = np.array([[ 0.0, 0.5, 1.0, 1.0],   # 5 X lev0Num
                   [ 1.0, 1.0, 1.0, 1.0],
                   [ 1.0, 0.0, 1.0, 0.0],
                   [ 1.0, 0.8, 1.0, 0.2],
                   [ 0.0, 1.0, 1.0, 0.0]])
    OUT = np.array([[0.5, 1.0],            # 5 x lev2Num
                    [1.0, 0.0],
                    [1.0, 0.3],
                    [0.6, 0.8],
                    [0.9, 0.0]])
    return IN, OUT


def main(IN=None, OUT=None, lev1Num=4):  # number of middle-level neurons
    # lev0Num = 4
    # lev1Num = 4
    # lev2Num = 2

    if IN is None:
        IN = np.array([[ 0.0, 0.5, 1.0, 1.0],   # 5 X lev0Num
                       [ 1.0, 1.0, 1.0, 1.0],
                       [ 1.0, 0.0, 1.0, 0.0],
                       [ 1.0, 0.8, 1.0, 0.2],
                       [ 0.0, 1.0, 1.0, 0.0]])
    lev0Num = IN.shape[1]                      # cols=number of indep inputs
    if OUT is None:
        OUT = np.array([[0.5, 1.0],            # 5 x lev2Num
                        [1.0, 0.0],
                        [1.0, 0.3],
                        [0.6, 0.8],
                        [0.9, 0.0]])
    lev2Num = OUT.shape[1]                     # cols=number of indep inputs

    seed(1)
    syn01 = 2 * random((lev0Num, lev1Num)) - 1
    syn12 = 2 * random((lev1Num, lev2Num)) - 1

    print('------------- Start')
    prtArr(IN,    'IN')
    prtArr(syn01, 'syn01')
    prtArr(syn12, 'syn12')
    prtArr(OUT,   'OUT')

    for j in range(int(1e8)):
        # fwd prop the reponse
        lev1 = sigmoid(  IN @ syn01)    # Num
        lev2 = sigmoid(lev1 @ syn12)
        # pdb.set_trace()
        # back prop the error
        outErrs = (OUT - lev2)
        lev2Dlta = outErrs            * sigmoidds(lev2, accel=1.0)
        lev1Dlta = lev2Dlta @ syn12.T * sigmoidds(lev1, accel=1.0)
        # update synapse weights
        syn12 += lev1.T @ lev2Dlta
        syn01 += IN.T   @ lev1Dlta
        if (j % 10000==0):
            print(f'{j} ', end='')
            print('err=',(outErrs * outErrs).sum())
        if (outErrs * outErrs).sum() < 1e-5: # sum of sqrd errors
            print(outErrs)
            print()
            break

    print('------------- Done')
    prtArr(IN,    'IN')
    prtArr(syn01, 'syn01')
    prtArr(lev1,  'lev1')
    prtArr(syn12, 'syn12')
    prtArr(lev2, 'lev2')
    prtArr(OUT,   'OUT')
    print('err=',(outErrs * outErrs).sum())

if True: df = getStockSeries(stck='AAPL')  # as weekly
else:    df = pd.read_csv('./yfData.csv')
print(df.head)
print(df.shape)
# 1=dummy variable to identify AAPL
# inSet = [ np.append(df['AAPL'].loc[-di:-di+6].values, 1) for di in range(-30,-20) ] # 10 trainig sample of six consectutive weeks
numTrain     =  15 # number of training samples
tsPttrnLen   =   4 # length of input pattern timeseries
seriesStrt   = -25
scaleFactor  = 1e-3
inSet = [ np.append(df[di:di+tsPttrnLen].AAPL.values * scaleFactor, 1)
          for di in range(seriesStrt, seriesStrt + numTrain) ] # 10 training sample of six consecutive wkly rtns
# TBD convert into Buy Sell: increase => Sell, decrease=>Buy
otSet = [ np.append(df[di+tsPttrnLen:di+tsPttrnLen+2].AAPL.values * scaleFactor, 1)
          for di in range(seriesStrt,seriesStrt + numTrain) ] # try to predict next 2 wkly rtns

main(IN=np.vstack(inSet), OUT=np.vstack(otSet), lev1Num=10)
#main(IN=df.values, OUT=dfb.values, lev1Num=4)

# main()
#for v in res['Adj Close'][-100:]: print(v)


