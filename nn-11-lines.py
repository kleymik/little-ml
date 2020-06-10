# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy.random import random, seed
import pdb
import yfinance as yf
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt


# sigmoid
def sigmoid(x):   return 1 / (1 + np.exp(-x))
def sigmoidd(x):  return sigmoid(x) * (1 - sigmoid(x))
def sigmoidds(s, accel=1.0): return s * (1 - s) * accel

# hyperbolic tangent
def hyptan(x):   return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def hyptandh(h): return (1 - h**2)  #  1 - tanh^2(x)

# ReLU

def prtArr(arr, lbl=None, flatP=False):
    if lbl: print('----', lbl, ' ', arr.shape)
    for rw in arr:
        for x in rw: print(f'{x:5.3f}', end='')
        if flatP: print(f'|', end='')
        else:     print()
    if flatP: print(f'', end='')
    else:     print()

# Data Sets

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
    for ticker in tickers_list:
        df[ticker] = yf.download(ticker,'2019-01-01','2019-08-01')['Adj Close']
    df = df.resample('W').last()  # Weekly Data
    df.to_csv('./yfData.csv')
    print(df.head)
    return df

def getCheckDummyData0():
    IN = np.array([[ 0.0, 0.0, 1.0],      # 4 X lev0Num
                   [ 1.0, 1.0, 1.0],
                   [ 1.0, 0.0, 1.0],
                   [ 0.0, 1.0, 1.0]])
    OUT = np.array([[0.0],                # 4 x lev2Num
                    [1.0],
                    [1.0],
                    [0.0]])
    return IN, OUT

def getCheckDummyData5():
    IN = np.array([[ 0.0, 0.5, 1.0, 1.0], # 5 X lev0Num
                   [ 1.0, 1.0, 1.0, 1.0],
                   [ 1.0, 0.0, 1.0, 0.0],
                   [ 1.0, 0.8, 1.0, 0.2],
                   [ 1.0, 0.4, 1.0, 0.0],
                   [ 0.0, 1.0, 1.0, 0.0]])
    OUT = np.array([[0.5, 1.0],           # 5 x lev2Num
                    [1.0, 0.0],
                    [1.0, 0.3],
                    [0.6, 0.8],
                    [0.6, 0.9],
                    [0.9, 0.0]])
    return IN, OUT

def genRandomCrossings():
    ''' create a set of training examples
        random series, count the number above and the number below
    '''
    # series = np.rand.rand(8)
    # seriesMean = series.mean()
    # seriesAbove = series > seriesMean
    # seriesBelow = series < seriesMean
    pass

def train(IN=None, OUT=None, midLayer=3, iterLim=1e6, errThresh=1e-4):  # number of middle-level neurons
    lev0Num = IN.shape[1]        # 4 cols=number of indep inputs
    lev1Num = midLayer           # 4 middle layer
    lev2Num = OUT.shape[1]       # 2 cols=number of indep inputs

    #seed(1)
    syn01 = 2 * random((lev0Num, lev1Num)) - 1
    syn12 = 2 * random((lev1Num, lev2Num)) - 1

    print('------------- Start')
    prtArr(IN,  '  IN    = numTrials   x numInputs')
    prtArr(syn01, 'syn01 = numInputs   x numMidLayer')
    prtArr(syn12, 'syn12 = numMidLayer x numOutputs')
    prtArr(OUT,   'OUT   = numTrials   x numOutputs')

    #pd.DataFrame(syn01.flatten().T).shape
    wghtsLog = pd.DataFrame([syn01.flatten()])

    for j in range(int(iterLim)):

        lev1 = sigmoid(  IN @ syn01)  # fwd prop the response
        lev2 = sigmoid(lev1 @ syn12)
                                      # pdb.set_trace()
        outErrs = (OUT - lev2)        # back prop the error
        lev2Dlta = outErrs            * sigmoidds(lev2, accel=1.0)
        lev1Dlta = lev2Dlta @ syn12.T * sigmoidds(lev1, accel=1.0)

        syn12 += lev1.T @ lev2Dlta    # update synapse weights
        syn01 += IN.T   @ lev1Dlta
        totErr = (outErrs * outErrs).sum()
        if (j % 1000==0):
            print(f'{j:12.0f}, ', end='')
            print(f'err={totErr:10.5f}, ', end='')
            prtArr(syn01, flatP=True)
            wghtsLog = wghtsLog.append([syn01.flatten()])  # prtArr(syn12, flatP=True)
            print()
        if totErr < errThresh: # sum of sqrd errors
            print(outErrs)
            print()
            break

    print('------------- Done')

    prtArr(IN,    'IN    = numTrials   x numInputs')
    prtArr(syn01, 'syn01 = numInputs   x numMidLayer')
    prtArr(lev1,  'lev1  = numTrials   x numMidLayer')
    prtArr(syn12, 'syn12 = numMidLayer x numOut')
    prtArr(lev2,  'lev2  = numTrials   x numOutputs' )
    prtArr(OUT,   'OUT   = numTrials   x numOuputs')
    print(f'numiters={j}: err = {(outErrs * outErrs).sum()}')

    wghtsLog = wghtsLog.reset_index()
    # plt.plot(wghtsLog.iloc[:, :3])
    plt.plot(wghtsLog)
    plt.show()


def main():

    numTrain     =  15 # number of training samples
    tsPttrnLen   =   4 # length of input pattern timeseries
    seriesStrt   = -25
    scaleFactor  = 1e-3

    if True:
        IN, OUT = getCheckDummyData5()
        train(IN=IN, OUT=OUT, midLayer=3, iterLim=1e6, errThresh=1e-6)
    if False:
        if True: df = getStockSeries(stck='AAPL')  # as weekly
        else:   df = pd.read_csv('./yfData.csv')
        print(df.head)
        print(df.shape)
        # 1=dummy variable to identify AAPL
        # inSet = [ np.append(df['AAPL'].loc[-di:-di+6].values, 1)
        #   for di in range(-30,-20) ] # 10 trainig sample of 6 consecve weeks
        inSet = [ np.append(df[di:di+tsPttrnLen].AAPL.values * scaleFactor, 1) for di in range(seriesStrt, seriesStrt + numTrain) ]
        # 10 training sample of six consecutive wkly rtns
        # TBD convert into Buy Sell: increase => Sell, decrease=>Buy
        otSet = [ np.append(df[di+tsPttrnLen:di+tsPttrnLen+2].AAPL.values * scaleFactor, 1) for di in range(seriesStrt,seriesStrt + numTrain) ]
        # try to predict next 2 wkly rtns
        train(IN=np.vstack(inSet), OUT=np.vstack(otSet), lev1Num=10)
        #train(IN=df.values, OUT=dfb.values, lev1Num=4)
        #for v in res['Adj Close'][-100:]: print(v)

main()

# if True: df = getStockSeries(stck='AAPL')  # as weekly
# else: df = pd.read_csv('./yfData.csv')
# print(df.head)
# print(df.shape)
# # 1=dummy variable to identify AAPL
# # inSet = [ np.append(df['AAPL'].loc[-di:-di+6].values, 1) for di in range(-30,-20) ] # 10 trainig sample of six consectutive weeks
# numTrain   =  15 # number of training samples
# tsPttrnLen   =   4 # length of input pattern timeseries
# seriesStrt   = -25
# scaleFactor  = 1e-3
# inSet = [ np.append(df[di:di+tsPttrnLen].AAPL.values * scaleFactor, 1)
#          for di in range(seriesStrt, seriesStrt + numTrain) ] # 10 training sample of six consecutive wkly rtns
# # TBD convert into Buy Sell: increase => Sell, decrease=>Buy
# otSet = [ np.append(df[di+tsPttrnLen:di+tsPttrnLen+2].AAPL.values * scaleFactor, 1)
#          for di in range(seriesStrt,seriesStrt + numTrain) ] # try to predict next 2 wkly rtns


# junk
    # if IN is None:
    #    IN = np.array([[ 0.0, 0.5, 1.0, 1.0],   # 5 X lev0Num
    #                   [ 1.0, 1.0, 1.0, 1.0],
    #                   [ 1.0, 0.0, 1.0, 0.0],
    #                   [ 1.0, 0.8, 1.0, 0.2],
    #                   [ 0.0, 1.0, 1.0, 0.0]])
    # if OUT is None:
    #    OUT = np.array([[0.5, 1.0],            # 5 x lev2Num
    #                    [1.0, 0.0],
    #                    [1.0, 0.3],
    #                    [0.6, 0.8],
    #                    [0.9, 0.0]])
