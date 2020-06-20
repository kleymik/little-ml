#
# succinct rendition of backprop in a 3 layer ANN
# based on / derivative of:
#  http://iamtrask.github.io/2015/07/12/basic-python-network/
# kleymik 20200105

import numpy as np
from numpy.random import random, seed

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def sigmoidds(s): return s * (1 - s)

def printArr(arr, lbl=None):
    if lbl: print(f'---- {lbl}  {arr.shape[0]} X {arr.shape[1]}')
    for rw in arr:
        for x in rw: print(f'{x: 5.4f}  ', end='')
        print()
    print()

def go(IN, OUT, midSize=4, pnSteps=100, errThresh=1e-5, maxSteps=1e5):

    seed(1)                                            # initialise the adjustable weights between +1.0 and -1.0
    syn01 = 2*random((IN.shape[1],  midSize)) - 1      # cross synapses to hidden layer "lev1"
    syn12 = 2*random((midSize, OUT.shape[1])) - 1      # cross synapses to output layer "lev2"

    lev0 = IN                                          # 4 samples of input triple

    print('------------- Start')

    for j in range(int(1e5)):

        lev1 = sigmoid(lev0 @ syn01)                   # forward propagate the response;  lev1 = level 1 neurons state
        lev2 = sigmoid(lev1 @ syn12)
                                                       # back propagate the error
        syn12 += lev1.T @ (lev2Delta := ((lev2Errs := (OUT - lev2))          * sigmoidds(lev2)))
        syn01 += lev0.T @ (lev1Delta := ((lev1Errs := (lev2Delta @ syn12.T)) * sigmoidds(lev1)))

        if j % pnSteps == 0:
            print(f'Minimisation Step: %8g' % j,
                   ', LEV1 SSQERR= %12.10f' % ((lev1Errs.T        @ lev1Errs)[0][0]),
                   ', LEV2 SSQERR= %12.10f' % ((ssqe:= lev2Errs.T @ lev2Errs)[0][0]))  # sum of squared errors
        if (ssqe) < 1e-4:
            print(lev2Errs.T)
            print()
            break

    printArr(IN,    'IN')
    printArr(syn01, 'syn01')
    printArr(syn12, 'syn12')
    printArr(OUT,   'OUT')
    printArr(lev2,  'RES')
    printArr(ssqe,  'SSQERR')

if __name__ == '__main__':

    go(np.array([[0, 0, 1],                            # IN: 4x3: 4 samples of input triple
                 [1, 1, 1],
                 [0, 1, 1],
                 [1, 0, 1]]),
       np.array([[0.5],                                 # OUT: 4x1: desired output for each sample
                 [1],
                 [0.75],
                 [0.5]]),
       midSize=3,                                      # num neurons in middle layer
       pnSteps=50,                                     # print error after each pnSteps of minimisation
       errThresh=1e-4,                                 # terminate when sum-squares error is below errThresh
       maxSteps=1e5)                                   # terminate after maxSteps number of mimimisastion steps



