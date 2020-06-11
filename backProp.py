# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy.random import random, seed

def sigmoid(x):   return 1 / (1+np.exp(-x))
def sigmoidd(x):  return sigmoid(x) * (1 - sigmoid(x)) # derivative of sigmoid in terms of sigmoid
def sigmoidds(s): return s * (1 - s)

def printArr(arr, lbl=None):
    if lbl: print('----',lbl)
    for rw in arr:
        for x in rw: print(f'{x:4.3f}  ', end='')
        print()
        # print(sum(abs(lev2Dlta)))
    print()

def go(IN, OUT):

    inSize  = 4  # input dimensionality
    midSize = 4  # m 1 middle/ hidden layer
    outSize = 1  # output dimensionality
    seed(1)

    syn01 = 2*np.random.random((inSize,  midSize)) - 1  # 12 +/- 1 adjustable loadings on links gives 3x4 cross-wire to hidden layer
    syn12 = 2*np.random.random((midSize, outSize1)) - 1 #  4 +/- 1 adjustable loadings onto one output node

    lev0 = IN                               # 4 samples of input triple

    print('------------- Start')
    printArr(IN,    'IN')
    printArr(syn01, 'syn01')
    printArr(syn12, 'syn12')
    printArr(OUT,   'OUT')

    for j in range(int(1e5)):

        # forward prop the response
        lev1 = sigmoid(lev0 @ syn01)         # lev1 = level 1 neuron state
        lev2 = sigmoid(lev1 @ syn12)

        # backprop the error
        lev2Delta = (outErrs := (OUT - lev2)) * sigmoidds(lev2)
        syn12 += lev1.T @ lev2Delta          # orignal code has this after next statement(!?)

        lev1Delta = lev2Delta @ syn12.T       * sigmoidds(lev1)
        syn01 += lev0.T @ lev1Delta

        if j % 100 == 0: print('SQERR=', j, (ssq:= outErrs.T @ outErrs)[0][0])
        if (ssq) < 1e-4: # sum of squared errors
            print(outErrs.T)
            print()
            break

    print('RES=',   lev2)
    print('SQERR=', ssq)
    printArr(IN)

if __name__ == '__main__':

    IN = np.array([[0, 0, 1, 1],            # 4 samples of input triple
                   [1, 1, 1, 1],
                   [0, 1, 1, 1],
                   [1, 0, 1, 0]])

    OUT = np.array([[0.5,                   # desired output for each sample
                       1,
                     0.75,
                     0.5]]).T

    go(IN, OUT)
