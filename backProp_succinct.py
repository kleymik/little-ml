#
# succinct rendition of 3 layer ANN
# based on:
#  http://iamtrask.github.io/2015/07/12/basic-python-network/
# kleymik 20200105

import numpy as np
from numpy.random import random, seed

def sigmoid(x):   return 1 / (1+np.exp(-x))
def sigmoidds(s): return s * (1 - s)

def printArr(arr, lbl=None):
    if lbl: print('----',lbl)
    for rw in arr:
        for x in rw: print(f'{x:4.3f}  ', end='')
        print()
    print()

def go(IN, OUT, midSize=4):

    seed(1)                                            # initialise adjustable loadings bertween +/-1
    syn01 = 2*random((IN.shape[1],  midSize)) - 1      # on cross synapses to hidden layer "lev1"
    syn12 = 2*random((midSize, OUT.shape[1])) - 1      # on cross synapses to output layer "lev2"

    lev0 = IN                                          # 4 samples of input triple

    print('------------- Start')

    for j in range(int(1e5)):

        lev1 = sigmoid(lev0 @ syn01)                   # forward propagate the response;  lev1 = level 1 neurons state
        lev2 = sigmoid(lev1 @ syn12)
                                                       # backpropagate the error
        syn12 += lev1.T @ (lev2Delta := ((outErrs := (OUT - lev2)) * sigmoidds(lev2)))
        syn01 += lev0.T @ (lev1Delta := (lev2Delta @ syn12.T       * sigmoidds(lev1)))

        if j % 100 == 0: print('SSQERR=', j, (ssqe:= outErrs.T @ outErrs)[0][0])  # sum of squared errors
        if (ssqe) < 1e-4:
            print(outErrs.T)
            print()
            break

    printArr(IN,    'IN')
    printArr(syn01, 'syn01')
    printArr(syn12, 'syn12')
    printArr(OUT,   'OUT')

    print('RES=',    lev2)
    print('SSQERR=', ssqe)

if __name__ == '__main__':

    go(np.array([[0, 0, 1, 1],                         # IN: 4 samples of input triple
                 [1, 1, 1, 1],
                 [0, 1, 1, 1],
                 [1, 0, 1, 0]]),
       np.array([[0.5,                                 # OUT: desired output for each sample
                  1,
                  0.75,
                  0.5]]).T,
       3)                                              # size of middle layer


