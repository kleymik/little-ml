# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy.random import random, seed

def sigmoid(x):   return 1 / (1+np.exp(-x))
def sigmoidd(x):  return sigmoid(x) * (1 - sigmoid(x))
def sigmoidds(s): return s * (1 - s)
def prtArr(arr, lbl=None):
    if lbl: print('----',lbl)
    for rw in arr:
        for x in rw: print(f'{x:4.3f}  ', end='')
        print()
        # print(sum(abs(lev2Dlta)))
    print()
    
lev0Num = 3
lev1Num = 3
lev2Num = 2

trainingSet = [1]

IN = np.array([[0, 0,   1],             # 4 samples of input triple
               [1, 1,   1],
               [1, 0.7, 1],               
               [0, 1,   1],
               [1, 0,   1]])
OUT = np.array([[0.5, 1.0, 0.8, 0.8, 0.7],                   # desired output for each sample
                [0.2, 1.0, 0.6, 0.7, 0.5]]).T
seed(1)
syn01 = 2 * random((lev0Num, lev1Num)) - 1 # 12 +/- 1 adjustable loadings on links gives 3x4 cross-wire to hidden layer
syn12 = 2 * random((lev1Num, lev2Num)) - 1 #  4 +/- 1 adjsutable loadings onto one output node

print('------------- Start')
prtArr(IN,    'IN')
prtArr(syn01, 'syn01')
prtArr(syn12, 'syn12')
prtArr(OUT,   'OUT')

for j in range(6000):
    # forward prop the response
    lev1 = sigmoid(IN   @ syn01)  # l1 = level 1 neuron state
    lev2 = sigmoid(lev1 @ syn12)

    # backprop the error
    outErrs = (OUT - lev2)
    lev2Dlta = outErrs * sigmoidds(lev2)
    lev1Dlta = lev2Dlta @ syn12.T * sigmoidds(lev1)

    syn12 += lev1.T @ lev2Dlta
    syn01 += IN.T   @ lev1Dlta
    
    if j % 500 == 0:
        print('SQERR=', j, np.sum(outErrs * outErrs))

    # if (outErrs * outErrs).sum() < 1e-3: # sum of squard errors
    #     print(outErrs.T)
    #    print()
    #    break

print('RES=',   lev2)
print('SMSQERR=', sum(outErrs * outErrs).sum) # sum to scalar
