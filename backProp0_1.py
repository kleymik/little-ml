# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def sigmoid(x):  return 1 / (1+np.exp(-x))
def sigmoidd(x): return sigmoid(x) * (1 - sigmoid(x)) # derivative of sigmoid in terms of sigmoid

inSize  = 4  # input dimensionality
midSize = 4  # m 1 middle/ hidden layer
outSize = 1  # output dimensionality
np.random.seed(1)

IN = np.array([[0, 0, 1, 1],             # 4 samples of input triple
               [1, 1, 1, 1],
               [0, 1, 1, 1],
               [1, 0, 1, 0]])

OUT = np.array([[0.5,                   # desired output for each sample
                   1,
                 0.75,
                 0.5]]).T
syn0 = 2*np.random.random((inSize, midSize)) - 1 # 12 +/- 1 adjustable loadings on links gives 3x4 cross-wire to hidden layer
syn1 = 2*np.random.random((midSize,      1)) - 1 #  4 +/- 1 adjsutable loadings onto one output node

lev0 = IN                               # 4 samples of input triple
for j in range(60000):
    # forward prop the response
    lev1 = sigmoid(lev0 @ syn0)         # l1 = level 1 neuron state
    lev2 = sigmoid(lev1 @ syn1)

    # backprop the error
    outErrs = (OUT - lev2)
    lev2_delta = outErrs             * (lev2 * (1 - lev2))
    syn1 += lev1.T @ lev2_delta

    lev1_delta = lev2_delta @ syn1.T * (lev1 * (1 - lev1))
    syn0 += lev0.T @ lev1_delta
    if j % 100 == 0: print('SQERR=',j,outErrs.T @ outErrs)

    if (outErrs.T @ outErrs) < 1e-4: # sum of squard errors
        print(outErrs.T)
        print()
        break

print('RES=',lev2)
print('SQERR=',outErrs.T @ outErrs)
