# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def sigmoid(x): return 1 / (1+np.exp(-x))

def go():
    IN = np.array([[0, 0, 1],               # 4 samples of input triple
                   [1, 1, 1],
                   [0, 1, 1],
                   [1, 0, 1] ])
    OUT = np.array([[0.5,                   # desired output for each sample
                       1,
                     0.8,
                     0.7]]).T

    syn0 = 2*np.random.random((3,4)) - 1    # 12 +/- 1 adjustable loadings on links gives 3x4 cross-wire to hidden layer
    syn1 = 2*np.random.random((4,1)) - 1    #  4 +/- 1 adjsutable loadings onto one output node

    lev0 = IN                               # 4 samples of input triple



    for j in range(int(1e5)):

        # forward prop the response
        lev1 = 1 / (1+np.exp(-(lev0 @ syn0)))  # l1 = level 1 neuron state
        lev2 = 1 / (1+np.exp(-(lev1 @ syn1)))

        # backprop the error
        outErrs = (OUT - lev2)
        lev2_delta = outErrs             * (lev2 * (1 - lev2))
        syn1 += lev1.T @ lev2_delta

        lev1_delta = lev2_delta @ syn1.T * (lev1 * (1 - lev1))
        syn0 += lev0.T @ lev1_delta
        if j % 1000 == 0: print('SQERR=',j, (outErrs * outErrs).sum())
        if (outErrs * outErrs).sum() < 1e-8: # sum of squard errors
            print(outErrs.T)
            print()
            break

    print('RES=', lev2)
    print('SQERR=', outErrs.T @ outErrs)
