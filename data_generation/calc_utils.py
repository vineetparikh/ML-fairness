import numpy as np
import random
import itertools
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

'''
Input for all of these: X as a Pandas dataframe with all input data in
order, y as dataframe with all input results in order, g as index
representing column to use when grouping data, w as np array
representing weight vector (and we can use this to hopefully extend to
all datasets we're looking at).
'''
def stat_parity(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return b.sum() / len(b)
def equal_opp(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return b[Y>0].sum() / len(b[Y>0])

def equal_unopp(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return b[Y<1].sum() / len(b[Y<1])

def equal_acc(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return np.square(b - Y).sum() / len(b)

def pred_value_parity_pos(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return Y[b>0].sum() / len(Y[b>0])

def pred_value_parity_neg(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return Y[b<1].sum() / len(Y[b<1])
