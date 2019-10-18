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

def pred_value_parity_pos(groups):
    ppvs = []
    for (preds,truth) in groups:
        num_true_pos = 0
        num_false_pos = 0
        for i in range(len(preds)):
            if preds[i]==1:
                if truth[i]==1:
                    num_true_pos+=1
                else:
                    num_false_pos+=1
        ppv = (num_true_pos)/(num_true_pos+num_false_pos)
        ppvs.append(ppv)
    # return max difference
    return (max(ppv)-min(ppv))






def pred_value_parity_neg(X, Y, d):
    b = np.apply_along_axis(lambda x : d[str(x)][2], 1, X)
    return Y[b<1].sum() / len(Y[b<1])
