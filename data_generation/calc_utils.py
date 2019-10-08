import numpy as np
import random
import itertools
import scipy.stats
import matplotlib.pyplot as plt

#Computes P(Y = y| X = x)
def Pr_Y_given_X(y,X,w):
    return 1/(1 + np.exp(-y*(np.matmul(X,w))))



#Returns all possible multinomial outcomes
def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in itertools.combinations_with_replacement(masks, n):
        yield sum(c)

#Computes P(Y = y)
def Pr_Y(y,w,probs,n_trials):
    AllX = np.array(list(partitions(n_trials,4)))
    AllXOffset = np.hstack((AllX, np.ones((AllX.shape[0],1))))
    PrX = scipy.stats.multinomial.pmf(AllX, n = np.full(AllX.shape[0],n_trials), p = probs)
    return np.dot(Pr_Y_given_X(y,AllXOffset,w), PrX)

#Computes P(Yhat given Y)
def Pr_Yhat_given_Y(y_outcome,y_given,w_hat,w_nat,probs,n_trials):
    AllX = np.array(list(partitions(n_trials,4)))
    AllXOffset = np.hstack((AllX, np.ones((AllX.shape[0],1))))
    PrX = scipy.stats.multinomial.pmf(AllX, n = np.full(AllX.shape[0],n_trials), p = probs)
    return np.dot(np.multiply(Pr_Y_given_X(y_outcome,AllXOffset,w_hat),
                              Pr_Y_given_X(y_given,AllXOffset,w_nat)),PrX) / Pr_Y(y_given,w_nat,probs,n_trials)
 
def Pr_Y_given_Yhat(y_outcome,y_given,w_hat,w_nat,probs,n_trials):
    AllX = np.array(list(partitions(n_trials,4)))
    AllXOffset = np.hstack((AllX, np.ones((AllX.shape[0],1))))
    PrX = scipy.stats.multinomial.pmf(AllX, n = np.full(AllX.shape[0],n_trials), p = probs)
    return np.dot(np.multiply(Pr_Y_given_X(y_outcome,AllXOffset,w_nat),
                              Pr_Y_given_X(y_given,AllXOffset,w_hat)),PrX) / Pr_Y(y_given,w_hat,probs,n_trials)

def accuracy(y,w_hat,w_nat,probs,n_trials):
    AllX = np.array(list(partitions(n_trials,4)))
    AllX = np.hstack((AllX, np.ones((AllX.shape[0],1))))
    return Pr_Y(y,w_hat,probs,n_trials) - Pr_Y(y,w_nat,probs,n_trials)

def plot_equal_odd(f, y_outcome, y_given, probs1, probs2, w_nat, ws, x, n_trials,marker, color):
    group1 = np.array([Pr_Yhat_given_Y(y_outcome, y_given, w_hat, w_nat, probs1, n_trials) for w_hat in ws])
    group2 = np.array([Pr_Yhat_given_Y(y_outcome, y_given, w_hat, w_nat, probs2, n_trials) for w_hat in ws])
    f(x, group1 - group2, marker = marker, color = color)
def plot_pred_value_parity(f, y_outcome, y_given, probs1, probs2, w_nat, ws, x, n_trials, marker, color):
    group1 = np.array([Pr_Y_given_Yhat(y_outcome, y_given, w_hat, w_nat, probs1, n_trials) for w_hat in ws])
    group2 = np.array([Pr_Y_given_Yhat(y_outcome, y_given, w_hat, w_nat, probs2, n_trials) for w_hat in ws])
    f(x, group1 - group2, marker = marker, color = color)
def plot_accuracy(f, y, probs, w_nat, ws, x, n_trials, marker, color):
    f(x, [accuracy(y,w_hat,w_nat,probs,n_trials) for w_hat in ws], marker = marker, color = color)
