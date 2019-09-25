import coin_flip as cf
import numpy as np
import random

#Returns a dictionary that takes in an x in string form and outputs a tuple where
#the third entry is the prediction. If balance = true, then it will reduce the
#number of positive classifications given
def bayes_opt(X, balance):
  feature_count = {}
  for r in X:
    x = str(r[0:len(r)-1])
    y = r[len(r)-1]
    if x in feature_count:
      (p,n,d) = feature_count[x]
      if y > 0:
        p += 1
      else:
        n += 1
      if p > n:
        d = np.random.binomial(1, n/(n+p)) if balance else 1
      elif p < n:
        d = 0
      else:
        d = random.randint(0,1)
      feature_count[x] = (p,n,d)
    else:
      if y > 0:
        feature_count[x] = (1,0,1)
      else:
        feature_count[x] = (0,1,1)
  return feature_count

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

def average(p,n,f,balance):
  avg = 0
  for _ in range(n):
    cfg = cf.CoinFlipGenerator(p)
    X = cfg.get_features()
    Y = cfg.get_labels(X)
    Y = Y.reshape(1000, 1)
    D = np.hstack((X, Y))
    d = bayes_opt(D, balance)
    avg += f(X, cfg.get_labels(X).flatten(), d)
  return avg/n
'''
p1 = [0.5,0.5,0.5,0.5,0.5]
p2 = [0.0,0.5,0.1,0.5,0.5]
balance = True
print("Statistical parity: ", average(p1, 100, stat_parity, balance), "vs ", average(p2, 100, stat_parity, False))
print("Equal opportunity: ", average(p1, 100, equal_opp, balance), "vs", average(p2, 100, equal_opp, False))
print("Equal unopportunity: ", average(p1, 100, equal_unopp, balance), "vs", average(p2, 100, equal_unopp, False))
print("Equal accuracy: ", average(p1, 100, equal_acc, balance), "vs", average(p2, 100, equal_acc, False))
print("Predictive value parity positive: ", average(p1, 100, pred_value_parity_pos, balance), "vs", average(p2, 100, pred_value_parity_pos, False))
print("Predictive value parity negative: ", average(p1, 100, pred_value_parity_neg, balance), "vs", average(p2, 100, pred_value_parity_neg, False))
'''
