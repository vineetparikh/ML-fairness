import numpy as np
import pandas as pd
from method_test import *

# dataset preprocessed as per https://newonlinecourses.science.psu.edu/stat508/resource/analysis/gcd
df = pd.read_csv("datasets/german_credit.csv")
df.apply(lambda x: np.around(x,decimals=-1) if x.name in ["Age (years)","Duration in Current address","Duration of Credit (month)","Credit Amount","Concurrent Credits"] else x)
print(df.columns)
# now i just need to apply the optimal classifier
def ds_apply(n,f,balance):
  avg = 0
  for _ in range(n):
    X = df.as_matrix(df.drop(columns=['Creditability']))
    Y = df.as_matrix(columns=["Creditability"])
    Y = Y.reshape(1000, 1)
    D = np.hstack((X, Y))
    d = bayes_opt(D, balance)
    avg += f(X, Y.flatten(), d)
  return avg/n

balance = True
print("Statistical parity: ", ds_apply(10, stat_parity, balance), "vs ", ds_apply(10, stat_parity, False))
print("Equal opportunity: ", ds_apply(10, equal_opp, balance), "vs", ds_apply(10, equal_opp, False))
print("Equal unopportunity: ", ds_apply(10, equal_unopp, balance), "vs", ds_apply(10, equal_unopp, False))
print("Equal accuracy: ", ds_apply(10, equal_acc, balance), "vs", ds_apply(10, equal_acc, False))
print("Predictive value parity positive: ", ds_apply(10, pred_value_parity_pos, balance), "vs", ds_apply(10, pred_value_parity_pos, False))
print("Predictive value parity negative: ", ds_apply(10, pred_value_parity_neg, balance), "vs", ds_apply(10, pred_value_parity_neg, False))
