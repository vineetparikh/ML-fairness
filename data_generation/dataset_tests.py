import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import calc_utils

# dataset preprocessed as per https://newonlinecourses.science.psu.edu/stat508/resource/analysis/gcd
file_path = "datasets/german_credit.csv"
group_check = ["Sex & Marital Status"] # groups we think could be discriminated against
y_label = ["Creditability"] # column defining the "y" predictor

def load_data(file_path, y_label):
    """
    Given a file_path and list of feature_names returns
    cleaned dataframe with data from file_path and with
    the features in feature_names in X, and the ground-truth
    of these features in y.
    """
    df = pd.read_csv(file_path)
    # drop non-present rows
    df = df.dropna()
    dfs = dict(tuple(df.groupby(group_col)))
    listdf = [dfs[x] for x in dfs]
    groups = []
    for df in listdf:
        df_y = df[y_label]
        df_x = df[df.columns.difference(y_label)]
        df_trx, df_try = df_x.iloc[0:8*len(X)/10], df_y.iloc[0:8*len(y)/10]
        df_tex, df_tey = df_x.iloc[8*len(X)/10:len(X)], df_y.iloc[8*len(y)/10:len(y)]
        clf = GaussianNB()
        clf.fit(df_trx, df_try)
        df_preds = clf.predict(df_tey)
        groups.append((df_preds,df_tey)) # append test x, predictions, and gt

groups = load_data(file_path,y_label)
print("Predictive value parity: positive")
print(calc_utils.pred_value_parity_pos(groups))
print("Predictive value parity: negative")
print(calc_utils.pred_value_parity_neg(groups))
#train_X, train_y = X.iloc[0:8*len(X)/10], y.iloc[0:8*len(y)/10]
#classifier = LogisticRegression().fit(train_X, train_y)

#test_X, test_y = X.iloc[8*len(X)/10:len(X)], y.iloc[8*len(y)/10:len(y)]
#accuracy = classifier.score(test_X, test_y)
#print(accuracy)
