import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import calc_utils

# dataset preprocessed as per https://newonlinecourses.science.psu.edu/stat508/resource/analysis/gcd
file_path = "datasets/german_credit.csv"
group_check = "Sex & Marital Status" # groups we think could be discriminated against
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
    groups = df[group_check].unique() # get all unique groups of group_check
    listdf = []
    for g in groups:
        listdf.append(df[df[group_check]==g])
    group_results = []
    for dfg in listdf:
        clf = GaussianNB()

        df_x = dfg[df.columns.difference(y_label)].to_numpy()
        df_y = dfg[y_label].to_numpy()

        # train-test split
        df_trx = df_x[:int(.8*df_x.shape[0])]
        df_try = df_y[:int(.8*df_y.shape[0])].ravel()
        df_tex = df_x[int(8*(df_x.shape[0])/10):]
        df_tey = df_y[int(8*(df_y.shape[0])/10):].ravel()

        # get predictions
        clf.fit(df_trx, df_try)
        df_preds = clf.predict(df_tex)

        group_results.append((df_preds,df_tey)) # append test y and predictions

    return group_results

group_results = load_data(file_path,y_label)
print("Predictive value parity: positive")
print(calc_utils.pred_value_parity_pos(group_results))
print("Predictive value parity: negative")
print(calc_utils.pred_value_parity_neg(group_results))
#train_X, train_y = X.iloc[0:8*len(X)/10], y.iloc[0:8*len(y)/10]
#classifier = LogisticRegression().fit(train_X, train_y)

#test_X, test_y = X.iloc[8*len(X)/10:len(X)], y.iloc[8*len(y)/10:len(y)]
#accuracy = classifier.score(test_X, test_y)
#print(accuracy)
