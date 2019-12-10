import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# dataset preprocessed as per https://newonlinecourses.science.psu.edu/stat508/resource/analysis/gcd
file_path = "datasets/german_credit.csv"
group_check = "Sex & Marital Status" # groups we think could be discriminated against
y_label = ["Creditability"] # column defining the "y" predictor

def P_Yhat_given_Y(preds, y):
    # 0 any negative entries to support two different formats
    preds = 1.0*(preds > 0)
    return np.sum(np.multiply(preds, y)) / np.sum(y)

def load_data(file_path,y_label):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    df_x = df[df.columns.difference(y_label)]
    df_y = df[y_label]

    df_trx = df_x[:int(.8*df_x.shape[0])]
    df_try = df_y[:int(.8*df_y.shape[0])]
    df_tex = df_x[int(8*(df_x.shape[0])/10):]
    df_tey = df_y[int(8*(df_y.shape[0])/10):]

    return (df_x,df_y,df_trx,df_try,df_tex,df_tey)

def train_classifier(tr_x, tr_y):
    tr_x = tr_x.drop(group_check,1) # should be unaware
    tr_x = tr_x.to_numpy()
    tr_y = tr_y.to_numpy().ravel()
    clf = LogisticRegression()
    clf.fit(tr_x,tr_y)
    return clf

def test_classifier(clf,tex):
    tex = tex.drop(group_check,1)
    tex = tex.to_numpy()
    return clf.predict(tex)

df_x,df_y,df_trx, df_try, df_tex, df_tey = load_data(file_path,y_label)
groups = df_tex[group_check].unique()
clf = train_classifier(df_trx,df_try)
yh_g_y_overall = []
yh_g_y_disparity = []
# print(clf.coef_[0]) # first coefficient is consistently one of the higher ones.
plot_x = []
for i in range(20):
    preds = test_classifier(clf,df_tex)
    yh_g_y_overall.append(P_Yhat_given_Y(preds,df_tey.to_numpy().ravel()))
    group_vals = []
    for g in groups:
        dfg_x = df_tex[df_tex[group_check]==g]
        preds_g = test_classifier(clf,dfg_x)
        dfg_y = df_tey[df_tex[group_check]==g].to_numpy().ravel()
        group_vals.append(P_Yhat_given_Y(preds_g,dfg_y))
    yh_g_y_disparity.append(max(group_vals)-min(group_vals))
    clf.coef_[0][0] -= 0.02 #small change for now
    plot_x.append(i*.02)
print(len(yh_g_y_overall))
print(len(yh_g_y_disparity))
plt.plot(plot_x,yh_g_y_overall)
plt.show()
plt.plot(plot_x,yh_g_y_disparity)
plt.show()
