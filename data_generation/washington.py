import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
file_path = "datasets/Washington_State_HDMA-2016.csv"

# the features we want to use to make the predictor, with the last one being the label
feature_names = ["tract_to_msamd_income", "applicant_income_000s", "loan_amount_000s", "applicant_sex_name", "action_taken_name"]

def load_data(file_path, feature_names):
    """
    Given a file_path and list of feature_names returns
    cleaned dateframe with data from file_path and with
    the features in feature_names
    """
    df = pd.read_csv(file_path)
    df = df.loc[:, feature_names]
    df.iloc[:,-1] = df.iloc[:,-1].map({"Loan originated" : 1, "Application denied by financial institution": 0})
    df.loc[:,"applicant_sex_name"] = df.loc[:,"applicant_sex_name"].map({"Male": 1, "Female": 0})
    # drop non-present rows
    df = df.dropna()
    # shuffle the rows around randomly to prevent uneven testing split
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X,y

X,y = load_data(file_path, feature_names)
train_X, train_y = X.iloc[0:8*len(X)/10], y.iloc[0:8*len(y)/10]
classifier = LogisticRegression().fit(train_X, train_y)

test_X, test_y = X.iloc[8*len(X)/10:len(X)], y.iloc[8*len(y)/10:len(y)]
accuracy = classifier.score(test_X, test_y)
print(accuracy)


