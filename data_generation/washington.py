import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

file_path = "datasets/Washington_State_HDMA-2016.csv"

# the features we want to use to make the predictor, with the last one being the label
feature_names = ["tract_to_msamd_income", "minority_population","number_of_1_to_4_family_units", "applicant_income_000s", "loan_amount_000s", "hud_median_family_income", "applicant_sex_name", "hud_median_family_income", "number_of_owner_occupied_units", "owner_occupancy_name", "loan_purpose_name", "county_name", "agency_name", "lien_status_name", "action_taken_name"]
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
    # PROCESS 1 HOT FEATURES
    #df = pd.get_dummies(df, prefix=[])
    df = pd.get_dummies(df, prefix=["owner_occupancy_name", "loan_purpose_name", "county_name", "agency_name", "lien_status_name"])
    print([col for col in df.columns])
    # END PROCESS 1 HOT FEATURES
    # drop non-present rows
    df = df.dropna()
    return df

def get_X_Y(df):
    df = df.sort_values(by=['action_taken_name'])
    num_denied = len(df[df["action_taken_name"] == 0])
    # balance the training data
    df = df.iloc[0:2*num_denied]
    # shuffle the rows around randomly to prevent uneven testing split
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.pop("action_taken_name")
    df["action_taken_name"] = y
    X = df.iloc[:,:-1]
    return X,y,df

def P_Yhat_given_Y(preds, y):
    return np.sum(np.multiply(preds, y)) / np.sum(y)

def P_Y_given_Yhat(preds, y):
    return np.sum(np.multiply(preds, y)) / np.sum(preds)

df = load_data(file_path, feature_names)
X,y,df = get_X_Y(df)
train_X, train_y = X.iloc[0:8*len(X)/10], y.iloc[0:8*len(y)/10]
classifier = LogisticRegression().fit(train_X, train_y)

test_X, test_y = X.iloc[8*len(X)/10:len(X)], y.iloc[8*len(y)/10:len(y)]
accuracy = classifier.score(test_X, test_y)
print("accuracy is: " + str(accuracy))

# Get the actual predictions on the test set
preds = classifier.predict(test_X)
print("Probability of Y_hat_given_Y overall is: " + str(P_Yhat_given_Y(preds, test_y)))
print("Probability of Y_given_Y_hat overall is: " + str(P_Y_given_Yhat(preds, test_y)))

# Split test set by male and female
X_test_male = test_X[test_X["applicant_sex_name"] == 1]
X_test_female = test_X[test_X["applicant_sex_name"] == 0]

y_test_male = test_y[test_X["applicant_sex_name"] == 1]
y_test_female = test_y[test_X["applicant_sex_name"] == 0]

# predictions for males and females respectively
preds_male = classifier.predict(X_test_male)
preds_female = classifier.predict(X_test_female)

# statistics for males
print("Probability of Y_hat_given_Y male is: " + str(P_Yhat_given_Y(preds_male, y_test_male)))
print("Probability of Y_given_Y_hat male is: " + str(P_Y_given_Yhat(preds_male, y_test_male)))

# statistics for females
print("Probability of Y_hat_given_Y female is: " + str(P_Yhat_given_Y(preds_female, y_test_female)))
print("Probability of Y_given_Y_hat female is: " + str(P_Y_given_Yhat(preds_female, y_test_female)))

# probablity that the lone is accepted/denied
total_entries = X.shape[0]
prob_accepted = df.sum(axis=0, skipna=True).loc["action_taken_name"]/total_entries
print("Probability of lone acceptance overall is: " + str(prob_accepted))
prob_denied = 1 - prob_accepted
print("Probability of lone denial overall is: " + str(prob_denied))





