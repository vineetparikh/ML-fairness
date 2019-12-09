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
    df.loc[:,"applicant_sex_name"] = df.loc[:,"applicant_sex_name"].map({"Male": 1, "Female": -1})
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
    # 0 any negative entries to support two different formats
    preds = 1.0*(preds > 0)
    return np.sum(np.multiply(preds, y)) / np.sum(y)

def P_Y_given_Yhat(preds, y):
    # 0 any negative entries to support two different formats
    preds = 1.0*(preds > 0)
    return np.sum(np.multiply(preds, y)) / np.sum(preds)

def evaluate_model(X, w):
    probs = 1/(1+np.exp(-np.matmul(X,w)))
    preds = np.sign(probs - 0.5)
    return preds

def evaluate_accuracy(y, preds):
    y = 2*y - 1
    correct = sum(y*preds > 0)
    accuracy = float(correct)/len(preds)
    return accuracy

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
X_test_female = test_X[test_X["applicant_sex_name"] == -1]

y_test_male = test_y[test_X["applicant_sex_name"] == 1].to_numpy()
y_test_female = test_y[test_X["applicant_sex_name"] == -1].to_numpy()

# predictions for males and females respectively
preds_male = classifier.predict(X_test_male)
preds_female = classifier.predict(X_test_female)

# statistics for males
print("Probability of Y_hat_given_Y male is: " + str(P_Yhat_given_Y(preds_male, y_test_male)))
print("Probability of Y_given_Y_hat male is: " + str(P_Y_given_Yhat(preds_male, y_test_male)))

# statistics for females
print("Probability of Y_hat_given_Y female is: " + str(P_Yhat_given_Y(preds_female, y_test_female)))
print("Probability of Y_given_Y_hat female is: " + str(P_Y_given_Yhat(preds_female, y_test_female)))


opportunity = P_Yhat_given_Y(preds_male, y_test_male) - P_Yhat_given_Y(preds_female, y_test_female)
predictive_value = P_Y_given_Yhat(preds_male, y_test_male) - P_Y_given_Yhat(preds_female, y_test_female)

print("\n\n")
print("difference in opportunity between male and female is: " + str(opportunity))
print("difference in predictive value between male and female is: " + str(predictive_value))
# probablity that the lone is accepted/denied
# total_entries = X.shape[0]
# prob_accepted = df.sum(axis=0, skipna=True).loc["action_taken_name"]/total_entries
# print("Probability of lone acceptance overall is: " + str(prob_accepted))
# prob_denied = 1 - prob_accepted
# print("Probability of lone denial overall is: " + str(prob_denied))

# test the classification on the weight vector
w = classifier.coef_[0]
test_X = test_X.to_numpy()
test_y = test_y.to_numpy()

# convert negative lable to -1
opportunities = [opportunity]
predictive_values = [predictive_value]
ws = [w[6]]
accuracies = [accuracy]
steps = [0]
for i in range(10):
    steps.append(i + 1)
    print("\n\n\n")
    w[6] -= 0.01
    ws.append(w[6])
    print(w[6])
    preds = evaluate_model(test_X, w)
    preds_male = evaluate_model(X_test_male, w)
    preds_female = evaluate_model(X_test_female, w)
    accuracy = evaluate_accuracy(test_y, preds)
    accuracies.append(accuracy)
    print("accuracy for the following round was: " + str(accuracy))
    print("P_Yhat_given_Y for male is : " + str(P_Yhat_given_Y(preds_male, y_test_male)))
    print("P_Y_given_Yhat for male is : " + str(P_Y_given_Yhat(preds_male, y_test_male)))
    print("P_Yhat_given_Y for female is : " + str(P_Yhat_given_Y(preds_female, y_test_female)))
    print("P_Y_given_Yhat for female is : " + str(P_Y_given_Yhat(preds_female, y_test_female)))
    print("\n")
    opportunity = P_Yhat_given_Y(preds_male, y_test_male) - P_Yhat_given_Y(preds_female, y_test_female)
    opportunities.append(opportunity)
    predictive_value = P_Y_given_Yhat(preds_male, y_test_male) - P_Y_given_Yhat(preds_female, y_test_female)
    predictive_values.append(predictive_value)
    print("difference in opportunity between male and female is: " + str(opportunity))
    print("difference in predictive value between male and female is: " + str(predictive_value))

fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(steps, opportunities, label='difference in opportunites')
ax1.plot(steps, predictive_values, label='difference in predictive values')

fig, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.plot(steps, accuracies, label='accuracies')

#plt.plot(ws, opportunities, label='difference in opportunites')
#plt.plot(ws, predictive_values, label='difference in predictive values')
#plt.plot(ws, accuracies, label='accuracy')
ax1.legend()
ax2.legend()
plt.show()







