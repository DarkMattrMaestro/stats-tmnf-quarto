
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/#:~:text=XGBoost%20provides%20a%20wrapper%20class%20to%20allow%20models,The%20XGBoost%20model%20for%20classification%20is%20called%20XGBClassifier.

import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

###################################

FILE_NAME = "./collected-data/flat-replay-data-5rep.csv"

# # load data
# dataset = loadtxt(FILE_NAME, delimiter=",")

# full_dataset = pd.read_csv(FILE_NAME) # For Pandas
full_dataset = np.genfromtxt(FILE_NAME, delimiter=',', skip_header=1)

print(full_dataset)

# feature_types = ["q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q"]
feature_names = pd.read_csv(FILE_NAME, nrows=1).drop("Tag", axis=1).columns.tolist() # For Pandas

mapping = {
  "Normal": 0,
  # "Stunt": 1,
  # "Maze": 2,
  "Offroad": 1,
  # "Laps": 4,
  "Fullspeed": 2,
  "LOL": 3,
  "Tech": 4,
  "SpeedTech": 5,
  # "RPG": 9,
  "PressForward": 6,
  # "Trial": 11,
  "Grass": 7,
}

mappingNumToCat = {
  0: "Normal",
  # "Stunt": 1,
  # "Maze": 2,
  3: "Offroad",
  # "Laps": 4,
  5: "Fullspeed",
  6: "LOL",
  7: "Tech",
  8: "SpeedTech",
  # "RPG": 9,
  10: "PressForward",
  # "Trial": 11,
  12: "Grass",
}

# for removed in [None, "AvgAbsDisplacementHorizontal","AvgAbsDisplacementY","AvgRPM","AvgSteerBias","AvgAbsSteer","AvgSpeedForward","AvgAbsSpeedForward","AvgSpeedSidewardBias","AvgAbsSpeedSideward","AvgSpeedSidewardOppSteer","PercentPitchLowerThird","PercentRollLowerThird","PercentPitchMiddleThird","PercentRollMiddleThird","PercentPitchUpperThird","PercentRollUpperThird","PercentTurbo"]:
  # print(removed)
  
dataset = full_dataset.copy()

  # dataset['Tag'] = dataset['Tag'].map(mappingNumToCat).map(mapping)
  # if removed is not None: dataset = dataset.drop(removed, axis=1)
  # print(dataset.head)

# split data into X and y
# X, y = dataset.drop("Tag", axis=1), dataset[['Tag']] # For Pandas
X = dataset[:,1:]
y = dataset[:,0]

# split data into train and test sets
# seed = 3141
# test_size = 0.15

num_round = 500
param = {
  "objective": "multi:softprob",
  "tree_method": "hist",
  "num_class": 7,
  "device": "cuda",
}

# Set up training and testing datasets
mask = np.random.rand(len(X)) < 0.8
dtrain = xgb.DMatrix(X[mask,:], label=y[mask], feature_names=feature_names)
dtest = xgb.DMatrix(X[~mask,:], label=y[~mask], feature_names=feature_names)

# Set up evaluation
watchlist = [(dtest, "eval"), (dtrain, "train")]
evals_result = {}

print("")
print("Access metrics through a loop:")
for e_name, e_mtrs in evals_result.items():
    print("- {}".format(e_name))
    for e_mtr_name, e_mtr_vals in e_mtrs.items():
        print("   - {}".format(e_mtr_name))
        print("      - {}".format(e_mtr_vals))

# GPU accelerated training
model = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result)

# run prediction
preds = model.predict(dtest)
labels = dtest.get_label()
print(
    "error=%f"
    % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))
    )
)

# Print results
# print(evals_result["eval"])

# Compute shap values using GPU with xgboost
model.set_param({"device": "cuda"})
shap_values = model.predict(dtrain, pred_contribs=True)

# Compute shap interaction values using GPU
shap_interaction_values = model.predict(dtrain, pred_interactions=True)

# shap will call the GPU accelerated version as long as the device parameter is set to "cuda"
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# # visualize the first prediction's explanation
# shap.force_plot(
#     explainer.expected_value,
#     shap_values[0, :],
#     X[0, :],
#     feature_names=feature_names,
#     matplotlib=True,
# )

# Show a summary of feature importance
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)

##########################

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# min_max_scaler = X_train.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)

# # fit model no training data
# model = xgb.XGBClassifier()
# model.fit(X_train, y_train)

# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]

# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
  
  ################################
  
# X = dataset.drop('Tag', axis = 1)
# Y = dataset['Tag']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify = Y, random_state=2022)
# logistic_classifier = LogisticRegression(solver='newton-cholesky', max_iter=1000000)
# logistic_classifier.fit(X_train, Y_train)
# Y_pred = logistic_classifier.predict(X_test)
# print(confusion_matrix(Y_test,Y_pred))
# print(classification_report(Y_test,Y_pred))