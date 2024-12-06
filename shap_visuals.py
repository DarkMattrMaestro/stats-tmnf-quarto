import xgboost as xgb
import shap
import pandas as pd
import joblib

from sklearn.preprocessing import OrdinalEncoder


from staty_base import stratified_train_test_split

FILE_NAME = "./collected-data/flat-replay-data-5rep.csv"
MODEL_STORARE_DIR = "./models/"
SEED = 3142

dataset = pd.read_csv(FILE_NAME) # For Pandas
dataset = dataset.astype({"Tag": str})
mappingNumToCat = {
  "0": "Normal",
  # "Stunt": 1,
  # "Maze": 2,
  "3": "Offroad",
  # "Laps": 4,
  "5": "Fullspeed",
  "6": "LOL",
  "7": "Tech",
  "8": "SpeedTech",
  # "RPG": 9,
  "10": "PressForward",
  # "Trial": 11,
  "12": "Grass",
}
dataset['Tag'] = dataset['Tag'].replace(mappingNumToCat) # Map to categorical
# split data into X and y
X, y = dataset.drop("Tag", axis=1), dataset[['Tag']] # For Pandas
# Encode y to numeric
y_encoded = OrdinalEncoder().fit_transform(y)
# Split the data
X_train, y_train, X_test, y_test = stratified_train_test_split(y_encoded, X)

dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
xgboost_model = joblib.load(MODEL_STORARE_DIR + "xgboost_model.pkl")

# shap will call the GPU accelerated version as long as the device parameter is set to "cuda"
explainer = shap.Explainer(xgboost_model)
shap_values = explainer.shap_values(X_train)

print(shap_values)

# visualize the first prediction's explanation
shap.summary_plot(shap_values, feature_names=list(pd.DataFrame(X_test).columns.values), plot_type='bar')