import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps

import sklearn
import pandas as pd
import sklearn.metrics
import shap
import xgboost as xgb
import joblib

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

from staty_base import stratified_train_test_split

def add_confusion_matrix_plot(axs, real, pred, name: str, col, accuracy: float):
  ax = axs[col-1]

  cmd = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    y_true=real,
    y_pred=pred,
    display_labels=y['Tag'].drop_duplicates(),
    ax=ax,
    colorbar=False,
  )
  cmd.ax_.tick_params(axis='x', labelrotation=75)
  cmd.ax_.set(
    xlabel=None,
    ylabel=None,
    title=name + f" ({accuracy * 100:.2f} %)".replace(".", ",")
  )

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

fig, axs = plt.subplots(figsize=(2*5, 1*5), ncols=3, nrows=1, sharex=True, sharey=True, layout="constrained")

if True:
  model = joblib.load(MODEL_STORARE_DIR + "dummy_model.pkl")
  y_pred = model.predict(X_test)
  
  add_confusion_matrix_plot(
    axs=axs,
    real=y_test,
    pred=y_pred,
    name="Simulacre",
    col=1,
    accuracy=model.score(X, y_encoded)
  )

if True:
  model = joblib.load(MODEL_STORARE_DIR + "logistic_regression_model.pkl")
  y_pred = model.predict(X_test)

  count_correct = 0
  for i in range(len(y_test)):
    if (y_pred[i] == y_test[i]):
      count_correct += 1
  percent_correct = count_correct / len(y_test)
  
  add_confusion_matrix_plot(
    axs=axs,
    real=y_test,
    pred=y_pred,
    name="Régression logistique",
    col=2,
    accuracy=percent_correct
  )

if True:
  dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)
  
  model = joblib.load(MODEL_STORARE_DIR + "xgboost_model.pkl")
  y_pred = model.predict(dtest_clf)
  
  count_correct = 0
  for i in range(len(y_test)):
    if (y_pred[i] == y_test[i]):
      count_correct += 1
  percent_correct = count_correct / len(y_test)

  add_confusion_matrix_plot(
    axs=axs,
    real=y_test,
    pred=y_pred,
    name="XGBoost",
    col=3,
    accuracy=percent_correct
  )


plt.suptitle("Matrices de confusion")

count_max = y_test[y_test[:] == y_test[0]].shape[0]
cbar_ax = fig.add_axes([0.05, 0.01, 0.02, 0.2])
colorbar = fig.colorbar(
  mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(
      vmin=0,
      vmax=count_max
    ),
    cmap=colormaps.get_cmap("viridis")
  ),
  cax=cbar_ax,
  orientation='vertical'
)
colorbar.set_ticks([0, count_max])

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.25, left=0.15)
fig.supylabel("Réel")
fig.supxlabel("Prédite")
plt.show()