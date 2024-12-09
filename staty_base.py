
import math
import random
import numpy as np

def stratified_train_test_split(y_encoded, X, test_size=0.25, seed=3142, ):
  """_summary_

  Args:
      y_encoded (_type_): Class
      X (_type_): Data
      test_size (float, optional): Test size, as a percentage from 0 to 1. Defaults to 0.25.
      seed (int, optional): _description_. Defaults to 3142.

  Returns:
      (tuple[NDArray, NDArray, NDArray, NDArray]): X_train, y_train, X_test, y_test
  """
  X_train_list = []
  y_train_list = []
  X_test_list = []
  y_test_list = []

  for stratyfier in np.unique(y_encoded):
    joined = np.concatenate((y_encoded, X), axis=1)
    stratum_joined = joined[y_encoded[:,0] == stratyfier]
    stratum_X = stratum_joined[:,1:]
    stratum_y = stratum_joined[:,0]
    
    test_count: int = math.floor(test_size * stratum_joined.shape[0])
    
    indices = np.arange(stratum_y.shape[0])
    random.Random(seed).shuffle(indices)
    
    X_test_list.extend(stratum_X[indices[:test_count], :].tolist())
    X_train_list.extend(stratum_X[indices[test_count:], :].tolist())
    
    y_test_list.extend(stratum_y[indices[:test_count]].tolist())
    y_train_list.extend(stratum_y[indices[test_count:]].tolist())

  X_train = np.array(X_train_list)
  y_train = np.array(y_train_list)
  X_test = np.array(X_test_list)
  y_test = np.array(y_test_list)
  
  return X_train, y_train, X_test, y_test