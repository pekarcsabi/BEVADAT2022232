import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

def load_iris_data() -> sk.utils.Bunch:
    iris = load_iris()
    return iris

def check_data(iris:sk.utils.Bunch) -> pd.core.frame.DataFrame:
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    return iris_df[:5]

def linear_train_data(iris:pd.core.frame.DataFrame) -> (np.ndarray, np.ndarray):
    X = iris.drop(['target', 'sepal length (cm)'], axis=1).values
    y = iris['sepal length (cm)'].values
    return X, y

def logistic_train_data(iris: pd.core.frame.DataFrame) -> (np.ndarray, np.ndarray):
    X = iris.drop(['target'], axis=1).values
    y = iris['target'].values
    return X, y

def split_data(X, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     return X_train, X_test, y_train, y_test