# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

# %%
'''
Készíts egy függvényt, betölti majd vissza adja az iris adathalmazt.


Egy példa a kimenetre: iris
return type: sklearn.utils.Bunch
függvény neve: load_iris_data
'''

# %%
def load_iris_data() -> sk.utils.Bunch:
    iris = load_iris()
    return iris

#iris = load_iris_data()
#iris

# %%
'''
Készíts egy függvényt, ami a betölti az virágokhoz tartozó levél méretket egy dataframebe, majd az első 5 sort visszaadja.
Minden oszlop tartalmazza, hogy az milyen mérethez tartozik.

Egy példa a bemenetre: iris
Egy példa a kimenetre: iris_df
return type: pandas.core.frame.DataFrame
függvény neve: check_data
'''

# %%
def check_data(iris:sk.utils.Bunch) -> pd.core.frame.DataFrame:
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return iris_df[:5]

#iris_df = check_data(iris)
#iris_df

# %%
''' 
Készíts egy függvényt ami előkészíti az adatokat egy lineaáris regressziós model feltanításához.
Featurejeink legyenek a levél méretek kivéve a "sepal length (cm)", ez legyen a targetünk.

Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
függvény neve: linear_train_data
'''

# %%

def linear_train_data(iris):
    X = iris.data[:, 1:]
    y = iris.data[:, 0]
    return (X, y)

#X_lin, y_lin = linear_train_data(iris)
#X_lin, y_lin

# %%
''' 
Készíts egy függvényt ami előkészíti az adatokat egy logisztikus regressziós model feltanításához.
Featurejeink legyenek a levél méretek, targetünk pedig a 0, 1-es virág osztályok.
Fontos csak azokkal az adatokkal tanítsunk amihez tartozik adott target. 

Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
függvény neve: logistic_train_data
'''

# %%
def logistic_train_data(iris):
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    X = iris_df.loc[np.where(iris.target < 2)].values
    y = iris.target[np.where(iris.target < 2)]
    return (X, y)

#X_log, y_log = logistic_train_data(iris)
#X_log, y_log

# %%
'''
Készíts egy függvényt ami feldarabolja az adatainkat train és test részre. Az adatok 20%-át használjuk fel a teszteléshez.
Tegyük determenisztikussá a darabolást, ennek értéke legyen 42.

Egy példa a bemenetre: X, y
Egy példa a kimenetre: X_train, X_test, y_train, y_test
return type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
függvény neve: split_data
'''

# %%
def split_data(X, y):
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     return X_train, X_test, y_train, y_test

#X_t, X_T, y_t, y_T = split_data(X_lin, y_lin)
#X_t, X_T, y_t, y_T

# %%
'''
Készíts egy függvényt ami feltanít egy lineaáris regressziós modelt, majd visszatér vele.

Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LinearRegression
függvény neve: train_linear_regression
'''

# %%
def train_linear_regression(X_train, y_train) -> sk.linear_model._base.LinearRegression:
    model = LinearRegression().fit(X_train, y_train)
    return model

#lin_reg = train_linear_regression(X_t, y_t)

# %%
'''
Készíts egy függvényt ami feltanít egy logisztikus regressziós modelt, majd visszatér vele.

Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LogisticRegression
függvény neve: train_logistic_regression
'''

# %%
def train_logistic_regression(X_train, y_train) -> sk.linear_model._base.LogisticRegression:
    model = LogisticRegression().fit(X_train, y_train)
    return model

#log_reg = train_logistic_regression(X_t, y_t)

# %%
''' 
Készíts egy függvényt, ami a feltanított modellel predikciót tud végre hajtani.

Egy példa a bemenetre: model, X_test
Egy példa a kimenetre: y_pred
return type: numpy.ndarray
függvény neve: predict
'''

# %%
def predict(model, X_test) -> np.ndarray:
    y_pred = model.predict(X_test)
    return y_pred

#ypred = predict(lin_reg, X_T)
#ypred

# %%
'''
Készíts egy függvényt, ami vizualizálni tudja a label és a predikciók közötti eltérést.
Használj scatter plotot a diagram elkészítéséhez.

Diagram címe legyen: 'Actual vs Predicted Target Values'
Az x tengely címe legyen: 'Actual'
Az y tengely címe legyen: 'Predicted'

Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: scatter plot
return type: matplotlib.figure.Figure
függvény neve: plot_actual_vs_predicted
'''

# %%
def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_title('Actual vs Predicted Target Values')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    return fig

#plot_actual_vs_predicted(y_T, ypred)

# %%
''' 
Készíts egy függvényt, ami a Négyzetes hiba (MSE) értékét számolja ki a predikciók és a valós értékek között.

Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: mse
return type: float
függvény neve: evaluate_model
'''

# %%
def evaluate_model(y_test, y_pred) -> float:
    return mean_squared_error(y_test, y_pred)

#evaluate_model(y_T, ypred)


