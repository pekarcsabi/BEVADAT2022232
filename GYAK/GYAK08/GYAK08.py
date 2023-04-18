import LinearRegressionSkeleton
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = df['petal width (cm)'].values
y = df['sepal length (cm)'].values

lrg = LinearRegressionSkeleton.LinearRegression()
lrg.fit(X, y)
lrg.predict(X)
lrg.evaluate(X, y)
'''
y_pred = lrg.predict(X)

plt.scatter(lrg.X_test, lrg.y_test)
plt.plot([min(lrg.X_test), max(lrg.X_test)], [min(lrg.y_pred), max(lrg.y_pred)], color='red') # predicted
plt.show()
'''