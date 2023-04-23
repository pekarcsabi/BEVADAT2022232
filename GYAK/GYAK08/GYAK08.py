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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegressionSkeleton.LinearRegression()

reg.fit(X_train, y_train)

reg.predict(X_test)

reg.predict(X_test)

reg.evaluate(X_test, y_test)

print('m:', reg.m, 'c:', reg.c)

plt.scatter(X_test, y_test)
plt.plot([min(X_test), max(X_test)], [min(reg.y_pred), max(reg.y_pred)], color='red') # predicted
plt.show()