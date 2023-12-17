import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.coef_, model.intercept_)
    print(model.score(X_test, y_test))
    print(model.predict(X_test[:10]))