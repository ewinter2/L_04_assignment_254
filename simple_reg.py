#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(x, y, degree, x_pred):
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    #generate points 
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_range_poly = poly.fit_transform(X_range)
    Y_pred = model.predict(X_range_poly)

    #plot the data
    plt.scatter(X, y, color='blue', label='data points')
    plt.plot(X_range, Y_pred, color='pink', label='polynomial fit (degree {degree})')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.show()

    #predict the value of x
    x_pred_poly = poly.fit_transform(np.array([[x_pred]]))
    y_pred_value = model.predict(x_pred_poly)

    #Display equation coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    print(f'Polynomial Eq: y = {intercept} ', end='')
    for i, coef in enumerate(coefficients[1:], 1):
        print(f'+ {coef}x^{i} ', end='')
    print()

    print(f'Predicted value of x={x_pred}: {y_pred_value[0]}')


