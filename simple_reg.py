#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(x, Y, degree, x_pred):
    X = np.array(x).reshape(-1, 1)
    Y = np.array(Y)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, Y)

    #generate points 
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_range_poly = poly.fit_transform(X_range)
    Y_pred = model.predict(X_range_poly)

    #plot the data
    plt.scatter(X, Y, color='blue', label='data points')
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

    print(f'Polynomial Eq: y = {intercept:.3f} ', end='')
    for i, coef in enumerate(coefficients[1:], 1):
        print(f'+ {coef:.3f}x^{i} ', end='')
    print()

    print(f'Predicted value of x={x_pred}: {y_pred_value[0]:.3f}')

    print("Computing Error Metrics:")
    Y_fit = model.predict(X_poly)
    SE = np.sum((Y - Y_fit) ** 2)
    MSE = mean_squared_error(Y, Y_fit)
    RMSE = np.sqrt(MSE)

    print(f'Sum of Squared Errors: {SE:.3f}')
    print(f'Mean Squared Error: {MSE:.3f}')
    print(f'Root Mean Squared Error: {RMSE:.3f}')
    print('-' * 50)

def main():
    #First Given Dataset 
    X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y1 = [2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8]
    print('-' * 50)
    print("Dataset 1:")
    polynomial_regression(X1, Y1, degree=2, x_pred=100)

    #Second Given Dataset
    X2 = [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
    Y2 = [17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4]
    print('-' * 50)
    print("Dataset 2:")
    polynomial_regression(X2, Y2, degree=2, x_pred=0.5)

if __name__ == '__main__':
    main()
