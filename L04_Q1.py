#!/usr/bin/env python3
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import argparse

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

def simple_reg():
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

def multiple_reg(suppress_print=False):

    file_path = 'Q2.csv'
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    model = LinearRegression()
    model.fit(X, Y)

    coefficients = model.coef_
    intercept = model.intercept_

    random_index = np.random.randint(0, len(df))
    x_random = X.iloc[[random_index]]
    y_pred = model.predict(x_random)[0]

    pred = model.predict(X)
    RMSE = np.sqrt(mean_squared_error(Y, pred))
    if not suppress_print:
        print('Equation:')
        print(f'Y = {coefficients[0]:.3f} * X1 + {coefficients[1]:.3f} * X2 + {intercept:.3f}')
        print(f'Predicted value for random index X ({random_index}): {y_pred:.3f}')
        print(f'Root Mean Squared Error: {RMSE:.10f}')

        std_Y = np.std(Y)

        print('\nModel Evaluation:')
        if RMSE < 0.1 * std_Y:
            print(f'Model is highly accurate. Because the RMSE is {RMSE:.10f}')
        elif RMSE < 0.3 * std_Y:
            print(f'Model is moderately accurate. Because the RMSE is {RMSE:.10f}')
        else:
            print(f'Model has low accuracy. Because the RMSE is {RMSE:.10f}')

    #return values for comparison with least squares funciton
    return coefficients, intercept, RMSE

def least_sq():
    file_path = 'Q2.csv'
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X = np.hstack((np.ones((X.shape[0], 1)), X))

    X_transpose = X.T
    w = np.linalg.inv(X_transpose @ X) @ (X_transpose @ Y)

    Y_pred_ls = X @ w
    RMSE_ls = np.sqrt(mean_squared_error(Y, Y_pred_ls))

    print('Least Squares Coefficients:')
    print(f'Intercept(w0): {w[0]:.2f}')
    for i in range(1, len(w)):
        print(f'w{i}: {w[i]:.2f}')

    print('\nEquation from Least Squares:')
    print(f'Y = {w[0]:.2f} + {w[1]:.2f} * X1 + {w[2]:.2f} * X2')
    print(f'Least Squares RMSE: {RMSE_ls:.10f}\n')

    print('Equation from Multiple Regression:')
    part_b_coef, part_b_intc, part_b_rmse = multiple_reg(suppress_print=True)
    print(f'Y = {part_b_intc:.2f} + {part_b_coef[0]:.2f} * X1 + {part_b_coef[1]:.2f} * X2')
    print(f'Multiple Regression RMSE: {part_b_rmse:.10f}\n')

    if RMSE_ls < part_b_rmse:
        print('Least Squares is a more accurate method than Multiple Regression')
    else: 
        print('Multiple Regression is a more accurate method than Least Squares')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiple Regression')
    parser.add_argument('function', choices=['multiple_reg', 'least_sq', 'simple_reg'], help='Function to run')
    args = parser.parse_args()

    if args.function == 'multiple_reg':
        multiple_reg()
    elif args.function == 'least_sq':
        least_sq()
    elif args.function == 'simple_reg':
        simple_reg()