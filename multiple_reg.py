#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():

    file_path = 'Q2.csv'
    df = pd.read_csv(file_path)

    #df.columns = ['X1', 'X2', 'Y']

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

    return coefficients, intercept

if __name__ == '__main__':
    main()
