#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():

    file_path = 'Q2.csv'
    df = pd.read_csv(file_path)

    df.columns = ['X1', 'X2' 'Y']

    X = df[['X1', 'X2']]
    Y = df['Y']

    model = LinearRegression()
    model.fit(X, Y)

    coefficients = model.coef_
    intercept = model.intercept_

    random_index = np.random.randint(0, len(df))
    x_random = X.iloc[random_index].values.reshape(1, -1)
    y_pred = model.predict(x_random)[0]

    pred = model.predict(X)
    MSE = np.sqrt(mean_squared_error(Y, pred))

    print('Equation:')
    print(f'Y = {coefficients[0]:.3f} * X1 + {coefficients[1]:.3f} * X2 + {intercept:.3f}')
    print(f'Predicted value for random index X ({random_index}): {y_pred:.3f}')
    print(f'Root Mean Squared Error: {MSE:.3f}')

if __name__ == '__main__':
    main()
