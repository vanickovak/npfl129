#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create the data
    #data
    xs = np.linspace(0, 7, num=args.data_size)
    #target values
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)
    y=ys

    rmses = []
    #X=np.zeros(xs.shape[0],dtype=float)
    for order in range(1, args.range + 1):
        # TODO: Create features of x^1, ..., x^order.
        #generovat matici pro xs(je jeden radek, s rostoucim radkem roste exponent, order je nejvyssi exponent
        # a take pocet radku v matici
        X1 = np.power(xs, order)
        X1 = X1.reshape(-1, 1)
        if order == 1:
            X = X1
        else:
            X= np.concatenate((X, X1), axis=-1)
        # TODO: Split the data into a train set and a test set.
        # Use `sklearn.model_selection.train_test_split` method call, passing
        # arguments `test_size=args.test_size, random_state=args.seed`.
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = args.test_size , random_state=args.seed)
        X_train= X_train.reshape(-1, order)
        X_test= X_test.reshape(-1, order)
        y_train= y_train.reshape(-1, 1)
        y_test= y_test.reshape(-1, 1)
        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`.
        # https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn 
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(X_train,y_train)
        # TODO: Predict targets on the test set using the trained model.
        y_pred= reg.predict(X_test)
        # TODO: Compute root mean square error on the test set predictions
        #na celou predikci
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))

        rmses.append(rmse)

    return rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))
