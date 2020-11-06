#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"
    data_bias= np.append(dataset.data, np.ones((506,1)), axis=-1)
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    data_target= np.column_stack((data_bias, dataset.target))
    train_test_split= sklearn.model_selection.train_test_split(data_target, test_size=args.test_size,random_state=args.seed)
    train= train_test_split[0]
    print(train.shape)
    train_target= np.hsplit(train,15)[14]
    print(f"train_target:{train_target.shape}")
    train= train[:,0:14]
    print(f"train:{train.shape}")
    test= train_test_split[1]
    test_target= np.hsplit(test,15)[14]
    test= test[:,0:14]
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    trans_train= np.transpose(train)
    X_t_X=trans_train@train
    inver_X_t_X=np.linalg.inv(X_t_X)

    lr= inver_X_t_X@trans_train@train_target
    # TODO: Predict target values on the test set
    target_values= test@lr

    # TODO: Compute root mean square error on the test set predictions
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, target_values))

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
