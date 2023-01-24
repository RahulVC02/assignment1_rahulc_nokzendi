# METRICS
from typing import Union
import pandas as pd
import math


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here

    # initializing true_positive to 0 and total to the size of y_hat
    true_positive = 0
    total = y_hat.size

    # case when total = 0, to avoid division by zero
    if total == 0:
        return 1
    
    # Loop to add all the cases where y_hat and y are equal i.e true positives
    for i in range(y.size):
        if y[i] == y_hat[i]:
            true_positive += 1
    
    # Calculating accuracy
    accuracy = float(true_positive) / total
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert(y_hat.size == y.size)

    # initializing true_positive and positive_prediction to 0
    true_positive = 0
    positive_prediction = 0

    # Loop to add all the cases we get true positives and positive prediction
    for i in range(y.size):
        if y[i] == cls and y_hat[i] == cls:
            true_positive += 1
        if y_hat[i] == cls:
            positive_prediction += 1

    # case when positive_prediction = 0, to avoid division by zero
    if positive_prediction == 0:
        return 1
    
    # Calculating precision
    precision = float(true_positive) / positive_prediction
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert(y_hat.size == y.size)

    # initializing true_positive and actual_positive to 0
    positives = 0
    true_positive = 0

    # Loop to add all the cases we get true positives and actual positives
    for i in range(y.size):
        if y[i] == cls and y_hat[i] == cls:
            true_positive += 1
        if y[i] == cls:
            positives += 1

    # case when positives = 0, to avoid division by zero
    if positives == 0:
        return 1

    recall = float(true_positive) / positives
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert(y_hat.size == y.size)

    # initializing sq_error to 0
    sq_error_sum = 0

    # Loop to add the squared error
    for i in range(y.size):
        sq_error_sum += (y[i] - y_hat[i]) ** 2

    # Calculating mean of the sum of squared error
    mean_sq_error = sq_error_sum / y.size

    # Calculating root of the mean of the sum of squared error
    rmse = math.sqrt(mean_sq_error)
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert(y_hat.size == y.size)

    # initializing absolute_error_sum to 0
    absolute_error_sum = 0

    # Loop to add the absolute error
    for i in range(y.size):
        absolute_error_sum = abs(y[i] - y_hat[i])

    # Calculating mean of the sum of absolute error
    mae = float(absolute_error_sum) / y.size
    return mae
