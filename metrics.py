from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    
    n = y.size
    a=0
    for i in range(n):
        if(y[i]==y_hat[i]):
            a = a+1
    
    return float(a/n)

    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    a= 0
    b=0

    for i in range(y.size):
        if(y_hat[i]==cls):
            b=b+1

            if(y[i]==cls):
                a=a+1
    
    return (a/b)
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    a=0
    b=0

    for i in range(y.size):
        if(y[i]==cls):
            b=b+1

            if(y_hat[i]==cls):
                a=a+1
    
    return (a/b)
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    s=0
    
    for i in range(y.size):
        s=s+(y[i]-y_hat[i])**2
    
    return (s/y.size)**0.5

    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    s=0

    for i in range(y.size):
        s=s+abs(y[i]-y_hat[i])
    return s/y.size
    pass
