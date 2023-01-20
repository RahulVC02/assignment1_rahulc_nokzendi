import pandas as pd

#adding the math library for finding logarithm

import math as m


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    n = Y.size
    arr = Y.unique()
    
    probabilities={}

    for num in arr:
        probabilities[num]=0
    
    for occ in Y:
        probabilities[occ]+=1
    
    for val in probabilities.values():
        val = val/n
    
    s=0

    for val in probabilities.values():
        s+=(-1*val*(m.log2(val)))
    
    return s


    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    n = Y.size
    arr = Y.unique()
    
    probabilities={}

    for num in arr:
        probabilities[num]=0
    
    for occ in Y:
        probabilities[occ]+=1
    
    for val in probabilities.values():
        val = val/n
    
    s=0

    for val in probabilities.values():
        s+=(val*val)
    
    return (1-s)

    
    pass


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """

    # df_out = pd.crosstab(attr, Y, normalize = True)
    
    # weights={}
    # n = Y.size()

    # for x in attr.unique():
    #     weights[x]=0
    
    # for vals in attr:
    #     weights[vals]+=1
    
    # for w in weights:
    #     w = w/n

    n = Y.size
    init_entropy=entropy(Y)
  

    unique_attr = attr.unique()
    weights={}

    for attribute_val in unique_arr:
        weights[attribute_val]=0
    
    for i in range(attr.size):
        weights[attr[i]]+=1
    
    for val in weights.values():
        val = val/n
    
    table = pd.crosstab(attr, Y, normalize=True)

    s=0
    for i in range(len(unique_attr)):
        s+=get_entropy(table.loc[unique_attr[i]])*weights[unique_attr[i]]
    
    return init_entropy-s


    pass


def get_entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy of the dataset
    """

    n = Y.size
    s=0
    for i in range(n):
        s+=(-1*Y[i]*(m.log2(Y[i])))
    
    return s
    pass