# BASE.py
#c

"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index
np.random.seed(42)


class Node():
    '''
    Nodes for tree, Decision tree stores all the val in form of multiple
    nodes of this class.
    '''

    def __init__(self, split_column = None, val = None, depth = None):
        self.value = val # value of the TreeNode
        self.split_column = split_column # col that has to be taken into check
        self.split_value = None # when attribute/input is real, we get value to split on. Else None
        self.children = {} # to store child TreeNodes
        self.freq = None # for attr not there
        self.depth = depth # curr_depth


    def node_val(self, X, max_depth = np.inf):
        '''
        function to get the value of the node at maximum depth, i.e leaf node where the probability is the maximum
        
        '''
        ## Leaf node
        if (self.depth >= max_depth or self.split_column == None): 
            return self.value # We return the value of the leaf node
        ## Non-leaf node
        else:
            ## Classification type
            if(self.split_value == None): 
                curr_split = self.split_column # curr_split is the column name that our node is split on
                ## Case when we encounter a class that was present in the training dataset
                if(X[curr_split] in self.children): # We go to the child corresponding to the class 
                    return self.children[X[curr_split]].node_val(X.drop(curr_split), max_depth = max_depth)
                ## Case when we encounter a class that was not present in the training dataset
                else:
                    # We return the value of the node
                    # However, we will not encounter this case in our implementation as our testing dataset is the 
                    # same as the training dataset
                    return self.value 
            ## Regression type
            else: 
                curr_split = self.split_column # curr_split is the column name that our node is split on
                if(X[curr_split] > self.split_value): # checking if the value of the column is greater than the split value of the node
                    return self.children["right"].node_val(X, max_depth = max_depth) # if yes, we go to the right child
                else:
                    return self.children["left"].node_val(X, max_depth = max_depth) # else we go to the left child


    ## This function is used to print the tree in a readable format
    # NOTE: The end = "" method is used to print the next print statement on the same line
    # NOTE: The placement of the end = "" method on the multiple print statements have been done carefully through trial and error
    def print_tree(self, space = 1):
        # Base case, when we reach a leaf node and hence no column to further split on
        if(self.split_column == None):
            # We print the value of the node
            print(" Value: {:.2f}, Depth: {}".format(self.value, self.depth), end = "")
            return
        # When we have a non-leaf node
        else:
            # When the input is real, we have a split_value to split on
            if(self.split_value != None):
                ## This part just handles the first case when we have to print the first condition without any Y or N as the prefix
                if(space == 1):
                    print("?(X{} <= {:.2f})".format(self.split_column, self.value), end = "")
                else:
                    print(" ?(X{} <= {:.2f})".format(self.split_column, self.value), end = "")
                
                ## We loop through the left and right children and print the corresponding prefix Y or N
                for i in self.children:
                    if(i == "left"):
                        print("\n" + "|  " * space + f"Y:", end = "")
                    else:
                        print("\n" + "|  " * space + f"N:", end = "")
                    self.children[i].print_tree(space = space + 1) # Recurisve call
            # When the input is discrete, we have to split on each value in self.children
            else:
                for i in self.children:
                    print("\n" + "|  " * (space - 1) + f"?(X{self.split_column} = {i})", end = "")
                    self.children[i].print_tree(space = space + 1) # Recursive call


class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=10):
        self.criterion = criterion # criterion won't be used for regression
        self.max_depth = max_depth # The maximum depth the tree can grow to
        self.root = None # root node of the TreeNode DecisionTree
        self.attribute = None # column names in X
        self.output_type = None # data type of Y to identify whether a classification or regression problem
        self.X_size = None # size (length) of X

    def build_tree(self, X, Y, parent, depth=0):
        # the output column has only one class left, then we return a node with its value as the class
        if(Y.unique().size == 1): 
            return Node(val = Y.values[0], depth=depth)

        # when we need to split according to some attribute non-trivially
        if(len(X.columns) > 0 and depth < self.max_depth and len(list(X.columns)) != sum(list(X.nunique()))):
            split_col = None # column name to split on
            max_infoGain = -np.inf # max information gain
            real_split_value = None # real number value to split our real valued attribute(if it exists)

            # Iterating over all the columns to find the best column to split on
            for attr in list(X.columns): 
                curr_real_split_value = None
                curr_info_gain = 0
                # information_gain() returns two values if input is real, else it returns None for the second value for discrete inputs
                curr_iGain, curr_split = information_gain(Y, X[attr], self.criterion) 
                if(curr_split is not None): # When the input is real
                    curr_real_split_value = curr_split # we store the split value
                    curr_info_gain = curr_iGain # we store the information gain
                    if(curr_info_gain > max_infoGain): # we check if the current information gain is greater than the max information gain
                        # We set the values initialized above to the current values
                        max_infoGain = curr_info_gain
                        split_col = attr
                        real_split_value = curr_real_split_value
                else: # When the input is discrete
                    curr_info_gain = curr_iGain # we store the information gain
                    # NOTE: We don't need to check for the split value as it is None for discrete inputs
                    if(curr_info_gain > max_infoGain):
                        # We set the values initialized above to the current values
                        max_infoGain = curr_info_gain
                        split_col = attr

            # split_col stores the best column to split on to get the max information gain
            curr_node = Node(split_column = split_col) # creating a node with the best column to split on
            root_col = X[split_col]

            ## DISCRETE INPUT
            if(root_col.dtype.name == "category"):
                X = X.drop(split_col, axis=1) # removing the column from the attribute table
                # creating a pandas series with index as the unique values of root_col and values as the count of occurence of those values 
                root_classes = root_col.groupby(root_col).count() 
                
                # iterating over all the classes in the root column
                for class_type in list(root_classes.index):
                    curr_rows = (root_col == class_type) # array of rows where the class is equal to the current class
                    if(curr_rows.sum() > 0): # creating a branch for each class if at least one row has that class
                        # X[curr_rows] is the subset of X where the class is equal to curr_rows, similarly for Y
                        # We use these subsets to build the tree recursively
                        curr_node.children[class_type] = self.build_tree(X[curr_rows], Y[curr_rows], curr_node, depth=depth+1)
                        curr_node.children[class_type].freq = len(X[curr_rows])/self.X_size

            ## REAL INPUT
            else:
                # splitting the root column in left and right branches
                left_split = (root_col <= real_split_value)
                right_split = (root_col > real_split_value)

                # Creating a left and right child as the inputs are real
                curr_node.children["left"] = self.build_tree(X[left_split], Y[left_split], curr_node, depth=depth+1)
                curr_node.children["right"] = self.build_tree(X[right_split], Y[right_split], curr_node, depth=depth+1)
                curr_node.split_value = real_split_value   
                
            ## Setting the current node values
            
            if(Y.dtype.name == "category"):
                curr_node.value = Y.mode(dropna=True)[0]
            else:
                curr_node.value = Y.mean()

            # Setting the current node depth
            curr_node.depth = depth 
            return curr_node

        # max depth reached or equal values or dataset end reached. No more splitting possibe/required
        else:
            if(Y.dtype.name == "category"): # Classification. We return a node with its value set as the mode of the output column
                return Node(val = Y.mode(dropna=True)[0], depth = depth)
            else: # Regression. We return a node with its value set as the mean of the output column
                return Node(val = Y.mean(), depth = depth)

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        """
        self.output_type = y.dtype
        self.attribute = y.name
        self.X_size = len(X)
        self.root = self.build_tree(X, y, None)
        self.root.freq = 1

    def predict(self, X, max_depth=np.inf):
        """
        Funtion to run the decision tree on test inputs
        """
        Y = [] # list to store the predicted values
        for x in (X.index):
            ## for each row in the test dataset, we call the node_val function to get the prediction value
            Y.append(self.root.node_val(X.loc[x], max_depth=max_depth))
        ## converting the list to a pandas series
        Y_hat = pd.Series(Y, name=self.attribute).astype(self.output_type)
        return Y_hat

    def plot(self):
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.print_tree(space = 1)
        print('\n')
