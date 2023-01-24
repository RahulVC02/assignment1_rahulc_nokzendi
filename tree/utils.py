import numpy as np
import pandas as pd


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # np.unique returns an array of all the unique values in the input array in sorted order
    # return_count = True also returns an array that contains the count of each unique vlaue
    value, counts = np.unique(Y, return_counts=True)
    # .sum() returns the sum of all the elemets in the array
    total = counts.sum()
    # Dividing the entire array by the total to get the probabilities
    probabilities = counts / total
    entropy = 0
    for pi in probabilities:
        entropy -= pi * np.log2(pi) # formula for entropy
    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # Explained above
    values, counts = np.unique(Y, return_counts = True)
    total = counts.sum()
    probabilities = counts / total

    # initializing the gini index
    gini_index = 1
    # getting the sum of all the squared probabilities
    sum_of_squares = np.sum(np.square(probabilities))
    # subtracting the sum of squares from the 1 to get the final gini_index
    gini_index -= sum_of_squares
    return gini_index

def information_gain(Y, attr, criterion=None):
    """
    Function to calculate the information gain
    """
    # REAL INPUT REAL OUTPUT
    if (Y.dtype.name == "float64" and attr.dtype.name == "float64"):
        # Merging the two series into a dataframe to sort it while maintaining the relationship between the two
        table = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        table.columns = ['attribute', 'output']

        # Sorting the table by the attribute/input column
        table.sort_values(by = 'attribute', inplace = True)

        # Separating out the input and output columns into numpy arrays
        # We are converting it to numpy arrays because it is easier to work with them in our
        # for loop below
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        maximum_gain = -np.inf
        parent_variance = np.var(Y)
        for i in range(1, len(input_values)):
            # Calculating the split point
            curr_split = float(input_values[i] + input_values[i-1]) / 2 
            # Splitting the table into two parts based on the split point
            # left_split, right_split are pandas series
            # left_split contains all the rows from table where the attribute is less than or equal to the split point.
            # Then it stores the corresponding output values. Similary for right_split.
            left_split = table[table['attribute'] <= curr_split]['output']
            right_split = table[table['attribute'] > curr_split]['output']

            # Initializing the child variance to 0
            child_variance = 0
            # Finding the weighted average of the variances of the left and right splits
            child_variance += (left_split.size / output_values.size) * (np.var(left_split))
            child_variance += (right_split.size / output_values.size) * (np.var(right_split))
            split_variance = parent_variance - child_variance
            # The lesser the child_variance the better the split.
            # Therefore, the greater the split_variance the better the split.
            if(split_variance > maximum_gain):
                maximum_gain = split_variance
                optimum_split = curr_split

        return maximum_gain, optimum_split
    
    # REAL INPUT DISCRETE OUTPUT USING INFORMATION GAIN
    elif (Y.dtype.name == "category" and attr.dtype.name == "float64" and criterion == "information_gain"):
        # Merging the two series into a dataframe to sort it while maintaining the relationship between the two
        table = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        table.columns = ['attribute', 'output']

        # Sorting the table by the attribute/input column
        table.sort_values(by = 'attribute', inplace = True)

        # Separating out the input and output columns into numpy arrays
        # We are converting it to numpy arrays because it is easier to work with them in our
        # for loop below
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        maximum_gain = -np.inf
        parent_entropy = entropy(Y)
        for i in range(1, len(input_values)):
            # Calculating the split point
            curr_split = float(input_values[i] + input_values[i-1]) / 2

            # Splitting the table into two parts based on the split point
            # left_split, right_split are pandas series
            # left_split contains all the rows from table where the attribute is less than or equal to the split point.
            # Then it stores the corresponding output values. Similary for right_split.
            left_split = table[table['attribute'] <= curr_split]['output'].to_numpy()
            right_split = table[table['attribute'] > curr_split]['output'].to_numpy()

            # Initializing the entropy of the child node
            child_entropy = 0
            # Finding the weighted average of the entropy of the left and right splits
            child_entropy += (left_split.size / output_values.size) * (entropy(left_split))
            child_entropy += (right_split.size / output_values.size) * (entropy(right_split))
            split_entropy = parent_entropy - child_entropy
            # The lesser the child_entropy the better the split.
            # Therefore, the greater the split_entropy the better the split.
            if(split_entropy > maximum_gain):
                maximum_gain = split_entropy
                optimum_split = curr_split
        return maximum_gain, optimum_split
    
    # REAL INPUT DISCRETE OUTPUT USING GINI INDEX
    elif (Y.dtype.name == "category" and attr.dtype.name == "float64" and criterion == "gini_index"):
        # Merging the two series into a dataframe to sort it while maintaining the relationship between the two
        table = pd.concat([attr, Y], axis = 1).reindex(attr.index)
        table.columns = ['attribute', 'output']
        table.sort_values(by = 'attribute', inplace=True)

        # Separating out the input and output columns into numpy arrays
        # We are converting it to numpy arrays because it is easier to work with them in our
        # for loop below
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        maximum_gain = -np.inf
        parent_gini = gini_index(Y)
        for i in range(1, len(input_values)):
            # Calculating the split point
            curr_split = float(input_values[i] + input_values[i-1]) / 2

            # Splitting the table into two parts based on the split point
            # left_split, right_split are pandas series
            # left_split contains all the rows from table where the attribute is less than or equal to the split point.
            # Then it stores the corresponding output values. Similary for right_split.
            left_split = table[table['attribute'] <= curr_split]['output'].to_numpy()
            right_split = table[table['attribute'] > curr_split]['output'].to_numpy()

            # Initializing the gini index of the child node
            child_gini = 0
            # Finding the weighted average of the gini indices of the left and right splits
            child_gini += (left_split.size / output_values.size) * (gini_index(left_split))
            child_gini += (right_split.size / output_values.size) * (gini_index(right_split))
            split_gini = parent_gini - child_gini
            # The lesser the child_gini the better the split.
            # Therefore, the greater the split_gini the better the split
            if(split_gini > maximum_gain):
                maximum_gain = split_gini
                optimum_split = curr_split
        return maximum_gain, optimum_split

    # DISCRETE INPUT DISCRETE OUTPUT USING INFORMATION GAIN
    elif (Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "information_gain"):
        parent_entropy = entropy(Y) # Calculating the entropy of the parent node
        total_size = Y.size
        classes = np.unique(attr) # Getting a list of all the unique values in the attribute (all unique values in our column)
        child_entropy = 0
        for i in classes:
            curr_class = Y[attr == i] # Masking our series with the condition that the attribute is equal class "i"
            class_entropy = entropy(curr_class) # Calculating the entropy for each class
            child_entropy += (curr_class.size / total_size) * class_entropy # Taking the weighted average of the entropies of each class
        return parent_entropy - child_entropy, None 

    # DISCRETE INPUT DISCRETE OUTPUT USING GINI INDEX
    elif (Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "gini_index"):
        parent_gini = gini_index(Y) # Calculating the gini index of the parent node
        total_size = Y.size
        classes = np.unique(attr) # Getting a list of all the unique values in the attribute (all unique values in our column)
        child_gini = 0
        for i in classes:
            curr_class = Y[attr == i] # Masking our series with the condition that the attribute is equal class "i"
            class_gini = gini_index(curr_class) # Calculating the gini index for each class
            child_gini += (curr_class.size / total_size) * class_gini # Taking the weighted average of the gini indices of each class
        return parent_gini - child_gini, None

    # DISCRETE INPUT REAL OUTPUT
    elif (Y.dtype.name == "float64" and attr.dtype.name == "category"):
        parent_variance = np.var(Y) # Calculating the variance of the parent node
        total_size = Y.size
        classes = np.unique(attr) # Getting a list of all the unique values in the attribute (all unique values in our column)
        child_variance = 0
        for i in classes:
            curr_class = Y[attr == i] # Masking our series with the condition that the attribute is equal class "i"
            class_variance = np.var(curr_class) # Calculating the variance for each class
            child_variance += curr_class.size / Y.size * class_variance # Taking the weighted average of the variances of each class
        return parent_variance - child_variance, None


