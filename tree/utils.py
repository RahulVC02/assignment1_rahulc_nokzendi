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



def information_gain(Y, attr, criterion = None):

  ## DISCRETE INPUT AND DISCRETE OUTPUT USING INFORMATION GAIN
  if(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "information_gain"):
    parent_entropy = entropy(Y) # Calculating the entropy of the parent node
    total_size = Y.size
    classes = np.unique(attr) # Getting a list of all the unique classes from our attribute (all unique values in our column)
    child_entropy = 0
    for i in classes:
      curr_class = Y[attr == i] # Masking our Series with the condition attr == i
      class_entropy = entropy(curr_class) # Calculating the entropy for each class
      child_entropy += (curr_class.size/total_size) * class_entropy # Taking the weighted average of the entropies of each class
    return parent_entropy - child_entropy

  ## DISCRETE INPUT AND DISCRETE OUTPUT USING GINI INDEX
  elif(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "gini_index"):
    parent_gini = gini_index(Y) # Calculating the entropy of the parent node
    total_size = Y.size
    classes = np.unique(attr) # Getting a list of all the unique classes from our attribute (all unique value in our column)
    child_gini = 0
    for i in classes:
      curr_class = Y[attr == i] # Masking our Series with the condition attr = i
      class_gini = gini_index(curr_class) # Calculating the gini index for each class
      child_gini += (curr_class.size/total_size) * class_gini # Taking weighted average of the gini indexes of each class
    return parent_gini - child_gini
  
  ## REAL INPUT AND DISCRETE OUTPUT
  elif(Y.dtype.name == "float64" and attr.dtype.name == "category"):
    #merging the two Series together so I can get a merged dataframe which can be sorted together while retaining the index.
    table = pd.concat([attr,Y],axis=1).reindex(attr.index)
    table.columns = ["attribute","output"]

    #sorting step
    table.sort_values(by=["attribute","output"],inplace=True)

    values = table['attribute']
    output = table['output']

    optimum_split = None
    maximum_gain = -np.inf
    current_split = None

    if(criterion=="information_gain"):
      initial_disagreement = entropy(Y)
    elif(criterion=="gini_index"):
      initial_disagreement = gini_index(Y)
    
    current_disagreement=None

    for i in range(1,len(values)):

      #skipping those splits where the values don't change, only checking those splits around which transitions occur
      if(values[i]==values[i-1]):
        continue

      current_split = (values[i]+values[i-1])/2
      split_one = output[values<=current_split]
      split_two = output[values>current_split]

      if(criterion=="information_gain"):
        component_one = (split_one.size()/output.size())*(entropy(split_one))
        component_two = (split_two.size()/output.size())*(entropy(split_two))
      elif(criterion=="gini_index"):
        component_one = (split_one.size()/output.size())*(gini_index(split_one))
        component_two = (split_two.size()/output.size())*(gini_index(split_two))

      current_disagreement = 0
      current_disagreement = component_one + component_two

      if(initial_disagreement-current_disagreement>maximum_gain):
        maximum_gain = initial_disagreement-current_disagreement
        optimum_split = current_split

      return {maximum_gain,optimum_split}

  ## DISCRETE INPUT AND REAL OUTPUT
  elif(Y.dtype.name == "float64" and attr.dtype.name == "category"):
    parent_variance = np.var(Y) # Calculating the variance of the parent node
    total_size = Y.size
    classes = np.unique(attr) # Getting a list of all the unique classes from our attribute
    child_variance = 0
    for i in classes:
      curr_class = Y[attr == i] # Masking our Series with the condition attr = i
      class_variance = np.var(curr_class) # Calculating the variance for each clas
      child_variance += (curr_class.size/total_size) * class_variance # Taking weighted average of the variances of each class
    return parent_variance - child_variance

  ## REAL INPUT AND REAL OUTPUT
  elif(Y.dtype.name == "float64" and attr.dype.name == "float64"):
    #merging the two Series together so I can get a merged dataframe which can be sorted together while retaining the index.
    table = pd.concat([attr,Y],axis=1).reindex(attr.index)
    table.columns = ["attribute","output"]

    #sorting step
    table.sort_values(by=["attribute","output"],inplace=True)

    values = table['attribute']
    output = table['output']

    optimum_split = None
    maximum_gain = -np.inf
    current_split = None
    parent_variance = np.var(Y)
    total_size = Y.size

    for i in range(1, len(values)):
      if(values[i] == values[i-1]):
        continue
      current_split = (values[i] + values[i-1]) / 2
      split_one = output[values<=current_split]
      split_two = output[values>current_split]
      overall_variance = parent_variance
      overall_variance -= (split_one.size / total_size) * np.var(split_one)
      overall_variance -= (split_two.size / total_size) * np.var(split_two)
      if(overall_variance > maximum_gain):
        maximum_gain = overall_variance
        optimum_split = current_split
      return {maximum_gain, optimum_split}
