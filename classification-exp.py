import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from tqdm import tqdm
np.random.seed(42)

#Question 1a
#importing the dataset
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
# import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=y)

print("Part A: Real Input and Discrete Output")

#making the train-test split
X_train = pd.DataFrame(X[0:70,:])
y_train = pd.Series(y[0:70], dtype = "category")
X_test = pd.DataFrame(X[70:100,:])
y_test = pd.Series(y[70:100], dtype = "category")

#training and testing the data on both criteria
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))

    #reporting metrics
    print("Per-Class Precision and Recall are as follows:")
    for cls in y_test.unique():
        print("Class:{}, Precision:{}, Recall:{}".format(cls, precision(y_hat, y_test, cls), recall(y_hat, y_test, cls)))

    print("   ")   

print("    ")

print("Part B: Nested Cross Validation")

## Question 1b

depth = [0,1,2,3,4,5,6,7,8,9,10]
criteria = ['information_gain', 'gini_index']
k = 5
fold_size = int(len(X)/k)

#setting datatype for classification
X_frame = pd.DataFrame(X)
y_series = pd.Series(y,dtype="category")

avg_testing_accuracy = 0

for i in (range(5)):

    #making the outer train-test splits leading to generation of 5 models
    X_final_test = X_frame[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
    y_final_test = y_series[i*fold_size:(i+1)*fold_size].reset_index(drop=True)

    X_total_train = pd.concat((X_frame[0:i*fold_size], X_frame[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
    y_total_train = pd.concat((y_series[0:i*fold_size], y_series[(i+1)*fold_size:]), axis=0).reset_index(drop=True)

    # for j in range(4):
    # #making the train-validation splits for the k models corresponding to k folds
    # X_validation = X_frame[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
    # y_validation = y_series[i*fold_size:(i+1)*fold_size].reset_index(drop=True)

    # X_train = pd.concat((X_frame[0:i*fold_size], X_frame[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
    # y_train = pd.concat((y_series[0:i*fold_size], y_series[(i+1)*fold_size:]), axis=0).reset_index(drop=True)

    best_depth = None
    best_criterion = None
    max_avg_validation_accuracy=None

    #iterating over the entire hyperparameter space
    for d in depth:
        for criterion in criteria:
            current_avg_validation_accuracy=None
            accuracy_total = 0

            for j in range(4):
                X_validation = X_total_train[j*fold_size:(j+1)*fold_size].reset_index(drop=True)
                y_validation = y_total_train[j*fold_size:(j+1)*fold_size].reset_index(drop=True)

                X_current_train = pd.concat((X_total_train[0:j*fold_size], X_total_train[(j+1)*fold_size:]), axis=0).reset_index(drop=True)
                y_current_train = pd.concat((y_total_train[0:j*fold_size], y_total_train[(j+1)*fold_size:]), axis=0).reset_index(drop=True)

                tree = DecisionTree(criterion=criterion, max_depth=d)
                tree.fit(X_current_train, y_current_train)
                y_hat = tree.predict(X_validation)
                accuracy_score = accuracy(y_hat, y_validation)

                accuracy_total+=accuracy_score

            current_avg_validation_accuracy=accuracy_total/4

            if(max_avg_validation_accuracy==None or current_avg_validation_accuracy>max_avg_validation_accuracy):
                max_avg_validation_accuracy = current_avg_validation_accuracy
                best_depth=d
                best_criterion=criterion

    
    final_tree = DecisionTree(criterion=best_criterion,max_depth=best_depth)
    final_tree.fit(X_total_train,y_total_train)

    y_hat = final_tree.predict(X_final_test)
    testing_accuracy = accuracy(y_hat,y_final_test)
    avg_testing_accuracy+=testing_accuracy
    print("For Fold: {}, Best depth:{}, Best criteria:{}, Testing accuracy:{}".format(i,best_depth, best_criterion,testing_accuracy))


print("Overall Average Testing Accuracy over 5 folds:{}".format(avg_testing_accuracy/5))