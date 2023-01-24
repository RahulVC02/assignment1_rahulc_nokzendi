import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from metrics import *
from tqdm import tqdm


np.random.seed(42)

# We have deleted the column containing the car names as they are unique for each instance
# We did this in the csv file itself
df = pd.read_csv('./auto-efficiency.csv')
df.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7},inplace=True)

# # There are some missing values in the horsepower column
# # We can drop those rows

# Here i have dropped those rows which have the value '?' in the horsepower column
df = df[df[3]!='?'].reset_index(drop=True)
df.dropna(inplace=True)

#Here, y is the labelled outputs and X is the input features of the dataset. I separated them
y = df[0]
X = df.drop(0, axis=1).reset_index(drop=True)

#change column name of dataframe to enable indexing from 0
X.rename(columns={1:0,2:1,3:2,4:3,5:4,6:5,7:6}, inplace=True)

X = X.astype('float64')


#This is the train-test split of the dataset where the first 320 of 392 instances
#are used for training and the rest for testing
X_frame = X[:320].reset_index(drop=True)
y_series = y[:320].reset_index(drop=True)

X_final_test = X[320:].reset_index(drop=True)
y_final_test = y[320:].reset_index(drop=True)


# I have implemented k-fold cross validation here to determine the appropriate
# hyperparameters for the decision tree
k = 8  # number of folds
depths = [1, 2, 3,4,5,6,7,8,9,10]
fold_size = int(len(X_frame)/k)


#Since the rmse value has the same dimensions as the output, I will use the mean of the output
#to normalize the error and get an approximate accuracy value

#My defined metric for accuracy using rmse is acc = (1-rmse/output_mean)*100
#where rmse/output_mean is the error in percentage

output_mean = y_final_test.mean()

optimal_depth = None
criterion = "information_gain"
min_avg_rmse = np.inf

for depth in tqdm(depths):
    current_avg_rmse=0
    
    for i in range(k):
        X_train = pd.concat((X_frame[0:i*fold_size], X_frame[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
        y_train = pd.concat((y_series[0:i*fold_size], y_series[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
        
        X_validation = X_frame[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
        y_validation = y_series[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
        
        tree = DecisionTree(criterion=criterion, max_depth=depth)
        tree.fit(X_train, y_train)
        y_hat = tree.predict(X_validation)
        current_rmse = rmse(y_hat,y_validation)

        current_avg_rmse+=current_rmse
    
    current_avg_rmse = current_avg_rmse/k

    if(current_avg_rmse<min_avg_rmse):
        optimal_depth = depth
        min_avg_rmse = current_avg_rmse
            

#I calculated the optimal hyperparams for the training set using above loops 
#minimizing rmse values.

    
#I implemented the decision tree using my own implementation below measuring rmse and percentage accuracy
custom_model = DecisionTree(criterion=criterion, max_depth=optimal_depth)
custom_model.fit(X_frame, y_series)
y_hat = custom_model.predict(X_final_test)
custom_rmse_value = rmse(y_hat, y_final_test)
custom_mae_value = mae(y_hat,y_final_test)
custom_accuracy_value = (1-(custom_rmse_value/output_mean))*100

#I implemented the decision tree using sklearn's implementation below measuring rmse and percentage accuracy
sklearn_model = DecisionTreeRegressor(max_depth=optimal_depth)
sklearn_model.fit(X_frame, y_series)
y_hat_sklearn = sklearn_model.predict(X_final_test)
sklearn_rmse_value = rmse(y_hat_sklearn, y_final_test)
sklearn_mae_value = mae(y_hat_sklearn,y_final_test)
sklearn_accuracy_value = (1-(sklearn_rmse_value/output_mean))*100

print("Optimal Depth:{}, Optimal Criterion:{}".format(optimal_depth,criterion))
print("Custom Model RMSE:{}, Custom Model MAE:{}, Custom Model Accuracy:{}".format(custom_rmse_value,custom_mae_value,custom_accuracy_value))
print("Sklearn Model RMSE:{}, Sklearn Model MAE:{}, Sklearn Model Accuracy:{}".format(sklearn_rmse_value,sklearn_mae_value,sklearn_accuracy_value))