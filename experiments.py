
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

N = 30
M = 5
K = 3

## Populate function used to create data
## Discrete outputs can take 5 possible values from 0 to 4
def populate(N, M, type):
    if(type == 0):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif(type == 1):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(5, size=N), dtype="category")
        return X, y
    elif(type == 2):
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(5, size=N), dtype="category")
        return X, y
    else:
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
        return X, y


def plot_values(data, x_axis, y_axis, title):
    # Create the heatmap for learning time
    plt.imshow(data, extent=[x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()], origin='lower')
    plt.colorbar(label = "Time in seconds")

    # Adding x and y axis labels
    plt.xlabel('N')
    plt.ylabel('M')

    # Adding the appropriate title
    plt.title(title)

    # Show the plot
    plt.show()



def experiment(N, M):
    ## Looping for all the four cases {i: Case == 0: RIRO, 1: RIDO, 2: DIDO, 3: DIRO}
    for i in range(0, 4):
        # Initializing the arrays to store the learning and predicting time
        learning_time = np.arange(N*M).reshape(N, M).astype(float)
        predicting_time = np.arange(N*M).reshape(N, M).astype(float)
        std_learning = np.arange(N*M).reshape(N, M).astype(float)
        std_predicting = np.arange(N*M).reshape(N, M).astype(float)
        x_axis = np.arange(1, N+1)
        y_axis = np.arange(1, M+1)
        ## Looping for the different values of N and M
        ## N ranges from 1 to 30, M ranges from 1 to 5
        for n in range(1, N+1):
            for m in range(1, M+1):
                # Temporary arrays to store the learning and predicting time 
                learning_time_temp = np.zeros(K)
                predicting_time_temp = np.zeros(K)
                # Looping to find avg values for K iterations
                for k in range(K):
                    X, y = populate(n, m, i) # Getting our dataset 

                    # Creating the decision tree and storing the time taken
                    tree = DecisionTree(criterion="information_gain")
                    start_time = time.time()
                    tree.fit(X, y)
                    end_time = time.time()
                    learning_time_temp[k] = end_time - start_time

                    # Predicinng using our decision tree and storing the time taken
                    start_time = time.time()
                    y_cap = tree.predict(X)
                    end_time = time.time()
                    predicting_time_temp[k] = end_time - start_time
                
                # Storing the average and standard deviation of the learning and predicting time
                learning_time[n-1][m-1] = np.mean(learning_time_temp)
                predicting_time[n-1][m-1] = np.mean(predicting_time_temp)
                std_learning[n-1][m-1] = np.std(learning_time_temp)
                std_predicting[n-1][m-1] = np.std(predicting_time_temp)
        
        # Setting the title for the plot depending on the decision tree case
        title = ""
        if(i == 0):
            title = "Real Input Real Output"
        elif(i == 1):
            title = "Real Input Discrete Output"
        elif(i == 2):
            title = "Discrete Input Discrete Output"
        else:
            title = "Discrete Input Real Output"

        # plotting the learning and predicting average time and standard deviation
        plot_values(learning_time, x_axis, y_axis, title + " learning time")
        plot_values(std_learning, x_axis, y_axis, title + " learning std")
        plot_values(predicting_time, x_axis, y_axis, title + " predicting time")
        plot_values(std_predicting, x_axis, y_axis, title + " predicting std")


experiment(N, M)
