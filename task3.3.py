import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
# FILE FOR LINEAR REGRESSION 
# takes a feature matrix and normalizes each feature/column)
def normalize(X):
    for feature_index in range(len(X[0])):
        sum = 0
        squared_sum = 0

        for x_index in range(len(X)):
            sum         += X[x_index][feature_index]
            squared_sum += X[x_index][feature_index]**2

        mean = sum/len(X)
        variance = squared_sum/len(X) - mean**2

        std = np.sqrt(variance)

        for x_index in range(len(X)):
            X[x_index][feature_index] = (X[x_index][feature_index] - mean)/std


    return

def normalize_y(y):
    sum = 0
    squared_sum = 0
    for y_index in range(len(y)):
        sum += y[y_index]
        squared_sum += y[y_index]**2

    mean = sum/len(y)
    variance = squared_sum/len(y) - mean**2

    std = np.sqrt(variance)

    for y_index in range(len(y)):
        y[y_index] = (y[y_index] - mean)/std

    return (mean, std)

def gradient(w, X, y):

    return (((X.T @ X @ w - X.T@y)))/(len(X))


def gradient_descent_maxit(X, y, eta, iter_count):
    w = np.zeros(len(X[0]))
#    print(X.shape)
#    print(w.shape)
    mse_losses = []
    mae_losses = []
    for iter in range(iter_count):
 #       X@w
        mse_losses.append(np.sum((X@w - y).T@(X@w - y)/len(X)))
        mae_losses.append(np.sum(np.absolute(X@w - y)/len(X)))
        gradient_vector = gradient(w, X, y)
#        print(gradient_vector)
        w = w - eta*gradient_vector

    return (w, mse_losses, mae_losses)

## TO DO gradient_descent_reltol

def gradient_descent_reltol(X, y, eta, threshold):
    w = np.zeros(len(X[0]))
    mse_losses = []
    mae_losses = []

    curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X))
    prev_loss = 0
    while np.absolute((prev_loss-curr_loss)/curr_loss) > threshold:
        
        mse_losses.append(np.sum((X@w - y).T@(X@w - y)/len(X)))
        mae_losses.append(np.sum(np.absolute(X@w - y)/len(X)))

        gradient_vector = gradient(w, X, y)
#        print(gradient_vector)
        w = w - eta*gradient_vector
        prev_loss = curr_loss
        curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X))

    return (w, mse_losses, mae_losses)
    

def predict(X, w, y_mean, y_std):
    y_pred_un_normalized = (X@w)*y_std + y_mean

    
    return y_pred_un_normalized

if __name__ == '__main__':

    # parsing the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-trp", "--train_path", help="Path to training data")
    parser.add_argument("-vap", "--val_path", help="Path to validation data")
    parser.add_argument("-tep", "--test_path", help="Path to test data")
    parser.add_argument("-oup", "--out_path", help="Path to file where output has to be written")
    parser.add_argument("-s", "--section", help="Section value", type = int)

    args = parser.parse_args()


    train_file_path   = args.train_path
    val_file_path     = args.val_path
    test_file_path    = args.test_path
    output_file_path  = args.out_path
    section           = args.section
    # constructing the training data set
    train_data = pd.read_csv(train_file_path, header = None)

    sample_names = train_data.iloc[:, 0]
    X_train            = train_data.iloc[:, 2:]
    y_train            = train_data.iloc[:, 1]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Using sklearn

    linear_model = LinearRegression().fit(X_train, y_train)

    y_train_pred = linear_model.predict(X_train)

    print(f"The MSE for sklearn model on training data is {np.sum((y_train_pred - y_train)*(y_train_pred - y_train))/len(y_train_pred)}")
    print(f"The MAE for sklearn model on training data is {np.sum(np.absolute(y_train_pred - y_train))/len(y_train_pred)}")


    val_data = pd.read_csv(val_file_path, header = None)

    sample_names = val_data.iloc[:, 0]
    X_val            = val_data.iloc[:, 2:]
    y_val            = val_data.iloc[:, 1]
    
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
#    print(len(linear_model.coef_))

    y_val_pred = linear_model.predict(X_val)

    print(f"The MSE for sklearn model on validation data is {np.sum((y_val_pred - y_val)*(y_val_pred - y_val))/len(y_val_pred)}")
    print(f"The MAE for sklearn mode on validation data is {np.sum(np.absolute(y_val_pred - y_val))/len(y_val_pred)}")

    result = pd.DataFrame()
    result['Pred_val'] = y_train_pred
    result['True_val'] = y_train
    print(result)

    

#     y_train_un_normalized = np.copy(y_train)
# #    print(X_train)

#     normalize(X_train)
#     y_mean, y_std = normalize_y(y_train)
# #    print(X_train)
#         # learning rate
#     eta = 0.001
# #    gradient descent with fixed number of iterations
#     iter_count = 500
#     # w_maxit, mse_losses_maxit, mae_losses_maxit = gradient_descent_maxit(X_train, y_train, eta, iter_count)

#     # print(mse_losses_maxit)
#     # y_pred_un_normalized = predict(X_train, w_maxit, y_mean, y_std)
# #    print(y_pred_un_normalized)
# #    print(y_train_un_normalized)
#     # gradient descent with thresholding on relative iteration
#     error_threshold = 0.001

#     w_reltol, mse_losses_reltol, mae_losses_reltol = gradient_descent_reltol(X_train, y_train, eta, error_threshold)


#     print(mse_losses_reltol)
#     print(len(mse_losses_reltol))
    

