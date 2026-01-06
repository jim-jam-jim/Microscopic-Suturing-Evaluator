import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
# FILE FOR LINEAR REGRESSION 
# takes a feature matrix and normalizes each feature/column)
def normalize(X):
    
    feature_means = np.zeros(len(X[0])) # will be used for normalizing validation and test features
    feature_stds  = np.zeros(len(X[0]))
    for feature_index in range(len(X[0])):
        sum = 0
        squared_sum = 0

        for x_index in range(len(X)):
            sum         += X[x_index][feature_index]
            squared_sum += X[x_index][feature_index]**2

        mean = sum/len(X)
        variance = squared_sum/len(X) - mean**2

        std = np.sqrt(variance)
        
        feature_means[feature_index] = mean
        feature_stds[feature_index]  = std
        
        for x_index in range(len(X)):
            X[x_index][feature_index] = (X[x_index][feature_index] - mean)/std


    return (feature_means, feature_stds)

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

def normalize_val_test(X, feature_means, feature_stds):

    for feature_index in range(len(X[0])):
        for x_index in range(len(X)):
            X[x_index][feature_index] = (X[x_index][feature_index] - feature_means[feature_index])/feature_stds[feature_index]

    return

def normalize_val_test_y(y, y_mean, y_std):

    for y_index in range(len(y)):
        y[y_index] = (y[y_index] - y_mean)/y_std


    return
            
        

def gradient(w, X, y):

    return (((X.T @ X @ w - X.T@y)))/(len(X))


def gradient_descent_maxit(X, y, eta, iter_count):
    w = np.zeros(len(X[0]))
    # print(X.shape)
    # print(w.shape)
    curr_weights = []
    for iter in range(iter_count):
        # X@w
        curr_weights.append(w)
        gradient_vector = gradient(w, X, y)
#        print(gradient_vector/np.sqrt(gradient_vector* gradient_vector))

        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))
#        print(w)
    return (w, curr_weights)

## TO DO gradient_descent_reltol

def gradient_descent_reltol(X, y, eta, threshold):
    w = np.zeros(len(X[0]))
    curr_weights = []
    
    curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X))
    prev_loss = 0
    while np.absolute((prev_loss-curr_loss)/curr_loss) > threshold:
        
        curr_weights.append(w)
        gradient_vector = gradient(w, X, y)
#        print(gradient_vector)
#        w = w - (eta*gradient_vector) 
        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))       
#        w = w - eta*gradient_vector
        prev_loss = curr_loss
        # print(curr_loss)
        curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X))

    return (w, curr_weights)
    

def predict(X, w, y_mean, y_std):
    y_pred_un_normalized = (X@w)*y_std + y_mean

    
    return y_pred_un_normalized


def compute_iter_losses(X, weight_vectors, y_true_un_normalized, y_mean, y_std):

    losses = []

    for weight_index in range(len(weight_vectors)):
        weights = weight_vectors[weight_index]

        y_pred_un_normalized = predict(X, weights, y_mean, y_std)

        losses.append(np.sum(((y_true_un_normalized - y_pred_un_normalized)*(y_true_un_normalized - y_pred_un_normalized))/len(y_true_un_normalized)))


    losses = np.array(losses)

    return losses
    

def plot_losses(point_count, mse_losses_train, mse_losses_val, x_label, y_label, title):
#    print(len(mse_losses_train))
#    print(len(mse_losses_val))
    gap_train = len(mse_losses_train)//point_count
    iterations_train = []
    mse_losses_train_sample = []

    for i in range(0, len(mse_losses_train), gap_train):
        mse_losses_train_sample.append(mse_losses_train[i])
        iterations_train.append(i/gap_train)

    plt.plot(iterations_train, mse_losses_train_sample, color = 'blue')


    gap_val = len(mse_losses_val)//point_count
    iterations_val = []
    mse_losses_val_sample = []

    for i in range(0, len(mse_losses_val), gap_val):
        mse_losses_val_sample.append(mse_losses_val[i])
        iterations_val.append(i/gap_val)

    plt.plot(iterations_val, mse_losses_val_sample, color = 'orange')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(["Train loss", "Validation loss"])
    plt.show()

    plt.savefig(title+'.png')

    
if __name__ == '__main__':

    # parsing the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--dir_path", help="Path to directory")

    args = parser.parse_args()

    dimensions = [2, 5, 10, 100]

    dir_path = args.dir_path

    error_diff = []
    train_sample_count = []
    for dim in dimensions:
        

    # constructing the training data set
        train_data = pd.read_csv(os.path.join(dir_path, f"{dim}_d_train.csv"), header = None)

        data_fraction      = 1
        sample_names       = train_data.iloc[:int(data_fraction*len(train_data)), 0]
        X_train            = train_data.iloc[:int(data_fraction*len(train_data)), 2:]
        y_train            = train_data.iloc[:int(data_fraction*len(train_data)), 1]

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        y_train_un_normalized = np.copy(y_train)
        #    print(X_train)

        feature_means, feature_stds = normalize(X_train)

        # will be required to get un-normalized predictions
        y_mean, y_std = normalize_y(y_train)
        # learning rate
        eta = 0.01
        #    gradient descent with fixed number of iterations
        # iter_count = 400
        # print(f"Using fixed no of iterations with iterations = {iter_count}, eta = {eta}")    
        # w_maxit, weight_vectors_maxit = gradient_descent_maxit(X_train, y_train, eta, iter_count)

        # print(mse_losses_reltol)
        # y_pred_un_normalized = predict(X_train, w_reltol, y_mean, y_std)
        # print(y_pred_un_normalized)
        # print(y_train_un_normalized)
        # gradient descent with thresholding on relative iteration

        error_threshold = 0.001
        print(f"Using relative thresholding with threshold = {error_threshold}, eta = {eta}")

        w_reltol, weight_vectors_reltol = gradient_descent_reltol(X_train, y_train, eta, error_threshold)
#        print(f"Weights are: {w_reltol}")
        print(f"Number of iterations that took place with: {len(weight_vectors_reltol)}")

        # print(mse_losses_reltol)
        # print(len(mse_losses_reltol))

        # TRAIN MSE AND MAE LOSS
        y_pred_train_un_normalized = predict(X_train, w_reltol, y_mean, y_std)
        #    print(y_std*y_std)
        train_mse = np.mean((y_pred_train_un_normalized - y_train_un_normalized)*(y_pred_train_un_normalized - y_train_un_normalized))



        test_data = pd.read_csv(os.path.join(dir_path, f"{dim}_d_test.csv"), header = None)
        sample_names_test = train_data.iloc[:, 0]
        X_test            = test_data.iloc[:, 2:]
        y_test            = test_data.iloc[:, 1]

        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        y_test_un_normalized = np.copy(y_test)

        # IMP - normalized the testidation and test data with respect to mean and standard deviation of training data

        normalize_val_test(X_test, feature_means, feature_stds)
        normalize_val_test_y(y_test,y_mean, y_std)

        # predicting normalized y values on normalized X testues for validation set

        y_pred_test_un_normalized = predict(X_test, w_reltol, y_mean, y_std)
        test_mse = np.mean((y_pred_test_un_normalized - y_test_un_normalized)*(y_pred_test_un_normalized - y_test_un_normalized))

        train_sample_count.append(len(y_train))
        error_diff.append(test_mse - train_mse)

    error_diff = np.array(error_diff)
    print(error_diff)
    dimensions = np.array(dimensions)
    train_sample_count = np.array(train_sample_count)
    plt.plot(dimensions, error_diff)
    plt.plot(dimensions, np.sqrt(dimensions/train_sample_count))
    plt.legend(["Actual loss diff", "Theoretical loss diff"])
    plt.savefig('Generalization_error.png')
    
