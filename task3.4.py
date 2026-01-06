import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
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

    return (((X.T @ X @ w - X.T@y)))/(2*len(X))


def gradient_descent_maxit(X, y, eta, iter_count):
    w = np.zeros(len(X[0]))
    curr_weights = []
#    print(X.shape)
#    print(w.shape)

    for iter in range(iter_count):
 #       X@w
        curr_weights.append(w)
        gradient_vector = gradient(w, X, y)
#        print(gradient_vector)
        w = w - eta*gradient_vector

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
        w = w - eta*gradient_vector
        prev_loss = curr_loss
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
    

def plot_losses(point_count, mse_losses_train, mse_losses_val, x_label, y_label, title, directory):
    print(len(mse_losses_train))
    print(len(mse_losses_val))
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
#    plt.show()
    plt.savefig(os.path.join(directory, title + '.png'))
    plt.cla()




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

    directory = 'Task_3_4'
    # constructing the training data set
    train_data = pd.read_csv(train_file_path, header = None)

    sample_names = train_data.iloc[:, 0]
    X_train            = train_data.iloc[:, 2:]
    y_train            = train_data.iloc[:, 1]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train_un_normalized = np.copy(y_train)
#    print(X_train)

    # feature_means, feature_stds = normalize(X_train)
    # y_mean, y_std = normalize_y(y_train)
    y_mean = 0
    y_std  = 1
#    print(X_train)
#        learning rate
#    eta = 0.001
# #    gradient descent with fixed number of iterations
#     iter_count = 500
    
    # w_maxit, mse_losses_maxit, mae_losses_maxit = gradient_descent_maxit(X_train, y_train, eta, iter_count)

    # print(mse_losses_maxit)
    # y_pred_un_normalized = predict(X_train, w_maxit, y_mean, y_std)
#    print(y_pred_un_normalized)
#    print(y_train_un_normalized)
    # gradient descent with thresholding on relative iteration
    #############################################################################################
    # Using eta = 0.1 and top 10 features

    # USING SelectKBest for feature selection
    ## TODO check for different types of function other than f_regression
    feature_selector_sk = SelectKBest(k = 10)
    X_train_new_sk = feature_selector_sk.fit_transform(X_train, y_train)
    X_train_new_sk = np.c_[ X_train_new_sk, np.ones(len(X_train_new_sk)) ]       
    print(X_train_new_sk.shape)
    eta = 0.1
    error_threshold = 0.001

    w_reltol_sk, weight_vectors_reltol_sk = gradient_descent_reltol(X_train_new_sk, y_train, eta, error_threshold)
    print(f"The number of iterations that took place in select k best:{len(weight_vectors_reltol_sk)}")
    y_train_pred_sk = predict(X_train_new_sk, w_reltol_sk, y_mean, y_std)
    print(f"The MSE on training data(based on top 10 features given by SelectKBest) is {np.sum((y_train_pred_sk - y_train_un_normalized)*(y_train_pred_sk - y_train_un_normalized))/len(y_train_pred_sk)}")
    print(f"The MAE on training data(based on top 10 features given by SelectKBest) is {np.sum(np.absolute(y_train_pred_sk - y_train_un_normalized))/len(y_train_pred_sk)}")

    #    print(mse_losses_reltol)
    #    print(len(mse_losses_reltol))
    #    USING SelectFromModel FOR FEATURE SELECTION
    feature_selector_sf = SelectFromModel(Ridge(), max_features = 10)

    X_train_new_sf = feature_selector_sf.fit_transform(X_train, y_train)
    X_train_new_sf = np.c_[ X_train_new_sf, np.ones(len(X_train_new_sf)) ]       
    print(f"The number of features given by SelectFromModel selector = {len(X_train_new_sf[0])}")

    w_reltol_sf, weight_vectors_reltol_sf = gradient_descent_reltol(X_train_new_sf, y_train, eta, error_threshold)
    print(f"The number of iterations that took place in select from model:{len(weight_vectors_reltol_sf)}")

    y_train_pred_sf = predict(X_train_new_sf, w_reltol_sf, y_mean, y_std)

     
    print(f"The MSE on training data(based on top 10 features given by SelectFromModel) is {np.sum((y_train_pred_sf - y_train_un_normalized)*(y_train_pred_sf - y_train_un_normalized))/len(y_train_pred_sf)}")
    print(f"The MAE on training data(based on top 10 features given by SelectFromModel) is {np.sum(np.absolute(y_train_pred_sf - y_train_un_normalized))/len(y_train_pred_sf)}")



    # VALIDATION MSE AND MAE LOSS

    # loading the validation data

    val_data = pd.read_csv(val_file_path, header = None)
    sample_names_val = train_data.iloc[:, 0]
    X_val            = val_data.iloc[:, 2:]
    y_val            = val_data.iloc[:, 1]
    
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
    y_val_un_normalized = np.copy(y_val)

    # IMP - normalized the validation and test data with respect to mean and standard deviation of training data

#    normalize_val_test(X_val, feature_means, feature_stds)
#    normalize_val_test_y(y_val,y_mean, y_std)


    # USING TOP 10 FEATURES GIVEN BY SelectKBest FOR VALIDATION DATA

    X_val_new_sk = feature_selector_sk.transform(X_val) # IMP - Use transform; NOT fit_transform
    X_val_new_sk = np.c_[ X_val_new_sk, np.ones(len(X_val_new_sk)) ]       
    y_val_pred_sk = predict(X_val_new_sk, w_reltol_sk, y_mean, y_std)
    print(f"The MSE on validation data(based on top 10 features given by SelectKBest) is {np.sum((y_val_pred_sk - y_val_un_normalized)*(y_val_pred_sk - y_val_un_normalized))/len(y_val_pred_sk)}")
    print(f"The MAE on validation data(based on top 10 features given by SelectKBest) is {np.sum(np.absolute(y_val_pred_sk - y_val_un_normalized))/len(y_val_pred_sk)}")


    # USING TOP 10 FEATURES GIVEN BY SelectFromModel FOR VALIDATION DATA

    X_val_new_sf = feature_selector_sf.transform(X_val) # IMP - Use transform; NOT fit_transform
    X_val_new_sf = np.c_[ X_val_new_sf, np.ones(len(X_val_new_sf)) ]       
    y_val_pred_sf = predict(X_val_new_sf, w_reltol_sf, y_mean, y_std)
    print(f"The MSE on validation data(based on top 10 features given by SelectFromBest) is {np.sum((y_val_pred_sf - y_val_un_normalized)*(y_val_pred_sf - y_val_un_normalized))/len(y_val_pred_sf)}")
    print(f"The MAE on validation data(based on top 10 features given by SelectFromBest) is {np.sum(np.absolute(y_val_pred_sf - y_val_un_normalized))/len(y_val_pred_sf)}")


    

    # PLOTTING THE LEARNING CURVE

    # FOR THE FEATURES GIVEN BY SK

    iteration_losses_train_sk = compute_iter_losses(X_train_new_sk, weight_vectors_reltol_sk, y_train_un_normalized, y_mean, y_std)
    iteration_losses_val_sk = compute_iter_losses(X_val_new_sk, weight_vectors_reltol_sk, y_val_un_normalized, y_mean, y_std)

    plot_losses(20, iteration_losses_train_sk, iteration_losses_val_sk, "Iterations", "MSE error", f"Linear regression SelectKBest, eta = {eta}, threshold = {error_threshold}", directory)
    
    # FOR THE FEATURES GIVEN BY SF
#    print(weight_vectors_reltol_sf)
#    print(weight_vectors_reltol_sk)
    iteration_losses_train_sf = compute_iter_losses(X_train_new_sf, weight_vectors_reltol_sf, y_train_un_normalized, y_mean, y_std)
    iteration_losses_val_sf = compute_iter_losses(X_val_new_sf, weight_vectors_reltol_sf, y_val_un_normalized, y_mean, y_std)


    plot_losses(20, iteration_losses_train_sf, iteration_losses_val_sf, "Iterations", "MSE error", f"Linear regression SelectFromModel, eta = {eta}, threshold = {error_threshold}", directory)

    
