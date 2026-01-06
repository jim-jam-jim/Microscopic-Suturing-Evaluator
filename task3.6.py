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

    return (((X.T @ X @ w - X.T@y)))/(2*len(X))


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

        w = w - (eta*gradient_vector)
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
        w = w - (eta*gradient_vector)       
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
    

def plot_losses(point_count, mse_losses_train, mse_losses_val, x_label, y_label, title, directory):
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
    # plt.show()
    plt.savefig(os.path.join(directory, title+'.png'))

    
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

    directory         = 'Task_3_6'
    # constructing the training data set
    train_data = pd.read_csv(train_file_path, header = None)

    data_fraction      = 1
    sample_names       = train_data.iloc[:int(data_fraction*len(train_data)), 0]
    X_train            = train_data.iloc[:int(data_fraction*len(train_data)), 2:]
    y_train            = train_data.iloc[:int(data_fraction*len(train_data)), 1]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train_un_normalized = np.copy(y_train)
#    print(X_train)

#    feature_means, feature_stds = normalize(X_train)

    # will be required to get un-normalized predictions
 #   y_mean, y_std = normalize_y(y_train)
    y_mean = 0
    y_std = 1
    # learning rate
    eta = 0.0005
#    gradient descent with fixed number of iterations
    # iter_count = 200
    # print(f"Using fixed no of iterations with iterations = {iter_count}, eta = {eta}")    
    # w_maxit, weight_vectors_maxit = gradient_descent_maxit(X_train, y_train, eta, iter_count)


    # print(y_pred_un_normalized)
    # print(y_train_un_normalized)
    # gradient descent with thresholding on relative iteration

    error_threshold = 0.001
    print(f"Using relative thresholding with threshold = {error_threshold}, eta = {eta}")

    w_reltol, weight_vectors_reltol = gradient_descent_reltol(X_train, y_train, eta, error_threshold)

    print(f"Number of iterations that took place with: {len(weight_vectors_reltol)}")

    # # print(mse_losses_reltol)
    # # print(len(mse_losses_reltol))
    
    # TRAIN MSE AND MAE LOSS
    y_pred_train_un_normalized = predict(X_train, w_reltol, y_mean, y_std)
#    print(y_std*y_std)
    print(f"The MSE loss on training data: {np.mean((y_pred_train_un_normalized - y_train_un_normalized)*(y_pred_train_un_normalized - y_train_un_normalized))}")
    print(f"The MAE loss on training data: {np.mean(np.absolute(y_pred_train_un_normalized - y_train_un_normalized))}")

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

    # predicting normalized y values on normalized X values for validation set

    y_pred_val_un_normalized = predict(X_val, w_reltol, y_mean, y_std)
    
    print(f"The MSE loss on validation data: {np.mean((y_pred_val_un_normalized - y_val_un_normalized)*(y_pred_val_un_normalized - y_val_un_normalized))}")
    print(f"The MAE loss on validation data: {np.mean(np.absolute(y_pred_val_un_normalized - y_val_un_normalized))}")

    
    
    # PLOTTING THE LEARNING CURVE

    iteration_losses_train = compute_iter_losses(X_train, weight_vectors_reltol, y_train_un_normalized, y_mean, y_std)
    iteration_losses_val   = compute_iter_losses(X_val, weight_vectors_reltol, y_val_un_normalized, y_mean, y_std)

    plot_losses(20, iteration_losses_train, iteration_losses_val, "Iterations", "MSE error", f"Linear regression, fraction data= {data_fraction}, threshold = 0.005", directory)
    # plot_losses(20, iteration_losses_train, iteration_losses_val, "Iterations", "MSE error", f"Linear regression, eta = {eta}, iterations = {iter_count}")

    # ANALYZING THE RESULTS
    result = pd.DataFrame()
    result['Pred_val'] = y_pred_val_un_normalized
    result['True_val'] = y_val_un_normalized
    print(result)
    

    # PLOTTING THE CLASS-WISE LOSSES
    class_labels = np.unique(y_train_un_normalized)
    class_wise_losses = np.zeros(len(class_labels))
    class_wise_freq   = np.zeros(len(class_labels))
    class_wise_avg_losses = np.zeros(len(class_labels))

    for i in range(len(y_train_un_normalized)):
        class_wise_losses[int(y_train_un_normalized[i]) - 1] += (y_pred_train_un_normalized[i] - y_train_un_normalized[i])**2
        class_wise_freq[int(y_train_un_normalized[i]) - 1] += 1

    for i in range(len(class_wise_losses)):
        class_wise_avg_losses[i] = class_wise_losses[i]/class_wise_freq[i]


#    print(class_wise_losses)
        

    
