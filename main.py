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
#        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))
#        print(w)
    return (w, curr_weights)

## TO DO gradient_descent_reltol

def gradient_descent_reltol(X, y, eta, threshold, X_val, y_val):
    w = np.zeros(len(X[0]))
    curr_weights = []
    
    curr_loss = np.sum((X_val@w - y_val).T@(X_val@w - y_val)/len(X_val))
    prev_loss = 0
    while np.absolute((prev_loss-curr_loss)/curr_loss) > threshold:
        
        curr_weights.append(w)
        gradient_vector = gradient(w, X, y)
#        print(gradient_vector)
        w = w - (eta*gradient_vector) 
#        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))       
#        w = w - eta*gradient_vector
        prev_loss = curr_loss
        # print(curr_loss)
        curr_loss = np.sum((X_val@w - y_val).T@(X_val@w - y_val)/len(X_val))
#        print(curr_loss)
    return (w, curr_weights)
    
def gradient_ridge(w, X, y, lamda):

    return (((X.T @ X @ w - X.T@y + lamda*w)))/(2*len(X))



def gradient_descent_maxit_ridge(X, y, eta, iter_count, lamda):
    w = np.zeros(len(X[0]))
    # print(X.shape)
    # print(w.shape)
    curr_weights = []
    for iter in range(iter_count):
 #       X@w
        curr_weights.append(w)
        gradient_vector = gradient_ridge(w, X, y, lamda)
#        print(gradient_vector)
        w = w - (eta*gradient_vector)
#        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))
        
    return (w, curr_weights)

## TO DO gradient_descent_reltol_ridge

def gradient_descent_reltol_ridge(X, y, eta, threshold, lamda):
    w = np.zeros(len(X[0]))
    curr_weights = []
    
    curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X)) + lamda*w.T@w
    prev_loss = 0
    while np.absolute((prev_loss-curr_loss)/curr_loss) > threshold:
        
        curr_weights.append(w)
        gradient_vector = gradient_ridge(w, X, y, lamda)
#        print(gradient_vector)
        w = w - (eta*gradient_vector)               
#        w = w - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))               
        prev_loss = curr_loss
        curr_loss = np.sum((X@w - y).T@(X@w - y)/len(X))

    return (w, curr_weights)

def gradient_logit(class_params, X, y, class_count):

    gradient_vector = np.zeros(len(X[0])*(class_count-1))
    for x_index in range(len(X)):
        x = X[x_index]
        class_label = y[x_index]                   # assuming the class label is from 1 to class_count
        
#        class_weight = class_params[(class_label - 1)*len(x):(class_label)*len(x)]
        #np.exp(class_weight.T@x)
            
        den = 1

        for class_index in range(class_count-1):
            den += np.exp(class_params[(class_index)*len(x):(class_index + 1)*len(x)].T@x)
                # constructing the first term out of the two terms

#        print(den)
        term1 = np.zeros((class_count-1)*len(x))
#        print(class_label)
        if class_label != class_count:
#            print(class_label)
            term1[(class_label-1)*len(x):(class_label)*len(x)] = x
 
        # constructing the second term

        term2 = np.zeros((class_count-1)*len(x))

        for class_index in range(class_count-1):
            term2[(class_index)*len(x):(class_index+1)*len(x)] = np.exp(class_params[(class_index)*len(x):(class_index + 1)*len(x)].T@x) * x

        
        gradient_vector += term2/den - term1
            
        
            
    
    return (gradient_vector)/len(X)


def gradient_descent_multiclass_logit_maxit(X, y, eta, iter_count, class_count):
    parameters = np.zeros(len(X[0])*(class_count-1))


    curr_parameters = []

    for iter in range(iter_count):

        curr_parameters.append(parameters)

        gradient_vector = gradient_logit(parameters, X, y, class_count)

        parameters = parameters - (eta*gradient_vector)

    return (parameters, curr_parameters)

def predict_logit(X, parameters, class_count):
    predictions = np.zeros(len(X))
    for x_index in range(len(X)):
        x = X[x_index]
        pred_class = -1
        max_prob = -1.0
        den = 1
        for class_index in range(class_count-1):
            den += np.exp(parameters[(class_index)*len(x):(class_index + 1)*len(x)].T@x)

        for class_index in range(class_count-1):
            curr_prob = np.exp(parameters[(class_index)*len(x):(class_index + 1)*len(x)].T@x)/den

            if curr_prob > max_prob:
                pred_class = class_index + 1
                max_prob = curr_prob

        if 1/den > max_prob:
            pred_class = class_count
            max_prob = 1/den
        
        predictions[x_index] = pred_class

    return predictions



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
#    plt.show()

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

        # constructing the training data set
    train_data = pd.read_csv(train_file_path, header = None)
    
    data_fraction      = 1
    sample_names       = train_data.iloc[:int(data_fraction*len(train_data)), 0]
    X_train            = train_data.iloc[:int(data_fraction*len(train_data)), 2:]
    y_train            = train_data.iloc[:int(data_fraction*len(train_data)), 1]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train_un_normalized = np.copy(y_train)

    X_train = np.c_[ X_train, np.ones(len(X_train)) ]    

    test_data = pd.read_csv(test_file_path, header = None)
    sample_names_test = test_data.iloc[:, 0]
    X_test            = test_data.iloc[:, 1:]
    
    X_test = X_test.to_numpy()
    # adding a constant feature of 1 for the new column
    X_test = np.c_[ X_test, np.ones(len(X_test)) ]    

    
    y_pred = np.zeros(len(y_train))
    # Linear regression
    if section == 1:
        y_mean = 0
        y_std  = 1
        iter_count = 625
        eta = 0.001
        w_maxit, weight_vectors_maxit = gradient_descent_maxit(X_train, y_train, eta, iter_count)
        y_pred_test_un_normalized = predict(X_test, w_maxit, y_mean, y_std)

        df = pd.DataFrame()
        df['sample_names'] = sample_names_test
        df['scores']       = y_pred_test_un_normalized
        df.to_csv(os.path.join(output_file_path,'1.csv'), header = False, index = False)
        
    elif section == 2:
        y_mean = 0
        y_std  = 1
        lamda = 100
        iter_count = 500
        eta = 0.001
        w_maxit, weight_vectors_maxit = gradient_descent_maxit_ridge(X_train, y_train, eta, iter_count, lamda)
        y_pred_test_un_normalized = predict(X_test, w_maxit, y_mean, y_std)

        df = pd.DataFrame()
        df['sample_names'] = sample_names_test
        df['scores']       = y_pred_test_un_normalized
        df.to_csv(os.path.join(output_file_path,'2.csv'), header = False, index = False)

    elif section == 5:
        iter_count = 250
        class_count = 9
        eta = 0.01
        parameters_maxit, parameter_vectors_maxit = gradient_descent_multiclass_logit_maxit(X_train, y_train, eta, iter_count, class_count)

        y_test_pred = predict_logit(X_test, parameters_maxit, class_count)
    
        df = pd.DataFrame()
        df['sample_names'] = sample_names_test
        df['scores']       = y_test_pred
        df.to_csv(os.path.join(output_file_path,'5.csv'), header = False, index = False)
