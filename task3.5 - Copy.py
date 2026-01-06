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

# def normalize_y(y):
#     sum = 0
#     squared_sum = 0
#     for y_index in range(len(y)):
#         sum += y[y_index]
#         squared_sum += y[y_index]**2

#     mean = sum/len(y)
#     variance = squared_sum/len(y) - mean**2

#     std = np.sqrt(variance)

#     for y_index in range(len(y)):
#         y[y_index] = (y[y_index] - mean)/std

#     return (mean, std)

def normalize_val_test(X, feature_means, feature_stds):

    for feature_index in range(len(X[0])):
        for x_index in range(len(X)):
            X[x_index][feature_index] = (X[x_index][feature_index] - feature_means[feature_index])/feature_stds[feature_index]

    return

# def normalize_val_test_y(y, y_mean, y_std):

#     for y_index in range(len(y)):
#         y[y_index] = (y[y_index] - y_mean)/y_std


#     return
            
        

def gradient(class_params, X, y, class_count):

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

        gradient_vector = gradient(parameters, X, y, class_count)

        parameters = parameters - (eta*gradient_vector)

    return (parameters, curr_parameters)



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

def compute_iter_losses(X, weight_vectors, y_true_un_normalized, y_mean, y_std):

    losses = []

    for weight_index in range(len(weight_vectors)):
        weights = weight_vectors[weight_index]

        y_pred_un_normalized = predict(X, weights, y_mean, y_std)

        losses.append(np.sum(((y_true_un_normalized - y_pred_un_normalized)*(y_true_un_normalized - y_pred_un_normalized))/len(y_true_un_normalized)))


    losses = np.array(losses)

    return losses

def compute_iter_losses_logit(X, parameter_vectors, y_true, class_count):

    losses = []


    for class_params in parameter_vectors:
        y_predict = predict_logit(X, class_params, class_count)
        loss = np.dot((y_true - y_predict), (y_true - y_predict))

                # loss += np.log(np.exp(class_params[(class_label-1)*len(x):(class_label)*len(x)].T@x)/den)
                

                

        losses.append(loss*(1/len(X)))

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
    plt.savefig(os.path.join(directory, title + '.png'))

def plot_losses_logit(point_count, logit_losses_train, logit_losses_val, x_label, y_label, title, directory):
    gap_train = len(mse_losses_train)//point_count
    iterations_train = []
    logit_losses_train_sample = []

    for i in range(0, len(logit_losses_train), gap_train):
        logit_losses_train_sample.append(logit_losses_train[i])
        iterations_train.append(i/gap_train)

    plt.plot(iterations_train, logit_losses_train_sample, color = 'blue')


    gap_val = len(logit_losses_val)//point_count
    iterations_val = []
    logit_losses_val_sample = []

    for i in range(0, len(logit_losses_val), gap_val):
        logit_losses_val_sample.append(logit_losses_val[i])
        iterations_val.append(i/gap_val)

    plt.plot(iterations_val, logit_losses_val_sample, color = 'orange')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(["Train loss", "Validation loss"])
    # plt.show()
    plt.savefig(os.path.join(directory, title + '.png'))


    
    
def accuracy(y_true, y_pred):
    incorrect = 0
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
        else:
            incorrect += 1

    return correct/(correct+incorrect)
    
    
if __name__ == '__main__':

    # parsing the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-trp", "--train_path", help="Path to training data")
    parser.add_argument("-vap", "--val_path", help="Path to validation data")
    parser.add_argument("-tep", "--test_path", help="Path to test data")
    parser.add_argument("-oup", "--out_path", help="Path to file where output has to be written")
    parser.add_argument("-s", "--section", help="Section value", type = int)

    directory = 'Task_3_5'
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

    # adding a column for the constant term
    X_train = np.c_[ X_train, np.ones(len(X_train)) ]    
    y_train_un_normalized = np.copy(y_train)
#    print(X_train)
    # try logistic regression without normalizing the data


    eta = 0.01

    # gradient descent with fixed number of iterations
    # highest accuracy on val set for eta = 0.01 and iter_count = 50, accuracy = 0.5714
    iter_count = 250
    class_count = 9
    print(f"Using multi class logistic regression with iter_count:{iter_count} and eta:{eta}")

    parameters_maxit, parameter_vectors_maxit = gradient_descent_multiclass_logit_maxit(X_train, y_train, eta, iter_count, class_count)

    y_train_pred = predict_logit(X_train, parameters_maxit, class_count)

    print(f"Accuracy on training data:{accuracy(y_train, y_train_pred)}")

    # CHECKING ON VALIDATION SET
    # loading the validation data

    val_data = pd.read_csv(val_file_path, header = None)
    sample_names_val = train_data.iloc[:, 0]
    X_val            = val_data.iloc[:, 2:]
    y_val            = val_data.iloc[:, 1]



    
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()
    # adding a column for the constant term
    X_val = np.c_[ X_val, np.ones(len(X_val)) ]    

    y_val_pred = predict_logit(X_val, parameters_maxit, class_count)
    print(f"Accuracy on validation data:{accuracy(y_val, y_val_pred)}")


    # PLOTTING THE LEARNING CURVE

    iteration_losses_train = compute_iter_losses_logit(X_train, parameter_vectors_maxit, y_train, class_count)
    iteration_losses_val = compute_iter_losses_logit(X_val, parameter_vectors_maxit, y_val, class_count)
    # iteration_losses_val   = compute_iter_losses(X_val, weight_vectors_reltol, y_val_un_normalized, y_mean, y_std)
    print(iteration_losses_val[-1])
    plot_losses(20, iteration_losses_train, iteration_losses_val, "Iterations", "Logistic loss", f"Multi-class Logit, eta = {eta}, iterations = {iter_count}", directory)

