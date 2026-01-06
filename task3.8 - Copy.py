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
            
        

def gradient(w, X, y):

    gradient_vector = np.zeros(len(X[0]))
    for x_index in range(len(X)):
        x = X[x_index]
        if y[x_index] == 1: 
            gradient_vector += -x/(1+np.exp(np.dot(w, x)))
        else:
            gradient_vector += x/(1+np.exp(-np.dot(w, x)))            
        
#           
    
            
    
    return (gradient_vector)/len(X)


def gradient_descent_multiclass_logit_maxit(X, y, eta, iter_count, class_count):
    parameters = np.zeros(len(X[0]))


    curr_parameters = []

    for iter in range(iter_count):

        curr_parameters.append(parameters)

        gradient_vector = gradient(parameters, X, y)

        parameters = parameters - (eta*gradient_vector/(np.sqrt(gradient_vector.T @ gradient_vector)))

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
        loss = 0
        for x_index in range(len(X)):
            x = X[x_index]
            class_label = y_true[x_index]

            den = 1

            for class_index in range(class_count-1):
                den += np.exp(class_params[(class_index)*len(x):(class_index + 1)*len(x)].T@x)

            if class_label != class_count:
                loss += np.log(np.exp(class_params[(class_label-1)*len(x):(class_label)*len(x)].T@x)/den)
            else:
                loss += np.log(1/den)

        losses.append(loss*(-1/len(X)))

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


def plot_losses_logit(point_count, logit_losses_train, logit_losses_val, x_label, y_label, title):
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
    plt.show()


def predict_one_vs_rest(X, class_parameters_list):

    y_pred = []
    for x_index in range(len(X)):
        confidence = []
        x = X[x_index]
        # getting P(y = Ck|x) for all classes
        for class_index in range(len(class_parameters_list)):
            current_parameters = class_parameters_list[class_index]


            confidence.append(1/(1+np.exp(-np.dot(x, current_parameters))))

        # take the argmax over the confidence values

#        confidence.append()
        max_prob       = confidence[0]
        max_prob_index = 1
        for i in range(len(confidence)):
            if confidence[i] > max_prob:
                max_prob       = confidence[i]
                max_prob_index = i

        y_pred.append(max_prob_index+1)

        
    return np.array(y_pred)
        
    
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
    
    # try logistic regression without normalizing the data


    eta = 0.01

    # gradient descent with fixed number of iterations
    # highest accuracy on val set for eta = 0.01 and iter_count = 110, accuracy = 0.42
    iter_count = 110
    class_count = 9

    class_parameters_list = []
    
    for class_no in range(1, class_count+1):
        
        y_train_relabelled = np.copy(y_train)

        for i in range(len(y_train)):
            if y_train[i] == class_no:
                y_train_relabelled[i] = 1
            else:
                y_train_relabelled[i] = 0  # 2 indicates classes other than desired class
            
        parameters_maxit_class, parameter_vectors_maxit = gradient_descent_multiclass_logit_maxit(X_train, y_train_relabelled, eta, iter_count, 2)
        class_parameters_list.append(parameters_maxit_class)

        

    y_train_pred = predict_one_vs_rest(X_train, class_parameters_list)


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


    y_val_pred = predict_one_vs_rest(X_train, class_parameters_list)
    print(y_val_pred)
    print(f"Accuracy on validation data:{accuracy(y_val, y_val_pred)}")

