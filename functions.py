# Import reader module from built-in csv library
from csv import reader
from math import sqrt, exp, pi
# Set seed for reproducability and apples-to-apples comparisons of algorithms, i.e. ensure same result of randomization every time code is executed.
from random import seed
from random import randrange
#from prettytable import PrettyTable

# ******** LOAD CSV FILE *********
# Load a CSV file, skipping empty rows
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:   # locally scoped file
        csv_reader = reader(file)
        for row in csv_reader:  # looping over each row
            if not row: # i.e. if empty
                continue    # go to next iteration, i.e. skip ahead to next row
            dataset.append(row)
    print('Loaded data file {0} with {1} rows and {2} columns.'.format(filename, len(dataset), len(dataset[0]))) # Prints that load was successful
    return dataset  # returns list of lists, but all values are strings
# ********************************

# *** STRING TO FLOATING POINT ***
# Convert column to floating point values, preferred in ML algorithms
def str_column_to_float(dataset, column):
    for row in dataset: # in each row
        row[column] = float(row[column].strip())    # casts as float and strips out whitespace
# ********************************

# ****** STRING TO INTEGER *******
# Some algorithms prefer all values to be numeric, including the outcome or predicted value:
# Use set() to get the unique strings, and enumerate() to give each an index int
# Store in dictionary, and replace strings in dataset with integers
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset] # Populates the list class_values with the column-value at each row in dataset
    unique = set(class_values)  # gets the (class=) set of unique values in class_values
    lookup = dict() # create dictionary to map each item in the set of unique values
    for i, value in enumerate(unique):  # iterates over unique, adding an index
        # enumerate returns for each iteration a tuple containing an index and the corresponding value
        # since we enumerated a set (list of unique), we're adding keys to each unique value
        lookup[value] = i   # Map key = unique string "value", to value = index number i
    for row in dataset:
        # replace string keys in specified dataset column with (index) integers
        row[column] = lookup[row[column]]
    # return dictionary so downstream user can convert predictions back to strings
    return lookup
# ********************************

# **** NORMALIZING TO [0, 1] *****
# Scaling numbers to between 0 and 1, min = 0 and max = 1

# Calculating minimum and maximum value in each dataset column
def dataset_minmax(dataset):
    # Note: will error if a column is not numerical
    minmax = list()
    # iterating over each column in dataset
    for i in range (len(dataset[0])):   # requires that first row has no empty columns!
        col_values = [row[i] for row in dataset]    # define list with all rows' values in column i
        # Find min and max values in each column, and add to minmax list (without indexing => ordering important!)
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])   # append min and max as a list, i.e. minmax = list of lists
    return minmax

# Normalize dataset, i.e. rescale columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        # -1 since we don't want to normalize the output/class/label
        for i in range (len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
# ********************************

# ******** STANDARDIZING *********
# Centering the distribution of data on the value 0 and standard deviation to the value 1

# Calculate the mean of each dataset column
def column_means(dataset):
    # Inside bracket returns a series of 0's len(...) long, creating the list 'means'
    means = [0 for i in range(len(dataset[0]))] # populate list of means with 0
    # range() defaults is start = 0, end = x, step = 1; returns that sequence of numbers
    for i in range(len(dataset[0])):    # Iterate over each column number in first row
        # Get list of values for each column in all rows
        col_values = [row[i] for row in dataset]    # dataset is list of rows
        means[i] = sum(col_values) / float(len(col_values)) # update column mean; cast to avoid integer division
    return means    # list of column means, index = column number

# Calculate the standard deviation of each dataset column, assuming means already calculated
def column_stdev(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]    # populate list with 0's, length = #columns
    for i in range(len(dataset[0])):    # iterate over column numbers in first row
        variance = [pow(row[i] - means[i], 2) for row in dataset]   # squares of deviations
        stdevs[i] = sum(variance)   # sum of squares of deviations; x below
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]  # square each sum to get st.dev.
    return stdevs   # list of column standard deviations, index = column number

# Standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset: # Iterate over each row
        for i in range(len(row)):   # Iterate over each column
            row[i] = (row[i] - means[i]) / stdevs[i]
# ********************************

# ***** TRAIN AND TEST SPLIT *****
# Split a dataset into a train and test set, default 60/40 split
def train_test_split(dataset, split=0.60):
    train = list()
    # Number of rows training set requires from original dataset
    train_size = split * len(dataset)
    # Python passes by reference, so would otherwise change original dataset
    dataset_copy = list(dataset)
    # continue taking random elements until training set is defined length
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        # Pop returns one element (row, here) and removes it from the object
        # Takes a random row from dataset_copy and removes it from the pool
        train.append(dataset_copy.pop(index))
    # returns training set (0.6) and remainder of dataset_copy, i.e. testing set (0.4)
    return train, dataset_copy
# ********************************

# *** K-FOLD CROSS_VALIDATION ****
# Instead of two, divide k groups (folds) of data of equal size.
# For each group k, train the algorithm on the remaining k-1 groups, and test on k.
# Split dataset into k folds (3 by default)
def cross_validation_split(dataset, folds=3):
    # List of folds, i.e. each list object is a fold
    dataset_split = list()
    # Operate on a copy since dataset passed by reference, keeps original intact
    dataset_copy = list(dataset)
    # int division trims off excess rows that keep it from dividing into equal chunks
    fold_size = int(len(dataset)/folds)
    # Iterating over 0 to folds
    for i in range(folds):
        # Create list to hold rows in fold i
        fold = list()
        # Populate fold with rows until requisite size (determined by k and dataset)
        while len(fold) < fold_size:
            # Pick random index of the dataset row to add to fold i
            index = randrange(len(dataset_copy))
            # Add row to fold and remove from pool of possible subsequent rows
            fold.append(dataset_copy.pop(index))
        # Add populated fold to list of fold objects
        dataset_split.append(fold)
    # Return list of folds; each is itself a list of rows
    return dataset_split
# ********************************

# **** CLASSIFICATION ACCURACY ***
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100
# ********************************

# ******* CONFUSION MATRIX *******
# Construct confusion matrix
# Returns two objects: the set of unique actual values, and the matrix
# Matrix: 1st is actual values, 2nd is corresponding predictions values
def confusion_matrix(actual, predicted):
    # The "Set" (class, similar to list) of unique values in actual
    unique = set(actual)
    # matrix is a list of lists; one for each unique actual value
    matrix = [list() for x in range(len(unique))]
    # Default each cell to 0, then change later.
    for i in range(len(unique)):
        # Confusion matrix is always square
        matrix[i] = [0 for x in range(len(unique))]
    # Dictionary to index unique actual values
    lookup = dict()
    # Enumerate set of unique actuals
    for i, value in enumerate(unique):
        # Assign each unique actual value an index i (from enumeration)
        # Uses "value" as dict key, index i as dict's value
        lookup[value] = i
    # Iterate over all actual/prediction pairs
    for i in range(len(actual)):
        # Get actual value's index (i) from dictionary
        x = lookup[actual[i]]
        # Get predicted value's index (i) from dictionary
        y = lookup[predicted[i]]
        # Increment matrix cell count at index_1 = actual, index_2 = predicted
        matrix[x][y] += 1
    # Returns the set of unique values, and the matrix itself
    return unique, matrix

# Print human-readable confusion matrix, using PrettyTable
def print_confusion_matrix(unique, matrix):
    table = PrettyTable()
    # Set table headers
    table.field_names = ["A\P"] + [str(x) for x in unique]
    # Matrix: iterate over unique actual values -> for each, get counts of unique prediction values
    for i, value in enumerate(unique):
        # Matrix[i][j] = count of prediction j for actual i, e.g. nrs. of "No" and "Yes" when actual is "Yes"
        row = [str(value)] + [str(count) for count in matrix[i]]
        table.add_row(row)
    print(table)
# ********************************

# ****** MEAN ABSOLUTE ERROR *****
# Calculate MAE
def mae_metric(actual, predicted):
    sum_error = 0.0
    # iterate over all the values
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    # return MAE, float conversion to avoid integer division
    return sum_error / float(len(actual))
# ********************************

# **** ROOT MEAN SQUARED ERROR ***
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        #prediction_error = predicted[i] - actual[i]
        sum_error += ((predicted[i] - actual[i])**2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)
# ********************************



# ** RANDOM PREDICTION ALGORITHM *
# Generate random predictions
def random_algorithm(train, test):
    # Store output values in training set. Assumes the final column [-1] is the output
    output_values = [row[-1] for row in train]
    # Set-object's constructor gets unique values, and is then converted to a list
    unique = list(set(output_values))
    # List of algo prediction, index in list = test dataset row number
    predicted = list()
    # For each row in test set, select random output value as prediction
    for row in test:
        # Picks a random output value: probability independent of distribution!!!
        index = randrange(len(unique))
        # Set the randomly selected output value as the prediction for that row
        predicted.append(unique[index])
    return predicted
# ********************************

# ****** ZERO RULE ALGORITHM *****
# Zero Rule for Classification models: for each testing row, predict the most common training output
def zero_rule_algorithm_classification(train, test):
    # Assuming output is final column in dataset, retrieve all outputs in training set
    output_values = [row[-1] for row in train]
    # Find most common (max of counts of each distinct value) output value in training set
    # If multiple predictions share the max count, the first observed in the set is returned
    prediction = max(set(output_values), key=output_values.count)
    # Predict the same value (most common in training set) for every row in testing set
    predicted = [prediction for i in range(len(test))]
    return predicted

# Zero Rule for Regression: for each testing row, predict the mean of training outputs
def zero_rule_algorithm_regression(train, test):
    # Assuming output is final column in dataset
    output_values = [row[-1] for row in train]
    # Calculates mean, used as our prediction
    prediction = sum(output_values) / float(len(output_values))
    # Predict the same value (mean of training outputs) for every row in testing set
    predicted = [prediction for i in range(len(test))]
    return predicted
# ********************************



# *** TRAIN-TEST SPLIT HARNESS ***
# algorithm is the algorithm to test, in our testing we use baseline algorithms
# *args is for any additional configuration parameters required by the above algorithm
# Requires train_test_split and accuracy_metric to be defined
def split_evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        # Assuming last column is output, clear outputs to avoid accidental cheating by baseline prediction
        row_copy[-1] = None
        test_set.append(row_copy)
    # Get baseline prediction using specified algorithm
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    # Assumes it's a classification problem, should make accuracy algorithm a parameter
    accuracy = accuracy_metric(actual, predicted)
    return accuracy
# ********************************

# *** CROSS-VALIDATION HARNESS ***
def cross_evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    # List to store accuracy scores of each fold k used as test set
    scores = list()
    # For each fold:
    for fold in folds:
        # Use all folds except the current as training data
        train_set = list(folds) # At this point, a list of folds where each fold is also a list
        # Remove the current fold from training set, instead using it as the test set
        train_set.remove(fold)
        # Flattens the list of lists into one big list containing all the rows (= appends folds together)
        train_set = sum(train_set, [])
        # Initialize empty list to hold test data
        test_set = list()
        # Set current fold as test set, and nullify the output to avoid accidental cheating
        for row in fold:
            # row_copy = row   would reference the same object, instead of copying it
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        # Predict based on training with all other folds than the current fold, and specified algorithm
        predicted = algorithm(train_set, test_set, *args)
        # Set current fold outputs as test set
        actual = [row[-1] for row in fold]
        # Rate accuracy by comparing prediction to test set (i.e. current fold)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
# ********************************

# ** DYNAMIC EVALUATION HARNESS **
# folds_or_split is number of folds or train/test split. Train/test used if <= 1.0, otherwise cross-validation
# default metric to evaluate is accuracy, hence assumes classification unless specified otherwise
# algorithm is the ML algorithm to test
# Metric accuracy indirectly determines whether it's a classification or regression algorithm
def evaluate_algorithm(dataset, algorithm, folds_or_split = 0.6, metric = accuracy_metric, *args):
    scores = list()
    folds = list()

    if folds_or_split <= 1.0:
        # Then train/test split
        train, test = train_test_split(dataset, folds_or_split)
        # Append test set first, since that will be the first to be iterated over
        folds.append(test)
        folds.append(train)
    else:
        # Then cross-validation
        folds = cross_validation_split(dataset, folds_or_split)

    for fold in folds:
        # Training set is all folds but the current
        train_set = list(folds)
        train_set.remove(fold)  # Remove current fold, since that is the test set
        train_set = sum(train_set, [])  # Flatten training data into one set/list

        # Nullify output values in test set to avoid accidental cheating
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        # Predict values of test set based on specified algorithm
        predicted = algorithm(train_set, test_set, *args)
        # Observed (real) data
        actual = [row[-1] for row in fold]

        # Calculate accuracy based on specified metric; depends on if classification or regression
        accuracy = metric(actual, predicted)

        # If train/test split, stop after first iteration since only one fold (i.e. the test set)
        if folds_or_split <= 1.0:
            # Return accuracy and exit function - breaking loop before reverse combination of "folds" is used
            return accuracy
        # If not, continue the iteration
        else:
            # Store score of this fold: will error if metric returns more than one value!
            scores.append(accuracy)
            continue

    return scores
# ********************************


# *** SIMPLE LINEAR REGRESSION ***
# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))

# Calculate variance of a lsit of numbers, provided mean
def variance(values, mean):
    return sum([(x - mean)**2 for x in values])

# Calculate covariance between x and y
def covariance(x, x_mean, y, y_mean):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (y[i] - y_mean)
    return covar

# Calculate coefficients
def coefficients(dataset):
    # Assumes x is first column, y second - creating a list for each
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    # Calculate simple linear regression coefficients
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    # Return list of coefficients
    return [b0, b1]

# Simple linear regression, assuming train/test split
def simple_linear_regression(train, test):
    predictions = list()
    # Calculate regression coefficients
    b0, b1 = coefficients(train)
    # Test data used only to count iterations; parameter can be removed
    # Include anwyway since most other algos require both test and train sets
    for row in test:
        # For each observation, predict y^ given x and coefficients
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions
# **********************************

# * MULTIVARIATE LINEAR REGRESSION *
# Make a prediction with provided coefficients
# Returns yhat, i.e. predicted y-value for the provided row
def predict_linear(row, coefficients):
    # First coefficient in is always the intercept B0 ("bias")
    yhat = coefficients[0]
    # Iterate over remaining columns in row, multiplying Xi * Bi
    for i in range(len(row)-1):
        # Sum for prediction
        yhat += coefficients[i + 1] * row[i]
    return yhat

# Estimate linear regression coefficients using stochastic gradient descent
# Returns list of coefficients, with intercept at first index
# Batch size b = 1 means stochastic gradient descent, b > 1 means (mini) batch gradient descent
def coefficients_linear_sgd(train, l_rate, n_epoch, b = 1):
    # Start with empty coefficients
    coef = [0.0 for col in range(len(train[0]))]
    # Iterate over epochs, performing SGD based on batch size for each
    for epoch in range(n_epoch):
        # Sum total error to track overall epoch performance
        sum_error = 0
        # Counter to track mini batch
        i = 0
        # Counter to track overall row, to capture mid-batch end of dataset
        j = 0
        # SUM(h - y) * Xi over coefficients i, to multiply full batch by learning rate
        adjustments = [0 for col in range(len(train[0]))]
        # Iterate over all rows of data
        for row in train:
            # Increment batch index and dataset counters
            i += 1
            j += 1
            # Predict y for this row with latest coefficients
            yhat = predict_linear(row, coef)
            # Find prediction error, assuming actual y is final column; i.e. (h - y)
            error = yhat - row[-1]
            # Squaring for absolute error, to track epoch performance
            sum_error += error**2
            # Add intercept (bias) error, since has no Xi
            adjustments[0] += error
            # Iterate over all coefficients, except the first (i.e. intercept)
            for k in range(len(row)-1):
                # Add to batch adjustment, i.e. (h - y) * Xi
                # row column j belongs to coefficient j + 1, since j = 1 is intercept
                adjustments[k + 1] += error * row[k]
            # Check whether this row is last in batch, or in dataset
            if i == b or j == len(train):
                # Iterate over all coefficients
                for k in range(len(adjustments)):
                    # Adjust coefficient based on batch sum gradient and learning rate
                    # i = b except if at end of dataset and in mid-batch
                    # "Averaging" adjustment of each row in mini batch
                    coef[k] = coef[k] - (1/i) * l_rate * adjustments[k]
                # Reset adjustments to 0 for next batch
                adjustments = [0 for col in range(len(train[0]))]
                # Reset batch counter
                i = 0

            # # Adjust intercept (bias) based on learning rate and prediction error
            # coef[0] = coef[0] - l_rate * error
            # # Iterate over all columns, except the final column = y
            # for i in range(len(row)-1):
            #     # Adjust variable slope Bi based on learning rate, error and Xi
            #     # i = 1 is the intercept, so exclude since has no Xi
            #     # First column belongs to second coefficient, since first is intercept
            #     coef[i + 1] = coef[i + 1] - l_rate * error * row[i]

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, l_rate, sum_error))
    # Return the final set of coefficients
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
# Returns the list of predictions for each dataset row
# Parameters: learning rate, number of epochs (iterations), mini-batch size b
def linear_regression_sgd(train, test, l_rate, n_epoch, b_size = 1):
    predictions = list()
    # Get coefficients based on SGD, learning rate and epochs
    coef = coefficients_linear_sgd(train, l_rate, n_epoch, b_size)
    # Get predictions for provided dataset using estimated coefficients
    for row in test:
        yhat = predict_linear(row, coef)
        predictions.append(yhat)
    return predictions
# **********************************

# ****** LOGISTIC REGRESSION *******
# Make a logistic prediction, provided row and coefficients
def logistic_predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent (SGD)
# Returns list of coefficients, with intercept at first index
# Batch size b = 1 means stochastic gradient descent, b > 1 means (mini) batch gradient descent
def logistic_coefficients_sgd(train, l_rate, n_epoch, b = 1):
    #coef = [0.0 for row in train]
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        # Sum squared errors to track epoch performance
        sum_error = 0
        # Counter to track mini batch
        i = 0
        # Counter to track overall row, to capture mid-batch end of dataset
        j = 0
        # SUM(h - y) * Xi over coefficients i, to multiply full batch by learning rate
        adjustments = [0 for col in range(len(train[0]))]
        for row in train:
            i += 1
            j += 1
            yhat = logistic_predict(row, coef)
            error = row[-1] - yhat
            # Add to overall epoch error
            sum_error += error**2
            # Add intercept error, since has no corresponding input Xi
            adjustments[0] += error
            for k in range(len(row)-1):
                # Begin with first coefficient, skipping intercept at k = 0
                adjustments[k + 1] += error * row[k]

            # Check whether this row is last in batch, or in overall dataset
            if i == b or j == len(train):
                # Iterate over all coefficients, including intercept
                for k in range(len(adjustments)):
                    coef[k] = coef[k] + l_rate * (1/i) * adjustments[k] * yhat * (1.0 - yhat)

                adjustments = [0 for col in range(len(train[0]))]
                i = 0

        # for row in train:
        #     yhat = predict_logistic(row, coef)
        #     error = row[-1] - yhat
        #     # Squared sum error to track and print epoch performance
        #     sum_error += error**2
        #     coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
        #     # Iterate over all but first coefficient (intercept) / last column (real y)
        #     # Avoid both at the same time by iterating up to len(row) - 1
        #     for i in range(len(row)-1):
        #         coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, l_rate, sum_error))
    return coef

# Logistic Regression Algorithm with Stochastic Gradient Descent
# Parameters: learning rate, number of epochs, (mini-) batch size
def logistic_regression_sgd(train, test, l_rate, n_epoch, b_size = 1):
    # Empty list to hold predictions matching test actual y's
    predictions = list()
    # Estimate coefficients using provided parameters and SGD
    coef = logistic_coefficients_sgd(train, l_rate, n_epoch, b_size)
    for row in test:
        # Predict yhat using estimated coefficients
        yhat = logistic_predict(row, coef)
        # Round to 0 or 1
        yhat = round(yhat)
        # Add integer yhat to list of predictions
        predictions.append(yhat)
    return predictions
# **********************************

# *********** PERCEPTRON ***********
# Make a prediction with weights based on perceptron (single neuron)
def perceptron_predict(row, weights):
    activation = weights[0]
    # Loop over all input columns except last (i.e. actual output)
    for i in range(len(row)-1):
        # For each column, multiply input by coefficient and update activation
        activation += weights[i + 1] * row[i]
    # Neuron fires if activation function >= 0, does not fire if < 0
    return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using Stochastic Gradient Descent (SGD)
# Inputs: learning rate, number of epochs, (mini) batch size with default 1
# Outputs: Perceptron weights
def perceptron_train_weights(train, l_rate, n_epoch, b = 1):
    # Initialize all weights to 0.0, including intercept (bias) at w0
    weights = [0.0 for col in range(len(train[0]))]
    for epoch in range(n_epoch):
        # Overall epoch squared error
        sum_error = 0.0
        # Counters to track batch index and overall train set index
        i, j = 0, 0
        # Average SUM(h - y) * xi over weights i, to multiply batch by learning rate
        adjustments = [0.0 for col in range(len(train[0]))]

        for row in train:
            # Increment batch and train set indices
            i += 1
            j += 1
            # Calculate prediction and error against actual based on latest weights
            prediction = perceptron_predict(row, weights)
            error = row[-1] - prediction
            # Square and add to overall epoch error
            sum_error += error**2

            # Add to batch weight adjustments based on corresponding inputs (except bias)
            adjustments[0] += error
            for k in range(len(row)-1):
                # Begin with first weight, skipping intercept at k = 0
                # First column in row corresponds to second weight, since first is intercept
                adjustments[k + 1] += error * row[k]

            # Adjust all weights if row is last in batch or overall train set
            if i == b or j == len(train):
                #print('Adjustments made at row # ', str(j))
                for k in range(len(adjustments)):
                    weights[k] = weights[k] + l_rate * (1/i) * adjustments[k]
                    # i = b when at end of batch, but i < b if train set ends mid-match
                # Reset batch adjustments and batch index counter
                adjustments = [0.0 for col in range(len(train[0]))]
                i = 0

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    # Return weighs after iterating through all epochs -- and rows within each
    return weights

# Perceptron Algorithm with Stochastic Gradient Descent
# Input: train and test sets, learning rate, number of epochs, and batch size
# Output: list of predictions to compare against test data actual outputs
def perceptron(train, test, l_rate, n_epoch, b_size = 1):
    predictions = list()
    # Get Perceptron weights using SGD and input parameters
    weights = perceptron_train_weights(train, l_rate, n_epoch, b_size)
    # Predict output for all test rows, using Perceptron model built on train set
    for row in test:
        # Predict output based on test set inputs and weights estimated using train set
        prediction = perceptron_predict(row, weights)
        predictions.append(prediction)
    # Return predicted outputs in test set using train set weight estimates
    return predictions
# **********************************

# ************* CART ***************
# Calculate the Gini index for a split dataset
# Inputs: list of group dataset rows after node split, list of distinct class values
# Outputs: Gini index, number between 0.0 and 1.0
def cart_gini_index(groups, class_values):
    gini = 0.0
    # Iterate over known class values, to calculate proportion of each in both split groups
    for class_value in class_values:
        # For each known class value, iterate over each group to count occurences
        for group in groups:
            # Check that the group is not empty to avoid zero division
            size = len(group)
            if size == 0:
                continue
            # [...] builds list of classes (observations, not distinct!) in current group
            # Count occurrences of current class in list, and divides by group count of rows
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            # Increments Gini index
            gini += (proportion * (1.0 - proportion))
    return gini

# Split (2-group) a dataset based on an attribute and an attribute value
# Inputs: dataset (or subset), index of column to split by, and column split value
# Outputs: left and right lists of groups of rows (either or both can be empty)
# Note: group 1 if < split value, group 2 if >= split value
def cart_test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            # If row value < threshold, add row to left group
            left.append(row)
        else:
            # If row value >= threshold, add row to right group
            right.append(row)
    return left, right

# NOTE: INEFFICIENT IN THAT IT DOESN'T IGNORE DUPLICATE COLUMN VALUES!
# Determine best split of provided dataset, giving dictionary with best possible split:
# Returns column index to split with, split value, Gini, and tuple with left/right lists
# Minimizes Gini by testing splitting dataset by each column, and each column value
def cart_get_split(dataset):
    # Get distinct class values, assuming class is last column. Converts set to list.
    class_values = list(set(row[-1] for row in dataset))
    # Placeholder best column index to split by, split value, Gini, and resulting groups
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # Iterating over columns; -1 since last column is the class (output)
    for index in range(len(dataset[0])-1):
        # Iterating over each row (i.e. column value) to test Gini of column + split value
        for row in dataset:
            # Splits groups using current column and current row as split value
            groups = cart_test_split(index, row[index], dataset)
            # Gets Gini Index when ussing current row value as split value
            gini = cart_gini_index(groups, class_values)
            #print('X%d < %.3f Gini = %.3f' % (index+1, row[index], gini))
            # If current Gini is best up to this point, store the split configuration
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    # Return dictionary of best column to split by, split value, and resulting groups tuple
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Create terminal node prediction value
# Input: group (list) of rows assigned to the terminal node
# Output: predicted class, based on most common class value in group
def cart_to_terminal(group):
    # Extracts class values in group, assuming class is last column
    outcomes = [row[-1] for row in group]
    # Return highest frequency class value
    return max(set(outcomes), key=outcomes.count)

# Recursively create child splits for a node, or make a terminal node
# Input: pre-split node, max tree depth, minimum rows per node, node's tree depth
# Output: None, i.e. void function
def cart_split(node, max_depth, min_size, depth):
    # Extract left and right lists of group rows from the supplied node (dictionary)
    left, right = node['groups']
    # Deletes groups of data from parent node, as it no longer needs access to the data
    del(node['groups'])
    # Checks whether left or right list empty, i.e. whether a no split (100% in one group)
    if not left or not right:
        # Make the only child a terminal node , and set 'left' and 'right' to point to it
        node['left'] = node['right'] = cart_to_terminal(left + right)
        # Exit current iteration, since terminal child node has no child nodes of its own
        return
    # Check whether supplied node is at or above maximum tree depth
    if depth >= max_depth:
        # Set left and right child nodes to terminal nodes
        node['left'], node['right'] = cart_to_terminal(left), cart_to_terminal(right)
        # Exit current iteration, i.e. halting progression down this branch
        return

    # If we reach this point, we neither have a no split, nor have reached max depth
    # Process left child: if shorter than minimum row size, make it terminal
    if len(left) <= min_size:
        node['left'] = cart_to_terminal(left)
    # Neither too deep nor too small, so split left child node to two child nodes
    else:
        # Split left child node
        node['left'] = cart_get_split(left)
        # Recursively call function on the split left child node in a depth first fashion
        cart_split(node['left'], max_depth, min_size, depth+1)

    # Process right child: if shorter than minimum, make it a terminal node
    if len(right) <= min_size:
        node['right'] = cart_to_terminal(right)
    # If not, split right child node and make a recursive function call
    else:
        # Split right child node
        node['right'] = cart_get_split(right)
        # Make recursive call on the split right child
        cart_split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def cart_build_tree(train, max_depth, min_size):
    # Split the root node into two child nodes
    root = cart_get_split(train)
    # Call recursive function to add left nodes then right nodes in a depth first fashion
    cart_split(root, max_depth, min_size, 1)
    # Return root node; now just a dictionary with two child node references
    # Similarly, its child nodes are only references, until terminal nodes which contain rows
    return root

# Print a decision tree, stepping recursively
def print_tree(node, depth=0):
    # If node is a dictionary, i.e. has column index, split value and tuple of groups
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth*' ', (node['index']+1), node['value']))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    # Else node is a terminal node, i.e. contains only class prediction at that node
    else:
        # Multiplies spaces for node indendation
        # Printing "node" is here a list with a single value, i.e. the class prediction
        print('%s[%s]' % (depth*' ', node))

# Recursively make a prediction with a decision tree, given starting node and row of data
def cart_predict(node, row):
    # Determine direction to traverse, using row data and node split column + value
    if row[node['index']] < node['value']:
        # Traversing left, so check whether left child is full node or terminal node
        if isinstance(node['left'], dict):
            # Dictionary, so fully defined node, hence make a recursive call to the left
            return cart_predict(node['left'], row)
        else:
            # Terminal node, so return that node - i.e. the single-value prediction
            return node['left']
    else:
        # Row value >= split value, so traversing right, hence check whether terminal
        if isinstance(node['right'], dict):
            # Fully defined node, so make a recursive call on the right child node
            return cart_predict(node['right'], row)
        else:
            # Terminal node, so return the predicted class
            return node['right']

# Classification and Regression Tree Algorithm
# Train and test are passed by evaluation harness directly, depth and size by *args
# Input: train and test sets, maximum tree depth, minimum rows per non-terminal node
def decision_tree(train, test, max_depth, min_size):
    # 'tree' is the root node of the tree after it is built
    tree = cart_build_tree(train, max_depth, min_size)
    predictions = list()
    # Iterating over test set, running model built on train set and storing predictions
    for row in test:
        prediction = cart_predict(tree, row)
        predictions.append(prediction)
    # Returns the list of prediction for each row in the test set
    return predictions
# **********************************

# ********** Naive Bayes ***********
# Split the dataset by class values
# Output: dictionary with distinct class values as keys, lists of corresponding records (full rows) as values
def bayes_separate_by_class(dataset):
    separated = dict()
    # Iterate over all rows in the dataset
    for i in range(len(dataset)):
        # Store row
        vector = dataset[i]
        # Extract class value (output), assuming last column
        class_value = vector[-1]
        # Check if first occurence of class value
        if class_value not in separated:
            # Add the new class value as a key in the dictionary
            separated[class_value] = list()
        # Append current record (row of data) to the corresponding list (dictionary value)
        separated[class_value].append(vector)
    return separated

# # Calculate the mean of a list of numbers
# def mean(numbers):
#     return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    # Makes numbers a list, calculating (x - avg)^2 and summing them, then dividing by count
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers)-1)
    # Return the standard deviation
    return sqrt(variance)

# Calculate the mean, stdev, and count for each input column in a dataset
# Output: list of tuples with input columns' summary statistics; mean, st. dev., count
def bayes_summarize_dataset(dataset):
    # The * operator separates the dataset (a list of lists) into separate lists for each row
    # zip() iterates over each element of each row, returns each column as a tuple of numbers
    # For each column in the zip result set, mean, stdev and length are calculated
    # These summary statistics are stored as a list of tuples - one per input column
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    # Remove summary of output class variable, as that is not needed
    del(summaries[-1])
    # Returns the list of tuples (one per input variable) of summary statistics
    return summaries

# Split dataset by class then get summary statistics for each input variable in each set
# Output: dictionary; key = class, value = list[] summary statistic tuples() - per column
def bayes_summarize_by_class(dataset):
    separated = bayes_separate_by_class(dataset)
    summaries = dict()
    # Iterating over key, value where value = list of rows in dataset of that class (key)
    for class_value, rows in separated.items():
        # key = class, value = list of summary statistic tuples for input variables
        # i.e. input variable summary statistics "grouped by" that class (key)
        summaries[class_value] = bayes_summarize_dataset(rows)
    return summaries

# Calculates the Gaussian probability distribution function for x
# Input: x (for which to calculate probability), mean of x, standard deviation of x
# Output: the probability of x, given its mean and st. dev.
def gauss_calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean)**2 / (2 * stdev **2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
# Input: set of prepared summaries (dictionary where value = list of tuples), a new test row
# Output: a dictionary of probabilities with one entry for each class, given the input row
def bayes_calculate_class_probabilities(summaries, row):
    # Counts total rows; per class, [0] is first tuple (input variable), [2] is count rows
    # We use first input variable [0], but they should all have same length in class subset
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    # Iterating over classes (keys) and input variable summaries in that class subset
    for class_value, class_summaries in summaries.items():
        # P(Class = class_value) = count(class_value) / total_rows
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        # Length of dictionary = number of keys = number of input variables
        for i in range(len(class_summaries)):
            # Get summary statistics for current input variable in current class
            mean, stdev, count = class_summaries[i]
            # P(class_value|X1,X2) = P(X1|class_value) * P(X2|class_value) * P(class_value)
            probabilities[class_value] *= gauss_calculate_probability(row[i], mean, stdev)
    return probabilities

# Calculate probabilities of a new row belonging to each class, and use highest to predict
def bayes_predict(summaries, row):
    # Calculate probability of the provided row belonging to each class
    probabilities = bayes_calculate_class_probabilities(summaries, row)
    # Best prediction of class value, and the corresponding probability
    best_label, best_prob = None, -1
    # Iterating over the different classes (keys), and the probability (value) of each
    for class_value, probability in probabilities.items():
        # If first iteration or current probability is the best so far
        if best_label is None or probability > best_prob:
            # Update prediction and corresponding probability
            best_label = class_value
            best_prob = probability
    # Return prediction
    return best_label

# Manages the application of the Naive Bayes algorithm
# Input: train and test data sets
# Output: list of predictions corresponding to test set
def naive_bayes(train, test):
    # Get summary statistics for each input variable, in each class subset of data
    summaries = bayes_summarize_by_class(train)
    predictions = list()
    # Iterate over test set rows, storing the prediction on each row of input values
    for row in test:
        predictions.append(bayes_predict(summaries, row))
    # Return the list of predictions corresponding to the test set
    return predictions
# **********************************

# ****** K Nearest Neighbours ******
# Calculate the Euclidian distance between two vectors
# Input: two rows of data, where the last vector element is assumed to be the output
# Output: Euclidian distance, i.e. squared differences element-wise (per input variable)
# Dependencies: NA
def euclidian_distance(row1, row2):
    distance = 0.0
    # Assuming last row value is output (class or regression value)
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Find the K nearest neighbours to a given row from a given (training) dataset
# Input: training dataset, row for which to find neighbours, number of neighbours to find
# Output: list of num_neighbours observations from train that are most similar to test_row
# Dependencies: euclidian_distance
def knn_get_neighbours(train, test_row, num_neighbours):
    # List of tuples, each tuple corresponds to an observation
    # Each tuple element thus contains the full row (vector) and its distance to test_row
    distances = list()
    for train_row in train:
        dist = euclidian_distance(test_row, train_row)
        # Add tuple of observation vector and distance to test_row to list of tuples
        distances.append((train_row, dist))
    # Sort the list of neighbours by distance (index 1), by default in ascending order
    distances.sort(key=lambda tup: tup[1])
    # Add the K closest rows to a list of complete data observations
    neighbours = list()
    for i in range(num_neighbours):
        # Add neighbour index 0, which is the train_row of data
        neighbours.append(distances[i][0])
    return neighbours

# Get the predicted class for a given row (vector), based on k neighbours from train set
# Input: test_row for which to predict class, based on num_neighbours from train set
# Output: class prediction for given row
# Dependencies: get_neighbours, euclidian_distance
def knn_predict_classification(train, test_row, num_neighbours):
    # Get num_neighbours nearest neighbours to test_row
    neighbours = knn_get_neighbours(train, test_row, num_neighbours)
    # Add output classes of K nearest neighbours to a list
    output_values = [row[-1] for row in neighbours]
    # Find distinct classes (using set), and find class with maximum count (occurrences)
    prediction = max(set(output_values), key = output_values.count)
    return prediction

# Get the predicted regression value for a given row, based on k neighbours from train set
# Input: row for which to predict regression value, based on num_neighbours from train set
# Output: predicted regression value; the average value of that of its K neighbours
# Dependencies: get_neighbours, euclidian_distance
def knn_predict_regression(train, test_row, num_neighbours):
    neighbours = knn_get_neighbours(train, test_row, num_neighbours)
    output_values = [row[-1] for row in neighbours]
    prediction = sum(output_values) / float(len(output_values))
    return prediction

# For a test set, get predictions using "training" data and problem type
# Input: train (neighbours), test, number of neighbours (K), classification or regression
# Output: predicted class or regression value for provided test observations
def k_nearest_neighbours(train, test, num_neighbours, type = "classification"):
    # List of predictions, equal length as rows in test set
    predictions = list()
    # There is strictly speaking no algorithm training step in KNN;
        # the K neighbours are found from among the training set
    # Get prediction based on K neighbours (from train) for each row in test set
    if(type == "regression"):
        for row in test:
            output = knn_predict_regression(train, row, num_neighbours)
            predictions.append(output)
    elif(type == "classification"):
        for row in test:
            output = knn_predict_classification(train, row, num_neighbours)
            predictions.append(output)
    else:
        print("Error: Invalid problem type.")
        return None

    return predictions
# **********************************

# ** Learning Vector Quantization **
# Locate the best matching unit
# Input: test_row to find BMU for, and codebooks (patterns) among which to look
# Output: the BMU; the codebook (pattern) with shortest Euclidian distance from test_row
def get_best_matching_unit(codebooks, test_row):
    # List of tuples; each contains a vector and its distance to test_row
    distances = list()
    # Iterate over codebooks (patterns to match against) to find closest match
    for codebook in codebooks:
        # Find Euclidian distance between test_row and current codebook (pattern)
        dist = euclidian_distance(codebook, test_row)
        # Add tuple of codebook (pattern) and its distance to test_row to list of tuples
        distances.append((codebook, dist))
    # Sort list of tuples by distance (second tuple element: tup[1]), by default in ascending order
    distances.sort(key=lambda tup: tup[1])
    # Return codebook (pattern) with shortest Euclidian distance from test_row
    return distances[0][0]

# Create a randomized codebook vector
# Input: the training set from which to pull random feature values
# Output: a codebook vector with randomized feature pattern values from among the training set
def random_codebook(train):
    n_records = len(train)
    # Number of features /  input variables / columns
    n_features = len(train[0])
    # Initialize codebook as a vector (set) of random features from training data
    # For each feature, take that of a random record in the training set
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook

# Make a prediction with codebook vectors
# Input: codebook vectors ("blueprint" class patterns), test row to predict for
# Output: predicted class for test row
def predict_LVQ(codebooks, test_row):
    # Find best matching unit, i.e. codebook vector whose pattern has least Euclidian distance
    bmu = get_best_matching_unit(codebooks, test_row)
    # Return the BMU class as the prediction
    return bmu[-1]

# Train a number of codebook vector feature values for a provided dataset, with learning rate decay
# Input: training data, number of codebooks (patterns), initial learning rate, number of epochs
# Output: list of tuned codebook vectors (practically class blueprint patterns of input data)
def train_codebooks(train, n_codebooks, lrate, epochs):
    # Populate a list[] of randomly initialized codebook vectors ("class" input data patterns)
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        # Update effective learning rate through epoch-based decay
        rate = lrate * (1.0 - (epoch / float(epochs)))
        # Start epoch error rate at 0
        sum_error = 0.0
        # Per epoch, iterate over all rows in train
        for row in train:
            # Find BMU of row from among codebook vectors
            bmu = get_best_matching_unit(codebooks, row)
            # Iterate over features to update codebook vector (i.e. "blueprint" class pattern)
            for i in range(len(row)-1):
                # Calculate feature value difference between BMU and current training row
                error = row[i] - bmu[i]
                # Add to epoch sum squared error
                sum_error += error**2
                # If last column of BMU same as train row, hence same (correct) class
                if bmu[-1] == row[-1]:
                    # Increment BMU feature value to bring it closer to train row feature value
                    # E.g. row[i] > bmu[i], then adding rate*error increases bmu[i] closer to row[i]
                    bmu[i] += rate * error
                # BMU is a different class, hence wrong prediction
                else:
                    # Decrement BMU feature value, i.e. moving it further from the test row pattern
                    bmu[i] -= rate * error
        print(">epoch%d, lrate=%.3f, error=%.3f" % (epoch, rate, sum_error))
    return codebooks

# Learning Vector Quantization Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    # Train codebook vector feature values
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        # Get predicted class based for given row based on trained codebook vectors
        output = predict_LVQ(codebooks, row)
        predictions.append(output)
    return predictions
# **********************************

# *** CART Classification Bagging **
# **********************************
# Create a random subsample from the dataset with replacement
# Input: dataset to bootstrap, ratio of dataset to use for bootstrap sample (e.g. 70% = 0.7)
# Output: bootstrapped dataset sample
def bootstrap_subsample(dataset, ratio = 1.0):
    # Create empty list for new bootstrapped sample
    sample = list()
    # Get number of observations in bootstrapped sample
    n_sample = round(len(dataset) * ratio)
    # Randomly add observations from dataset to bootstrap sample, with replacement
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# For a set of bagged trees, makes a prediction with each and combines into a single return prediction
# Input: row to predict output, list of bagged trees with whom to get prediction
# Output: predicted class; most common prediction among the bagged trees
def bagging_predict_mode(trees, row):
    # For given row, get prediction of each bagged tree in 'trees' and add to list of tree predictions
    predictions = [cart_predict(tree, row) for tree in trees]
    # Select most common prediction from those made by the bagged trees, and return as prediction
    return max(set(predictions), key=predictions.count)

# Creates bootstrapped samples, trains a decision tree on each, then predicts using the bagged trees
# Input: train and test sets, max tree depth, min rows per branch, sample size ratio, num trees/samples
# Output: list of predictions for provided test rows, using provided train rows to bootstrap aggregate
def cart_bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = bootstrap_subsample(train, sample_size)
        tree = cart_build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict_mode(trees, row) for row in test]
    return(predictions)
# **********************************

# ********** Random Forest *********
# Select the best Random Forest splits for a dataset (subsample), with constrained feature candidates
# Input: dataset (subsample, like bagging), number of input features to (randomly) evaluate as splits
# Output: dictionary of optimal split point with feature index, split value,
def rf_get_split(dataset, n_features):
    # Get unique output values by looking through last dataset column
    class_values = list(set(row[-1] for row in dataset))
    # Initialize best split feature index, split value, gini score, resulting split groups of records
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    # Randomly select feature indices to test as splits, and add to list of candidates
        # I.e. same as creating subsamples for bagging, but for input features
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        # Without replacement, e.g. if 8 features but n_features =4, get 4 random but unique indices
        if index not in features:
            features.append(index)
    # Iterate over each randomly selected candidate feature
    for index in features:
        # Iterate over all rows in dataset, to try every possible feature value as split value
        for row in dataset:
            # Split entire dataset (subsample) using current feature and current feature value as split
            groups = cart_test_split(index, row[index], dataset)
            # Calculate Gini index of resulting groups for given class values
            gini = cart_gini_index(groups, class_values)
            # If gini of this feature + value as split is better, update dictionary of best parameters
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    # Return dictionary with best feature and value to split on, and resulting groups of dataset records
        # Difference from CART is we didn't evaluate every input feature, only a random subset
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Random Forest recursively create child splits for a node, or make a terminal node
# Input: pre-split node, max tree depth, min rows per node, node's depth, num features to evaluate
# Output: None, i.e. void function
def rf_split(node, max_depth, min_size, depth, n_features):
    # Extract left and right lists of group rows from the supplied node (dictionary)
    left, right = node['groups']
    # Deletes groups of data from parent node, as it no longer needs access to the data
    del(node['groups'])
    # Checks whether left or right list empty, i.e. whether a no split (100% in one group)
    if not left or not right:
        # Make the only child a terminal node , and set 'left' and 'right' to point to it
        node['left'] = node['right'] = cart_to_terminal(left + right)
        # Exit current iteration, since terminal child node has no child nodes of its own
        return
    # Check whether supplied node is at or above maximum tree depth
    if depth >= max_depth:
        # Set left and right child nodes to terminal nodes
        node['left'], node['right'] = cart_to_terminal(left), cart_to_terminal(right)
        # Exit current iteration, i.e. halting progression down this branch
        return

    # If we reach this point, we neither have a no split, nor have reached max depth
    # Process left child: if shorter than minimum row size, make it terminal
    if len(left) <= min_size:
        node['left'] = cart_to_terminal(left)
    # Neither too deep nor too small, so split left child node to two child nodes
    else:
        # Split left child node
        node['left'] = rf_get_split(left, n_features)
        # Recursively call function on the split left child node in a depth first fashion
        rf_split(node['left'], max_depth, min_size, depth+1, n_features)

    # Process right child: if shorter than minimum, make it a terminal node
    if len(right) <= min_size:
        node['right'] = cart_to_terminal(right)
    # If not, split right child node and make a recursive function call
    else:
        # Split right child node
        node['right'] = rf_get_split(right, n_features)
        # Make recursive call on the split right child
        rf_split(node['right'], max_depth, min_size, depth+1, n_features)

# Build a Random Forest decision tree
def rf_build_tree(train, max_depth, min_size, n_features):
    # Split the root node into two child nodes
    root = rf_get_split(train, n_features)
    # Call recursive function to add left nodes then right nodes in a depth first fashion
    rf_split(root, max_depth, min_size, 1, n_features)
    # Return root node; now just a dictionary with two child node references
    # Similarly, its child nodes are only references, until terminal nodes which contain rows
    return root

# Random Forest Algorithm
# Input: train test sets, tree max depth, min rows per node, subsample ratio, num trees, num features
# Output: list of predictions corresponding to test set
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features = "default"):
    # If using default number of random features to evaluate, use square root of total num features
    if(n_features == "default"):
        n_features = int(sqrt(len(train[0])-1))
    # Create list to hold the tree trained on each subsample
    trees = list()
    # For each tree, bootstrap a subsample, train the random forest tree on it, and append to list
    for i in range(n_trees):
        sample = bootstrap_subsample(train, sample_size)
        tree = rf_build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    # For each row in test, get prediction as most common random forest tree prediction
    predictions = [bagging_predict_mode(trees, row) for row in test]
    # Returns list of predictions for each row in test dataset
    return predictions

# **********************************

# ***** Stacked Generalization *****

# **********************************
