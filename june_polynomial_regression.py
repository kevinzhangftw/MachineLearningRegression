#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

train_targets = values[0:100,1]
test_targets = values[101:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

training_set = np.array(values[0:100,7:])

#question 4.2.1
#plot the minimized error for each degree up to 6
feature_start_index = 3
feature_end_index = 44


def calculateErrors
  degree = 1
  e_train = trainingError(training_set, 1)
  e_test = testingError(testing_set, 1)
  # get training error for degree 1
  # get test error for degree 1

def designifyPolynomial(training_input_set, int degree)
  # bias column 
  # make a column of ones
  design_matrix = np.ones((100,), dtype=np.int)
  design_matrix = np.concatenate((design_matrix.T,training_set), axis=1)
  # for each degree 
  for i in range(2,degree):
    #concatenate the feature
    design_matrix=np.concatenate((design_matrix, np.power(design_matrix, degree)), axis=1)

  return design_matrix


def trainingError(matrix training_set, int degree)
  training_target_set = values[0:100, 1]
  training_input_set = values[0:100,7:]

  designMatrix = designify(training_input_set, degree)
  #TODO: inspect size of design matrix for certainty

  #fancy one-liner here...
  #weights = np.linalg.pinv(designMatrix) * training_target_set
  #weights = ( designMatrix * designMatrix^T ) * designMatrix^-1 * training_target_set
  
  #predicted_target = weights * design matrix

  #error = 1/2 * dot_product(predicted_target - training_target_set)


  return float of error ...

def plotError
  # plot against polynomial degree
  for degree in xrange(1,6):
  	#get errors for each degree..
  	error = trainingError(training_set, degree)
  	pass








# Produce a plot of results.
#plt.plot(train_err.keys(), train_err.values())
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
