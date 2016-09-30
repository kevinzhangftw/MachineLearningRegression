#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


#question 4.2.1
#plot the minimized error for each degree up to 6
feature_start_index = 3
feature_end_index = 44
training_set = np.array(values[0:100,7:])


def calculateErrors
  degree = 1
  e_train = trainingError(training_set, 1)
  e_test = testingError(testing_set, 1)
  # get training error for degree 1
  # get test error for degree 1

def designify(training_input_set, int degree)
  # bias column | each feature of each degree
  # make a column of ones
  design_matrix = ones(training_input_set.height())
  # for each degree 
  for int i = 1 until i == degree, i++
    #take it to the power of degree for each entry
    take_to_the_power(training_input_set, i)
    #concatenate the feature
    design_matrix.concatenate(features, axis = 0)


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
