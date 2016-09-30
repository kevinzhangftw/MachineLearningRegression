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
training_targets = values[0:100, 1]

#question 4.2.1
#calculateErrors for degree 1
  e_train = trainingError(training_set, 1)
  #e_test = testingError(testing_set, 1)

def trainingError(training_set, degree)
  designMatrix = designify(training_set, degree)
  
  inv = np.linalg.pinv(np.dot(design_matrix.T,design_matrix))

  weights = np.dot(np.dot(inv,design_matrix.T),train_targets)
  
  predicted_target = np.dot(design_matrix, weights)

  diff = predicted_target - training_target_set

  training_error = 1/2 * np.dot(diff, diff.T)


  return training_error

def designify(training_set, degree)
  # bias column 
  # make a column of ones
  design_matrix = np.ones((100,), dtype=np.int)
  design_matrix = np.concatenate((design_matrix.T,training_set), axis=1)
  # for each degree 
  for i in range(2,degree):
    #concatenate the feature
    design_matrix=np.concatenate((design_matrix, np.power(design_matrix, degree)), axis=1)

  return design_matrix

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
