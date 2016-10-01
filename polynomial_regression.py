#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
training_set = values[0:100,7:]
training_targets = values[0:100,1]
test_targets = values[101:,1]
x = values[:,7:]
#x = a1.normalize_data(x)
N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

#precondition: degree must range from 1 to 6
def designify(training_set, degree):
  # bias column 
  # make a column of ones
  design_matrix = np.matrix(np.zeros((training_set.shape[0],1)))
  for i in range(0, training_set.shape[0]):
    design_matrix[i,0]=1
  design_matrix = np.concatenate((design_matrix,training_set), axis=1)
  # for each degree 
  for i in range(1,degree):
    #concatenate the feature
    design_matrix=np.concatenate((design_matrix, np.power(design_matrix, degree)), axis=1)
  return design_matrix

def trainingError(training_set, degree):
  designMatrix = designify(training_set, degree)
  inv = np.linalg.pinv(np.dot(designMatrix.T,designMatrix))
  weights = np.dot(np.dot(inv,designMatrix.T),training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = 1/2 * np.dot(diff, diff.T)
  return training_error

#question 4.2.1
error_train_deg1 = trainingError(training_set, 1)
error_train_deg2 = trainingError(training_set, 2)
error_train_deg3 = trainingError(training_set, 3)
error_train_deg4 = trainingError(training_set, 4)
error_train_deg5 = trainingError(training_set, 5)
error_train_deg6 = trainingError(training_set, 6)
  #e_test = testingError(testing_set, 1)

# Produce a plot of results.
#plt.plot(train_err.keys(), train_err.values())
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
