#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
# values = a1.normalize_data(values[:,7:])
training_set = values[0:100,7:]
training_targets = values[0:100,1]
test_set = values[101:,7:]
test_targets = values[101:,1]
# x = values[:,7:]
# x = a1.normalize_data(x)
# N_TRAIN = 100;
# x_train = x[0:N_TRAIN,:]
# x_test = x[N_TRAIN:,:]

#precondition: degree must range from 1 to 6
def designify(set, degree):
  design_matrix = np.matrix(np.ones((set.shape[0],1)))
  for i in range(1,degree):
    design_matrix=np.concatenate((design_matrix, np.power(set, degree)), axis=1)
  return design_matrix

def trainingErrorFrom(training_set,training_targets, degree):
  designMatrix = designify(training_set, degree)
  weights = weightsFrom(designMatrix, training_targets)
  difference = differenceFrom(designMatrix, weights, training_targets)
  error = errorFrom(difference)
  return error


def testingErrorFrom(training_set,training_targets, testing_set,testing_targets, degree):
  designMatrix = designify(training_set, degree)
  training_weights = weightsFrom(designMatrix, training_targets)
  testingMatrix = designify(testing_set, degree)
  difference = differenceFrom(testingMatrix, training_weights, testing_targets)
  error = errorFrom(difference)
  return error

def weightsFrom(designMatrix, targets):
  inv = np.linalg.pinv(designMatrix)
  return np.dot(inv,training_targets)

def differenceFrom(designMatrix, weights, targets):
  predicted_target = np.dot(designMatrix, weights)
  return predicted_target - targets

def errorFrom(diff):
  return np.dot(diff.T, diff).item()/2

 

#question 4.2.1
train_err = np.zeros(6)
test_err = np.zeros(6)
degrees = np.arange(1,7)
for degree in degrees:
  train_err[degree-1] = trainingErrorFrom(training_set, training_targets, degree)
  test_err[degree-1] = testingErrorFrom(training_set, training_targets,test_set, test_targets, degree)

# Produce a plot of results.
#plt.plot(test_err.keys(), test_err.values())
plt.plot(degrees, train_err)
plt.plot(degrees, test_err)
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
