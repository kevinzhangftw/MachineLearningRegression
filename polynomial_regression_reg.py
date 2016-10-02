import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

def designify(training_set, degree):
  designMatrix = np.matrix(np.zeros((training_set.shape[0],1)))
  for i in range(0, training_set.shape[0]):
    designMatrix[i,0]=1
  designMatrix = np.concatenate((designMatrix,training_set), axis=1)
  designMatrix = np.concatenate((designMatrix,np.power(training_set,2)), axis=1)
  return designMatrix

def weightsFromTraining(training_set, training_targets, degree):
  designMatrix = designify(training_set, degree)
  inv = np.linalg.pinv(np.dot(designMatrix.T,designMatrix))
  weights = np.dot(np.dot(inv,designMatrix.T),training_targets)

def trainingError(training_set, training_targets, degree):
  designMatrix = designify(training_set, degree)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.dot(diff.T, diff).item()
  training_error = training_error/2
  return training_error

def regularizer(weights, lambda):
  return (lambda/2)*(np.dot(weights.T,weights))

def RegularizedValidationError(training_set, training_targets, degree, lambda):
  weights = weightsFromTraining(x_train1,2)
  designMatrix_val1 = designify(x_val1,2)
  predicted_target_val1 = np.dot(designMatrix_val1, weights)
  diff = predicted_target_val1 - training_targets
  regularizer = regularizer(weights, lambda)
  validation_error = np.asscalar(1/2 * np.dot(diff.T, diff)) + regularizer
  return validation_error

x = values[:,7:]
x = a1.normalize_data(x)
N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

x_val1 = x_train[0:10,:]
x_val1_target = x_train[0:10,1]

x_train1 = x_train[11:N_TRAIN,:]
x_train1_target = x_train[11:N_TRAIN,1]

#val_1 error at lambda 0