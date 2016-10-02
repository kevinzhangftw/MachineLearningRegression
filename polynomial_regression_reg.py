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
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  return weights

def trainingError(training_set, training_targets, degree):
  designMatrix = designify(training_set, degree)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.dot(diff.T, diff).item()
  training_error = training_error/2
  return training_error

def regularizer(weights, lamb):
  return np.dot(weights.T,weights).item()*lamb

def regularizedValidationError(training_set, training_targets, validation_set, validation_targets, lamb):
  weights = weightsFromTraining(training_set,training_targets,2)
  validation_DM = designify(validation_set,2)
  validation_prediction = np.dot(validation_DM, weights)
  diff = validation_prediction - validation_targets
  validation_error = np.dot(diff.T, diff).item() + regularizer(weights, lamb)
  return validation_error

x = values[:,7:]
x = a1.normalize_data(x)
N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]


x_train1 = x_train[11:N_TRAIN,:]
x_train1_target = x_train[11:N_TRAIN,1]

x_val1 = x_train[0:10,:]
x_val1_target = x_train[0:10,1]

#val_1 error at lambda 0

accumulatingError = np.zeros(10)
errorForEachLamb= np.zeros(8)
lamb = 0
for x in xrange(-3,5):
  if x==-3:
    lamb = 0
  else:
    lamb = 10**x
  for i in xrange(0,10):
    #from i to i * 10
    fromValue = i*10
    toValue = i*10+10
    validation_set = x_train[fromValue:toValue,:]
    validation_target = x_train[fromValue:toValue,1]
    #split
    before_set = x_train[0:fromValue,:]
    before_target = x_train[0:fromValue,1]
    after_set = x_train[toValue:N_TRAIN,:]
    after_target = x_train[toValue:N_TRAIN,1]
    train_set = np.concatenate((before_set, after_set),axis=0)
    train_target = np.concatenate((before_target, after_target),axis=0)
    #TODO: get average row error
    accumulatingError[i] = regularizedValidationError(train_set, train_target, validation_set, validation_target, lamb)
  averageRowError = np.average(accumulatingError)
  errorForEachLamb[x+3] =averageRowError

