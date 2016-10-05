import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

inputValues = values[:,7:]
inputValues = a1.normalize_data(inputValues)
N_TRAIN = 100;

features_INPUT = inputValues[:N_TRAIN,:]
target_INPUT = values[:N_TRAIN,1]

def designify(training_set):
  designMatrix = np.ones((training_set.shape[0],1))
  for i in xrange(1,3):
  	# print training_set.shape
  	designMatrix = np.hstack((designMatrix, np.power(training_set,i) ) )
  return designMatrix

def gettingPredictedTarget(training_set, training_targets):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  return predicted_target

def gettingWeights(training_set,training_targets):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  return weights

def trainingError(training_set, training_targets):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
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

#TODO derive weights with regularizer 
def gettingRegWeights(designMatrix, targets, lamb):
	#w = (λI + Φ^T * Φ)^-1*Φ^T*t
	designMatrixTranspose = designMatrix.T
	designMatrixSquare = np.dot(designMatrixTranspose, designMatrix)
	I = np.identity(designMatrixSquare.shape[0])
	innerSum = lamb*I + designMatrixSquare
	innerSumInversed = np.linalg.inv(innerSum)
	regularMatrix = innerSumInversed*designMatrixTranspose
	weights = regularMatrix*targets
	return weights

def gettingRegularizedRMS(features_TRAIN, target_TRAIN, lamb,features_VAL, target_VAL):
	designMatrix = designify(features_TRAIN)
	lamb = 1
	weights = gettingRegWeights(designMatrix, target_TRAIN, lamb)
	validationMatrix = designify(features_VAL) 
	regularizedDiff = validationMatrix*weights - target_VAL
	regularizedRMS = np.sqrt((regularizedDiff.T*regularizedDiff)/designMatrix.shape[0])
	return regularizedRMS.item()

#TODO: split input features into training features and validation feature...
features_TRAIN = features_INPUT[:90,:]
target_TRAIN = target_INPUT[:90,:]
features_VAL = features_INPUT[90:N_TRAIN,:]
target_VAL = target_INPUT[90:N_TRAIN,:]

lamb =1

regularizedRMSRow = np.zeros(10)
regularizedRMSRow[0] = gettingRegularizedRMS(features_TRAIN, target_TRAIN, lamb,features_VAL, target_VAL)


