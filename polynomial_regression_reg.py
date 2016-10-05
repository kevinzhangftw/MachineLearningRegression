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

#set 1
features_TRAIN = features_INPUT[10:N_TRAIN,:]
target_TRAIN = target_INPUT[10:N_TRAIN,:]

features_VAL = features_INPUT[:10,:]
target_VAL = target_INPUT[:10,:]
#wish: to split into 10 validation sets, and 10 features set

#set2
features_TRAINup = features_INPUT[0:10,:]
features_TRAINdown = features_INPUT[20:N_TRAIN,:]

target_TRAINup = target_INPUT[0:10,:]
target_TRAINdown = target_INPUT[20:N_TRAIN,:]
#concatenate features together
features_TRAIN1 = np.concatenate((features_TRAINup,features_TRAINdown),axis=0)
target_TRAIN1 = np.concatenate((target_TRAINup,target_TRAINdown),axis=0)

features_VAL = features_INPUT[10:20,:]
target_VAL = target_INPUT[10:20,:]

#set3
features_TRAINup = features_INPUT[0:20,:]
features_TRAINdown = features_INPUT[30:N_TRAIN,:]

target_TRAINup = target_INPUT[0:20,:]
target_TRAINdown = target_INPUT[30:N_TRAIN,:]
#concatenate features together
features_TRAIN1 = np.concatenate((features_TRAINup,features_TRAINdown),axis=0)
target_TRAIN1 = np.concatenate((target_TRAINup,target_TRAINdown),axis=0)

features_VAL = features_INPUT[20:30,:]
target_VAL = target_INPUT[20:30,:]


#init matrices
target_TRAINup = np.zeros((10,90,1))
target_TRAINdown = np.zeros((10,90,1))
#concatenate features together
features_TRAIN1 = np.zeros((10,90,33))
target_TRAIN1 = np.zeros((10,10,1))

features_VAL = np.zeros((10,10,33))
target_VAL = np.zeros((10,10,1))
for i in xrange(0,10):
	features_TRAINup = np.zeros((10,(i*10),33))
	features_TRAINup[i] = features_INPUT[0:(i*10),:]

	
	features_TRAINdown[i] = features_INPUT[(i+1)*10:N_TRAIN,:]
	features_TRAIN[i] = np.concatenate((features_TRAINup[i],features_TRAINdown[i]),axis=0)



	target_TRAIN[i] = np.concatenate((target_TRAINup,target_TRAINdown),axis=0)


	features_VAL[i] = features_INPUT[(i*10):(i+1)*10,:]
	target_VAL[i] = target_INPUT[(i*10):(i+1)*10,:]


lamb =1
#wish: lamba from = {10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3, 10^4}
for i in xrange(1,9):
	if i==1:
		lamb = 0
		#get regularizedRMSRow array when lamb = 0

	else:
		lamb = 10**(i-4)
		#get regularizedRMSRow array when lamb = 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3, 10^4}


regularizedRMSRow = np.zeros(10)
regularizedRMSRow[0] = gettingRegularizedRMS(features_TRAIN, target_TRAIN, lamb,features_VAL, target_VAL)



#RMSRow is array of 10, first position is regularizedRMSRow[0], rest of 9 are 0s
avgRegularizedRMSRow = np.average(regularizedRMSRow)

