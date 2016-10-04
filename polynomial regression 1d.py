import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

inputValues = values[:,7:]
inputValues = a1.normalize_data(inputValues)
N_TRAIN = 100;

features_TRAIN = np.zeros((8, 100,1))
target_TRAIN = values[:N_TRAIN,1]
train_ERROR = np.zeros(8)

features_TEST = np.zeros((8, 95, 1))
target_TEST = values[N_TRAIN:,1]
test_ERROR = np.zeros(8)

def designify(training_set):
  designMatrix = np.ones((training_set.shape[0],1))
  for i in xrange(1,4):
  	# print training_set.shape
  	designMatrix = np.hstack((designMatrix, np.power(training_set,i) ) )
  return designMatrix

def gettingPredictedTarget(training_set):
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

def testError(weights, test_set, test_targets):
  testMatrix = designify(test_set)
  predicted_target = np.dot(testMatrix, weights)
  diff = predicted_target - test_targets
  test_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
  return test_error

for i in xrange(0,8):
	features_TRAIN[i] = inputValues[0:N_TRAIN,i]
	features_TEST[i] = inputValues[N_TRAIN:,i]
	train_ERROR[i] = trainingError(features_TRAIN[i], target_TRAIN)
	weights = gettingWeights(features_TRAIN[i],target_TRAIN)
	test_ERROR[i] = testError(weights, features_TEST[i], target_TEST)

#wish linspace from features_train 11 12 13
linspace11 = np.linspace(np.amin(features_TRAIN[3]), np.amax(features_TRAIN[3]), num=500)
weights11 = gettingWeights(features_TRAIN[3],target_TRAIN)
linspaceMatrix11 = designify(np.asmatrix(linspace11).T)
linspacePredict11 = np.dot(linspaceMatrix11, weights11)


# Produce bar chart .
# index = np.arange(8)
# barWidth = 0.35
# plt.bar(index, train_ERROR, 0.35, alpha=1,
#                  color='b',
#                  label='Train Error')

# plt.bar(index+barWidth, test_ERROR, 0.35, alpha=1,
#                  color='r',
#                  label='Test Error')

# plt.ylabel('RMSE')
# plt.title('4.2 Bar Chart')
# plt.xlabel('features 8-15')
# plt.show()

#Produce feature 11 GNI
plt.plot(features_TRAIN[3], target_TRAIN, 'o')
plt.plot(features_TEST[3], target_TEST, 'o')
plt.plot(linspace11, linspacePredict11)
plt.ylabel('mortality')
plt.title('4.2 Feature 11 GNI')
plt.xlabel('GNI')
plt.show()

# #Produce feature 12 Life expectancy
# plt.plot(feature12_train, training_targets, 'o')
# plt.plot(feature12_trainX, predicted_target12)
# plt.ylabel('mortality')
# plt.title('4.2 Feature 12 Life Expectancy')
# plt.xlabel('Life expectancy')
# plt.show()

# #Produce feature 13 Literacy
# plt.plot(feature13_train, training_targets, 'o')
# plt.plot(feature13_trainX, predicted_target13)
# plt.ylabel('mortality')
# plt.title('4.2 Feature 13 Literacy')
# plt.xlabel('Literacy')
# plt.show()
