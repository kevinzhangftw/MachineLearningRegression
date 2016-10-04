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

def gettingLinspacepredict(features_TRAIN, target_TRAIN, featureIndex, linspace):
	linspaceposition = featureIndex - 3
	linspace[linspaceposition] = np.transpose(np.asmatrix(np.linspace(np.amin(features_TRAIN[featureIndex]), np.amax(features_TRAIN[featureIndex]), num=500)))
	weights = gettingWeights(features_TRAIN[featureIndex],target_TRAIN)
	linspaceMatrix = designify(np.asmatrix(linspace[linspaceposition]))
	linspacePredictTarget = np.dot(linspaceMatrix, weights)
	return linspacePredictTarget

for i in xrange(0,8):
	features_TRAIN[i] = inputValues[0:N_TRAIN,i]
	features_TEST[i] = inputValues[N_TRAIN:,i]
	train_ERROR[i] = trainingError(features_TRAIN[i], target_TRAIN)
	weights = gettingWeights(features_TRAIN[i],target_TRAIN)
	test_ERROR[i] = testError(weights, features_TEST[i], target_TEST)

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

linspace = np.zeros((3, 500,1))
linspacePredict = np.zeros((3, 500,1))
linspacePredict[0] = gettingLinspacepredict(features_TRAIN, target_TRAIN, 3, linspace)
linspacePredict[1] = gettingLinspacepredict(features_TRAIN, target_TRAIN, 4, linspace)
linspacePredict[2] = gettingLinspacepredict(features_TRAIN, target_TRAIN, 5, linspace)

def gettingPlot(featureIndex, ylabel, title, xlabel):
	plt.plot(features_TRAIN[featureIndex], target_TRAIN, 'o')
	plt.plot(features_TEST[featureIndex], target_TEST, 'o')
	linspaceposition = featureIndex - 3
	plt.plot(linspace[linspaceposition], linspacePredict[linspaceposition])
	plt.ylabel(ylabel)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.show()

# gettingPlot(3, 'mortality', '4.2 Feature 11 GNI', 'GNI')
gettingPlot(4, 'mortality', '4.2 Feature 12 Life Expectancy', 'Life expectancy')
# gettingPlot(5, 'mortality', '4.2 Feature 13 Literacy', 'Literacy')
