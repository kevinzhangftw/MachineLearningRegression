import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()

inputValues = values[:,7:]
# inputValues = a1.normalize_data(inputValues)
N_TRAIN = 100;

features_TRAIN = np.zeros((8, 100,1))
target_TRAIN = values[:N_TRAIN,1]
train_ERROR = np.zeros(8)

features_TEST = np.zeros((8, 95, 1))
target_TEST = values[N_TRAIN:,1]
test_ERROR = np.zeros(8)

def sigmoid(x, mu, s):
  return 1.0 / (1.0 + np.exp((mu-x) / s))

def designifySigmoid(set):
  sigmoidMatrix = np.ones((set.shape[0],3))
  for i in range(0, set.shape[0]):
    sigmoidMatrix[i,1]=sigmoid(set[i],100,2000)
    sigmoidMatrix[i,2]=sigmoid(set[i],10000,2000)
  return sigmoidMatrix


def gettingWeights(training_set,training_targets):
  designMatrix = designifySigmoid(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  return weights

def trainingError(training_set, training_targets):
  designMatrix = designifySigmoid(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
  return training_error

def testError(weights, test_set, test_targets):
  testMatrix = designifySigmoid(test_set)
  predicted_target = np.dot(testMatrix, weights)
  diff = predicted_target - test_targets
  test_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
  return test_error

def gettingLinspacepredict(features_TRAIN, target_TRAIN, featureIndex, linspace):
  linspaceposition = featureIndex - 3
  linspace[linspaceposition] = np.transpose(np.asmatrix(np.linspace(np.amin(features_TRAIN[featureIndex]), np.amax(features_TRAIN[featureIndex]), num=500)))
  weights = gettingWeights(features_TRAIN[featureIndex],target_TRAIN)
  linspaceMatrix = designifySigmoid(linspace[linspaceposition])
  linspacePredictTarget = np.dot(linspaceMatrix, weights)
  return linspacePredictTarget

#getting feature 11
i=3
features_TRAIN[i] = inputValues[0:N_TRAIN,i]
features_TEST[i] = inputValues[N_TRAIN:,i]
train_ERROR[i] = trainingError(features_TRAIN[i], target_TRAIN)
weights = gettingWeights(features_TRAIN[i],target_TRAIN)
test_ERROR[i] = testError(weights, features_TEST[i], target_TEST)

#plot training error and test error 
index = np.arange(1)
barWidth = 0.35
plt.bar(index, train_ERROR[3], 0.35, alpha=1,
                 color='b',
                 label='Train Error')

plt.bar(index+barWidth, test_ERROR[3], 0.35, alpha=1,
                 color='r',
                 label='Test Error')
plt.legend(['Training error','Test error'])
plt.ylabel('RMSE')
plt.title('4.3 Sigmoid Error Comparsion')
plt.xlabel('features 11')
plt.show()



linspace = np.zeros((3, 500,1))
linspacePredict = np.zeros((3, 500,1))
linspacePredict[0] = gettingLinspacepredict(features_TRAIN, target_TRAIN, 3, linspace)

def gettingPlot(ylabel, title, xlabel):
  plt.plot(features_TRAIN[3], target_TRAIN, 'o')
  plt.plot(features_TEST[3], target_TEST, 'o')
  plt.plot(linspace[0], linspacePredict[0])
  plt.ylabel(ylabel)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.show()

# gettingPlot('mortality', '4.3 Sigmoid Feature 11', 'GNI')

