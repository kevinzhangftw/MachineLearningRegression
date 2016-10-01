import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()
# get feature 11
GNI = values[0:100,11]
#feature 11 sorted out 
GNIx= np.linspace(np.asscalar(min(GNI)), np.asscalar(max(GNI)), num=100)
#target
mortality = values[0:100,1]
#test data
GNI_test = values[101:,11]
#target
mortality_test = values[101:,1]

def sigmoid(x, mu, s):
  return 1.0 / (1.0 + np.exp((mu-x) / s))

def designifySigmoid(training_set, length):
  sigmoidMatrix = np.matrix(np.zeros((length,3)))
  for i in range(0, length):
    sigmoidMatrix[i,0]=1

  for i in range(0, length):
    sigmoidMatrix[i,1]=sigmoid(training_set[i],100,2000)
    sigmoidMatrix[i,2]=sigmoid(training_set[i],10000,2000)
  return sigmoidMatrix

def calculateWeights(designMatrix):
  inv = np.linalg.pinv(np.dot(designMatrix.T,designMatrix))
  weights = np.dot(np.dot(inv,designMatrix.T),mortality)
  return weights

def calculateTrainingError(predicted_target):
  diff = predicted_target - mortality
  training_error = 1/2 * np.dot(diff.T, diff)
  return training_error

def calculateTestError(predicted_target):
  diff = predicted_target - mortality_test
  test_error = 1/2 * np.dot(diff.T, diff)
  return test_error

designMatrix = designifySigmoid(GNI, 100)
designMatrixX = designifySigmoid(GNIx, 100)
designMatrixTest = designifySigmoid(GNI_test, 195-101)

weights = calculateWeights(designMatrix)
predicted_target = np.dot(designMatrixX, weights)
 
predicted_test_target = np.dot(designMatrixTest, weights)

training_error = calculateTrainingError(predicted_target)
test_error = calculateTestError(predicted_test_target)

# Produce a plot of results.
plt.plot(GNI, mortality, 'o')
plt.plot(GNIx, predicted_target)
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('mortality')
plt.title('4.3 Sigmoid')
plt.xlabel('GNI')
plt.show()
