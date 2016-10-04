import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

x = values[:,7:]
x = a1.normalize_data(x)
N_TRAIN = 100;

feature8_train=x[0:N_TRAIN,8] 
feature9_train=x[0:100,9]
feature10_train=x[0:100,10]
feature11_train=x[0:100,11]
feature12_train=x[0:100,12]
feature13_train=x[0:100,13]
feature14_train=x[0:100,14]
feature15_train=x[0:100,15]


def designify(training_set):
  designMatrix = np.ones((training_set.shape[0],1))
  for i in xrange(1,4):
  	designMatrix = np.hstack((designMatrix, np.power(training_set,i) ) )

  return designMatrix

def gettingPredictedTarget(training_set):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  return predicted_target

def trainingError(training_set, training_targets):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
  return training_error

def testError(training_set, training_targets, test_set, test_targets):
  designMatrix = designify(training_set)
  inv = np.linalg.pinv(designMatrix)
  weights = np.dot(inv,training_targets)
  testMatrix = designify(test_set)
  predicted_target = np.dot(testMatrix, weights)
  diff = predicted_target - test_targets
  training_error = np.sqrt(np.dot(diff.T, diff).item()/diff.shape[0])
  return training_error



feature8_trainX= np.linspace(np.asscalar(min(feature8_train)), np.asscalar(max(feature8_train)), num=100)
feature9_trainX= np.linspace(np.asscalar(min(feature9_train)), np.asscalar(max(feature9_train)), num=100)
feature10_trainX= np.linspace(np.asscalar(min(feature10_train)), np.asscalar(max(feature10_train)), num=100)
feature11_trainX= np.linspace(np.asscalar(min(feature11_train)), np.asscalar(max(feature11_train)), num=100)
feature12_trainX= np.linspace(np.asscalar(min(feature12_train)), np.asscalar(max(feature12_train)), num=100)
feature13_trainX= np.linspace(np.asscalar(min(feature13_train)), np.asscalar(max(feature13_train)), num=100)
feature14_trainX= np.linspace(np.asscalar(min(feature14_train)), np.asscalar(max(feature14_train)), num=100)
feature15_trainX= np.linspace(np.asscalar(min(feature15_train)), np.asscalar(max(feature15_train)), num=100)

feature8_test=x[N_TRAIN:,8] 
feature9_test=x[N_TRAIN:,9]
feature10_test=x[N_TRAIN:,10]
feature11_test=x[N_TRAIN:,11]
feature12_test=x[N_TRAIN:,12]
feature13_test=x[N_TRAIN:,13]
feature14_test=x[N_TRAIN:,14]
feature15_test=x[N_TRAIN:,15]

training_targets=values[0:100,1]
test_targets=values[N_TRAIN:,1] 

train_error_8 = trainingError(feature8_trainX, training_targets)
train_error_9 = trainingError(feature9_trainX,training_targets)
train_error_10 = trainingError(feature10_trainX,training_targets)
train_error_11 = trainingError(feature11_trainX,training_targets)
train_error_12 = trainingError(feature12_trainX,training_targets)
train_error_13 = trainingError(feature13_trainX,training_targets)
train_error_14 = trainingError(feature14_trainX,training_targets)
train_error_15 = trainingError(feature15_trainX,training_targets)

test_error_8 = testError(feature8_train, training_targets,feature8_test, test_targets)
test_error_9 = testError(feature9_train,training_targets,feature9_test, test_targets)
test_error_10 = testError(feature10_train,training_targets,feature10_test, test_targets)
test_error_11 = testError(feature11_train,training_targets,feature11_test, test_targets)
test_error_12 = testError(feature12_train,training_targets,feature12_test, test_targets)
test_error_13 = testError(feature13_train,training_targets,feature13_test, test_targets)
test_error_14 = testError(feature14_train,training_targets,feature14_test, test_targets)
test_error_15 = testError(feature15_train,training_targets,feature15_test, test_targets)

predicted_target11 = gettingPredictedTarget(feature11_trainX)
predicted_target12 = gettingPredictedTarget(feature12_trainX)
predicted_target13 = gettingPredictedTarget(feature13_trainX)

# Produce bar chart .
featuresToPlot = [8,9,10,11,12,13,14,15]
barWidth = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.35]
plt.bar(featuresToPlot, [train_error_8,train_error_9,train_error_10,train_error_11,train_error_12,train_error_13,train_error_14,train_error_15], 0.35, alpha=1,
                 color='b',
                 label='Train Error')
plt.bar(featuresToPlot+barWidth, [test_error_8,test_error_9,test_error_10,test_error_11,test_error_12,test_error_13,test_error_14,test_error_15], 0.35, alpha=1,
                 color='r',
                 label='Test Error')

plt.ylabel('RMSE')
plt.title('4.2 Bar Chart')
plt.xlabel('features 8-15')
plt.show()

#Produce feature 11 GNI
plt.plot(feature11_train, training_targets, 'o')
plt.plot(feature11_trainX, predicted_target11)
plt.ylabel('mortality')
plt.title('4.2 Feature 11 GNI')
plt.xlabel('GNI')
plt.show()

#Produce feature 12 Life expectancy
plt.plot(feature12_train, training_targets, 'o')
plt.plot(feature12_trainX, predicted_target12)
plt.ylabel('mortality')
plt.title('4.2 Feature 12 Life Expectancy')
plt.xlabel('Life expectancy')
plt.show()

#Produce feature 13 Literacy
plt.plot(feature13_train, training_targets, 'o')
plt.plot(feature13_trainX, predicted_target13)
plt.ylabel('mortality')
plt.title('4.2 Feature 13 Literacy')
plt.xlabel('Literacy')
plt.show()
