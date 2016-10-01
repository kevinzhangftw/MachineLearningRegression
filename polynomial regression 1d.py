import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

def designify(training_set, degree):
  designMatrix = np.matrix(np.zeros((training_set.shape[0],4)))
  for i in range(0, training_set.shape[0]):
    designMatrix[i,0]=1
  for i in range(0, training_set.shape[0]):
    designMatrix[i,1]=np.power(training_set[i], 1)
    designMatrix[i,2]=np.power(training_set[i], 2)
    designMatrix[i,3]=np.power(training_set[i], 3)
  return designMatrix


def trainingError(training_set, degree):
  designMatrix = designify(training_set, degree)
  inv = np.linalg.pinv(np.dot(designMatrix.T,designMatrix))
  weights = np.dot(np.dot(inv,designMatrix.T),training_targets)
  predicted_target = np.dot(designMatrix, weights)
  diff = predicted_target - training_targets
  training_error = np.asscalar(1/2 * np.dot(diff.T, diff))
  return training_error

feature8_train=values[0:100,8] 
feature9_train=values[0:100,9]
feature10_train=values[0:100,10]
feature11_train=values[0:100,11]
feature12_train=values[0:100,12]
feature13_train=values[0:100,13]
feature14_train=values[0:100,14]
feature15_train=values[0:100,15]

training_targets=values[0:100,1] 

train_error_8 = trainingError(feature8_train,3)
train_error_9 = 9
train_error_10 = 10
train_error_11 = trainingError(feature11_train,3)
train_error_12 = trainingError(feature12_train,3)
train_error_13 = trainingError(feature13_train,3)
train_error_14 = trainingError(feature14_train,3)
train_error_15 = 15

# Produce bar chart .
#plt.bar([8,9,10,11,12,13,14,15], [train_error_8,train_error_9,train_error_10,train_error_11,train_error_12,train_error_13,train_error_14,train_error_15])
#plt.ylabel('RMSE')
#plt.title('4.2 Bar Chart')
#plt.xlabel('features 8-15')

#Produce feature 11 GNI
plt.plot(feature11_train, training_targets, 'o')
plt.ylabel('mortality')
plt.title('4.2 Feature 11 GNI')
plt.xlabel('GNI')
plt.show()

#Produce feature 12 Life expectancy
plt.plot(feature12_train, training_targets, 'o')
plt.ylabel('mortality')
plt.title('4.2 Feature 12 Life Expectancy')
plt.xlabel('Life expectancy')
plt.show()

#Produce feature 13 Literacy
plt.plot(feature13_train, training_targets, 'o')
plt.ylabel('mortality')
plt.title('4.2 Feature 13 Literacy')
plt.xlabel('Literacy')
plt.show()
