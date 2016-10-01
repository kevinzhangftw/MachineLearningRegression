import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

feature8_train=values[0:100,8] 
feature9_train=values[0:100,9]
feature10_train=values[0:100,10]
feature11_train=values[0:100,11]
feature12_train=values[0:100,12]
feature13_train=values[0:100,13]
feature14_train=values[0:100,14]
feature15_train=values[0:100,15]

training_target_set=values[0:100,1] 


train_error_8 = trainingError(feature8_train,3)
train_error_9 = trainingError(feature9_train,3)
train_error_10 = trainingError(feature10_train,3)
train_error_11 = trainingError(feature11_train,3)
train_error_12 = trainingError(feature12_train,3)
train_error_13 = trainingError(feature13_train,3)
train_error_14 = trainingError(feature14_train,3)
train_error_15 = trainingError(feature15_train,3)

For each (un-normalized) feature fit
a degree 3 polynomial (unregularized).

def trainingError(training_set, degree)
  designMatrix = designify(training_set, degree)
  
  inv = np.linalg.pinv(np.dot(design_matrix.T,design_matrix))

  weights = np.dot(np.dot(inv,design_matrix.T),train_targets)
  
  predicted_target = np.dot(design_matrix, weights)

  diff = predicted_target - training_target_set

  training_error = 1/2 * np.dot(diff, diff.T)

  return training_error

 def designify(training_set, degree)
  # bias column 
  # make a column of ones
  design_matrix = np.ones((100,), dtype=np.int)
  design_matrix = np.concatenate((design_matrix.T,training_set), axis=1)
  # for each degree 
  for i in range(2,degree):
    #concatenate the feature
    design_matrix=np.concatenate((design_matrix, np.power(design_matrix, degree)), axis=1)

  return design_matrix