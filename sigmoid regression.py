import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

# get feature 11
GNI = values[0:100,11]
#target
mortality = values[0:100,1]

def sigmoid(x, mu, s):
  return 1.0 / (1.0 + np.exp((mu-x) / s))

def designifySigmoid():
  sigmoidMatrix = np.matrix(np.zeros((100,3)))
  for i in range(0, 100):
    sigmoidMatrix[i,0]=1

  for i in range(0, 100):
    sigmoidMatrix[i,1]=sigmoid(GNI[i],100,2000)
    sigmoidMatrix[i,2]=sigmoid(GNI[i],10000,2000)
  return sigmoidMatrix

designMatrix = designifySigmoid()
inv = np.linalg.pinv(np.dot(design_matrix.T,design_matrix))
weights = np.dot(np.dot(inv,design_matrix.T),train_targets)  
predicted_target = np.dot(design_matrix, weights)

# Produce a plot of results.
plt.plot(GNI, mortality, 'o')
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('mortality')
plt.title('4.3 Sigmoid')
plt.xlabel('GNI')
plt.show()
