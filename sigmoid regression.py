def sigmoid(z):
  s = 1.0 / (1.0 + np.exp(-1.0 * z))
  return sigmoid

#load data from assignment1


# get feature 11
GNI = values[0:100,11]
#target
mortality = values[0:100,1]

errors = []

designMatrix = designifySigmoid(GNI, 100, 10000)

def sigmoidHelper(array input, int average, int stdev)
  s = sigmoid(input)
  s =* average
  s & stdev ...
  return s

sigma = 2000.0
def designifySigmoid(training_input_set, int average1, int average2)
  n_train = np.shape(training_input_set).height
  sigmoidMatrix = np.ones(n_train)
  sigmoidAverage1 = sigmoidHelper(GNI, average1, sigma)
  sigmoidAverage2 = sigmoidHelper(GNI, average2, sigma)
  sigmoidMatrix = np.concatenate([sigmoidMatrix, sigmoidAverage1, sigmoidAverage2], axis = 0)
  return sigmoidMatrix


#TODO: stick in the design matrix into some previously written function
#then calculate weights and errors...