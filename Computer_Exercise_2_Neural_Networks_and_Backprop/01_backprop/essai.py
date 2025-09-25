from template import *
from typing import Union

import numpy as np
import sklearn.datasets as datasets






def load_iris():
    '''
    Load the iris dataset that contains N input features
    of dimension F and N target classes.

    Returns:
    * inputs (np.ndarray): A [N x F] array of input features
    * targets (np.ndarray): A [N,] array of target classes
    '''
    iris = datasets.load_iris()
    return iris.data, iris.target, [0, 1, 2]


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)



print(sigmoid(0.5))
print(d_sigmoid(0.2))

print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))




# initialize the random generator to get repeatable results
np.random.seed(1234)
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)


# # initialize the random generator to get repeatable results
# np.random.seed(1234)

# # Take one point:
# x = train_features[0, :]
# K = 3 # number of classes
# M = 10
# D = 4
# # Initialize two random weight matrices
# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1
# y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

# print(y, z0, z1, a1, a2)



# # initialize random generator to get predictable results
# np.random.seed(42)

# K = 3  # number of classes
# M = 6
# D = train_features.shape[1]

# x = features[0, :]

# # create one-hot target for the feature
# target_y = np.zeros(K)
# target_y[targets[0]] = 1.0

# # Initialize two random weight matrices
# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1

# y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

# print(y, dE1, dE2)
# print(f'y = {y}')
# print(f'dE1 = {dE1}')
# print(f'dE2 = {dE2}')
# # if dE1 == [[-3.17372897e-03  3.13040504e-02 -6.72419861e-03  7.39219402e-02  -1.16539047e-04  9.29566482e-03] [-1.61860177e-02  1.59650657e-01 -3.42934129e-02  3.77001895e-01  -5.94349138e-04  4.74078906e-02] [-1.11080514e-02  1.09564176e-01 -2.35346951e-02  2.58726791e-01 -4.07886663e-04  3.25348269e-02] [-4.44322055e-03  4.38256706e-02 -9.41387805e-03  1.03490716e-01  -1.63154665e-04  1.30139307e-02] [-6.34745793e-04  6.26081008e-03 -1.34483972e-03  1.47843880e-02 -2.33078093e-05  1.85913296e-03]] :
# #     print(True)









# initialize the random seed to get predictable results
np.random.seed(1234)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)


print(f'W1tr = {W1tr}')
print(f'W2tr = {W2tr}')
print(f'Etotal = {Etotal}')
print(f'misclass = {misclassification_rate}')
print(f'last_guess = {last_guesses}')

guess = test_nn(train_features[:20, :], M, K, W1, W2)
print(f'guesses = {guess}')


