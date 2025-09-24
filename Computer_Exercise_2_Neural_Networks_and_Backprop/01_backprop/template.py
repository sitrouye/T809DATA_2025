from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0
    else :
        return 1/(1+np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x)*(1-sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    wsum = np.dot(x,w)
    return wsum, sigmoid(wsum)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    print(f"x = {x}")
    z0 = np.append([1.0], x)

    a1 = np.zeros(M)
    z = np.zeros(M)

    for m in range(M):
        ws, sigws = perceptron(z0, W1[:,m])
        z[m] = sigws
        a1[m] = ws

    z1 = np.append([1.0], z)
    

    a2 = np.zeros(K)
    y = np.zeros(K)
    for k in range(K):
        ws, sigws = perceptron(z1, W2[:,k])
        y[k] = sigws
        a2[k] = ws

    return y,z0,z1,a1,a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''

    y,z0,z1,a1,a2 = ffnn(x,M,K,W1,W2)

    deltaK = np.zeros(K)
    for k in range(K):
        deltaK[k] = y[k] - target_y[k]

    deltaJ = np.zeros(M)
    for j in range(M):
        deltaJ[j] = d_sigmoid(a1[j])*np.sum([W2[j+1,k]*deltaK[k] for k in range(K)])
    

    dE1 = np.zeros(np.shape(W1))
    dE2 = np.zeros(np.shape(W2))
    for i in range(len(x)+1):
        for j in range(M) : 
            dE1[i,j] = deltaJ[j]*z0[i]
            
    for j in range(M+1):
        for k in range(K) : 
            dE2[j,k] = deltaK[k]*z1[j]


    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    for i in range(len(X_train)):
        y,z0,z1,a1,a2 = ffnn(X_train[i],M,K,W1,W2) #output variables for 1 set of features

        


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    pass