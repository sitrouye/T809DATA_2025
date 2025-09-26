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
    # print(f"x = {x}")
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
    W1_upd = W1
    W2_upd = W2
    misclass = np.zeros(iterations)
    E_total = []
    N, _ = np.shape(X_train)
    
    for it in range (iterations) :
        dE1_tot = np.zeros(np.shape(W1))
        dE2_tot = np.zeros(np.shape(W2))
        guess = []
        loss = 0
        for i in range(N):
            # y,z0,z1,a1,a2 = ffnn(X_train[i],M,K,W1,W2) #output variables for 1 set of features
            target = np.zeros(K)
            target[t_train[i]] = 1
            y, dE1, dE2 = backprop(X_train[i,:], target, M, K, W1_upd,W2_upd)
            classif = np.argmax(y)
            for j in range(K):
                # loss += (y[j] - target[j])**2 #sum of squares
                loss -= target[j]*np.log(y[j])+ (1-target[j])*np.log(1-y[j]) #cross entropy
            
            if  classif != t_train[i]:
                misclass[it] += 1/N

            guess.append(int(classif))
                
            dE1_tot += dE1
            dE2_tot += dE2


        W1_upd = W1_upd - eta*dE1_tot/N #mean over all the training data
        W2_upd = W2_upd - eta*dE2_tot/N 
        E_total.append(float(loss/N))

        
        it += 1
    
    return W1_upd, W2_upd, E_total, misclass, guess

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
    N, _ = np.shape(X)
    guesses = np.zeros(N)
    for i in range(N):
        y, _, _, _, _ = ffnn(X[i,:], M, K, W1, W2)
        
        classif = np.argmax(y)
        guesses[i] = classif
    return guesses



if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    pass