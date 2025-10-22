# Author: Elvire Besnard
# Date:16/10/25
# Project: computer exercise 3 DMML
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer



def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    X_m = np.mean(X, axis = 0)
    X_sig = np.std(X, axis = 0)

    N, f = np.shape(X)
    X_hat = []
    for i in range(N) :
        X_hat.append([])
        for j in range(f):
            X_hat[i].append((X[i][j] - X_m[j])/X_sig[j])

    return np.array(X_hat)



def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_hat = standardize(X)
    N, f = np.shape(X)
    X_hat_1 = [X_hat[k][i] for k in range(N)]
    X_hat_2 = [X_hat[k][j] for k in range(N)]
    plt.scatter(X_hat_1, X_hat_2, s= 9)


def _scatter_cancer():
    X, y = load_cancer()
    plt.figure()
    # X = X[:5]
    N, f = np.shape(X)
    print('N = ', N)
    for i in range(f):
        plt.subplot(5,6,i+1)
        scatter_standardized_dims(X, 0, i)


def _plot_pca_components():
    pca = PCA()
    X, y = load_cancer()
    pca.fit_transform(standardize(X))
    comps = pca.components_
    print(np.shape(comps))
    for i in range(len(comps)):
        ax = plt.subplot(5,6, i + 1)
        plt.plot(comps[i])
        ax.get_xaxis().set_visible(False)
        plt.title(f"PCA {i+1}")
    plt.show()


def _plot_eigen_values():
    pca = PCA()
    X, y = load_cancer()
    pca.fit_transform(standardize(X))
    lambs = pca.explained_variance_

    plt.plot(np.arange(len(lambs)), lambs, )
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    pca = PCA()
    X, y = load_cancer()
    pca.fit_transform(standardize(X))
    lambs = pca.explained_variance_
    log_lambs = np.log10(lambs)
    plt.plot(np.arange(len(lambs)), log_lambs)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    pca = PCA()
    X, y = load_cancer()
    pca.fit_transform(standardize(X))
    lambs = pca.explained_variance_
    cumu_lambs = np.cumsum(lambs)
    cumu_lambs = cumu_lambs/cumu_lambs[len(cumu_lambs)-1]
    plt.plot(np.arange(len(lambs)), cumu_lambs)
    
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()
