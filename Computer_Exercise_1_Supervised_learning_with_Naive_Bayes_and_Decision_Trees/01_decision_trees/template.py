# Author: Elvire Besnard
# Date:24/08/2025
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.array:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    retlist = []
    for i in range(len(classes)):
        val = classes[i]
        I = 0
        for j in range(len(targets)):
            if targets[j] == val :
                I += 1
        retlist.append(I/len(targets))

    return np.array(retlist)




def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    cat = []
    for i in range(len(features)):
        if features[i][split_feature_index]>theta:
            cat.append(False)
        else :
            cat.append(True)



    features_1 = features[cat]
    targets_1 = targets[cat]
    cat_inv = [not(cat[i]) for i in range(len(cat))]
    features_2 = features[cat_inv]
    targets_2 = targets[cat_inv]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 0.5*(1-np.sum(prior(targets, classes)**2))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    n1 = len(t1)*g1/n
    n2 = len(t2)*g2/n

    return n1+n2


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1,t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)

    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        num_points = num_tries
        values = features[:,i]
        min_value = values.min()
        max_value = values.max()
        thetas = np.linspace(min_value, max_value, num_points+2)[1:-1]
        
        # iterate thresholds
        
        for theta in thetas:
            imp = total_gini_impurity(features,targets, classes, i, theta)
            if imp<best_gini:
                best_gini = imp
                best_dim = i
                best_theta = theta


    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        acc = 0
        
        l = self.tree.predict(self.train_features)
        for k in range (len(l)):
            if l[k] == self.train_targets[k] :
                acc +=1

        return acc/(len(self.train_targets))

    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        l = self.tree.predict(self.test_features)
        return l

    def confusion_matrix(self):
        mat_conf = [[0 for i in range(len(self.classes))] for j in range(len(self.classes))]
        guesses = self.guess()
        for i in range(len(guesses)):
            mat_conf[guesses[i]][self.test_targets[i]] += 1
        return mat_conf           
            