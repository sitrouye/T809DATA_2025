# Author: 
# Date:
# Project: 
# Acknowledgements: 
#



from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    samples = []
    target = []
    classes = []
    for i in range (len(locs)) :
        classes.append(i)
        for j in range(n):
            samples.append(norm.rvs(locs[i], scales[i]))
            target.append(i)
    return np.array(samples), np.array(target), classes



def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    sum = 0
    n = 0
    for i in range(len(targets)):
        if targets[i] == selected_class :
            sum += features[i]
            n += 1
    return sum/n


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    ptclass = []
    for i in range(len(targets)):
        if targets[i] == selected_class :
            ptclass.append(features[i])
    ptclass = np.array(ptclass)
    return np.cov(ptclass)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    # return abs(1/(np.sqrt(2*np.pi*class_covar))*np.exp(-(feature - class_mean)**2/(2*class_covar)))
    return norm.pdf(feature, class_mean, np.sqrt(class_covar))

def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [mean_of_class(train_features, train_targets,i) for i in range(len(classes))], [covar_of_class(train_features, train_targets,i) for i in range(len(classes))]
    # for class_label in classes:
    #     ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        like_feat = []
        for j in range(len(classes)):
            like_feat.append(likelihood_of_class(test_features[i], means[j], covs[j]))

        likelihoods.append(like_feat)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    pred = [np.argmax(likelihoods[i]) for i in range(len(likelihoods))]
    return np.array(pred)


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    pass
