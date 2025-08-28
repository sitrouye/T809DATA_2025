from typing import Union

import numpy as np
import sklearn.datasets as datasets

from template import *
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


print(norm.rvs(0,1))
print(gen_data(2,[0,2], [4,4]))


features, targets, classes = gen_data(50,[-1,1], [np.sqrt(5), np.sqrt(5)])
print(features)
print(features.shape[0])



plt.scatter(features,[0 for i in range (len(features))])
# plt.show()



(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, 0.8)
class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
print(class_mean, np.sqrt(class_cov))
print(-1, np.sqrt(5))
like = maximum_likelihood(train_features, train_targets, test_features, classes)
print(predict(like))