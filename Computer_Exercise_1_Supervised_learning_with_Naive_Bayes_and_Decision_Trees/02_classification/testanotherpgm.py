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


features, targets, classes = gen_data(50,[-4,4], [np.sqrt(2), np.sqrt(2)])
print(features)
print(features.shape[0])

class_feat = [[]for i in range(len(classes))]
for i in range(len(features)):
    for j in range(len(classes)):
        if targets[i] == classes[j] :
            class_feat[j].append(features[i])
color = ['b','r','g','k','m', 'c']
marker = ['o','x']
for j in range(len(classes)):
    plt.scatter(class_feat[j],[0 for i in range (len(class_feat[j]))], c = color[j], marker = marker[j] )
plt.show()



(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, 0.8)
class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
print(class_mean, np.sqrt(class_cov))
print(-1, np.sqrt(5))
like = maximum_likelihood(train_features, train_targets, test_features, classes)

l =predict(like)
acc = 0
for i in range(len(l)) :
    if test_targets[i] == l[i]:
        acc +=1
acc = acc/len(l)
print(l)
print('accuracy for dataset 2', acc)








features, targets, classes = gen_data(50,[-1,1], [np.sqrt(5), np.sqrt(5)])


(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, 0.8)
class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
print(class_mean, np.sqrt(class_cov))
print(-1, np.sqrt(5))
like = maximum_likelihood(train_features, train_targets, test_features, classes)

l =predict(like)
acc_1 = 0
for i in range(len(l)) :
    if test_targets[i] == l[i]:
        acc_1 +=1
acc_1 = acc_1/len(l)
print(l)
print('accuracy for dataset 1', acc_1)

