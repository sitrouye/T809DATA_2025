from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test
from template import *


features, targets, classes = load_iris()
# print(features)
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
print(len(f_1), len(f_2))
print(gini_impurity(t_1, classes))

print(weighted_impurity(t_1, t_2, classes))
print(total_gini_impurity(features, targets, classes, 2, 4.65))
# print(features)
# print(features[:,0])
print(brute_best_split(features, targets, classes, 30))

iristree = IrisTreeTrainer(features, targets, classes,0.8)
iristree.train()

print(iristree.accuracy())
iristree.plot()
print(iristree.confusion_matrix())

