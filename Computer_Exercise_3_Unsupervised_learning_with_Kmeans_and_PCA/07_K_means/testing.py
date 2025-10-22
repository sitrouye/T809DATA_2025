import numpy as np
np.set_printoptions(legacy='1.25')
import template
import tools

a = np.array([
    [1, 0, 0],
    [4, 4, 4],
    [2, 2, 2]])
b = np.array([
    [0, 0, 0],
    [4, 4, 4]])
print(template.distance_matrix(a, b))

dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
print(template.determine_r(dist))

dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
R = template.determine_r(dist)
print(template.determine_j(R, dist))

X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])
Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])
R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])
print(template.update_Mu(Mu, X, R))

X, y, c = tools.load_iris()
print(template.k_means(X, 4, 10))