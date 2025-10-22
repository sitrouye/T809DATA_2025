import tools
import template
import numpy as np
from matplotlib import pyplot as plt
import sklearn

# l = template.standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]]))
# print(l)

# X = np.array([
#     [1, 2, 3, 4],
#     [0, 0, 0, 0],
#     [4, 5, 5, 4],
#     [2, 2, 2, 2],
#     [8, 6, 4, 2]])
# template.scatter_standardized_dims(X, 0, 2)
# # plt.show()
X, y = tools.load_cancer()
# template._scatter_cancer()

# plt.show()

# plt.close('all')


# template._plot_pca_components()
# template._plot_eigen_values()
# template._plot_log_eigen_values()
template._plot_cum_variance()