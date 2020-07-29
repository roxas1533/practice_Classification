import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1], [2], [3]])
Wy = np.random.random((10, 4 + 1))
tempy = np.delete(Wy.T, len(Wy.T) - 1, 0).T

print(Wy)
print(tempy)
# Wy = np.random.random((3, 5 + 1))
# print(Wy)
# print(np.delete(Wy.T, len(Wy[0]) - 1, 0))
