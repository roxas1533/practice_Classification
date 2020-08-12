import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1000)
t = [1.0, 2.0]
y = np.hstack((t, 1))
print(y)
# Wy = np.random.random((3, 5 + 1))
# print(Wy)
# print(np.delete(Wy.T, len(Wy[0]) - 1, 0))
