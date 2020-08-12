import pickle

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softMax(x):
    yi = []
    sum = np.sum(np.exp(x))
    for i in x:
        yi.append(np.exp(i) / sum)
    return yi


def ReLU(x):
    if x < 0:
        return 0
    return x


class Model:
    alpha = 0.009
    Neuron1 = 22
    Neuron2 = 22
    out = 3
    In = 1
    Wx = np.random.random((Neuron1, In + 1))
    Wy = np.random.random((Neuron2, Neuron1 + 1))
    Wz = np.random.random((out, Neuron2 + 1))

    # Wz = np.random.random((1, Neuron2 + 1))

    def fit(self):
        for i in range(9000):
            self.learn(X, T)
            if i % 10 == 0:
                print(f"finished:{i}")

    def learn(self, k, t):
        SDWx = []
        SDWy = []
        SDWz = []
        num = len(self.Wy.T)
        for j in range(len(X)):
            out = self.func(k[j])
            Q, Iq, z, Iz, y, Iy, = out[0], out[1], out[2], out[3], out[4], out[5]
            tt = np.reshape(t[j], (1, len(t[j])))
            deltaWz = (Q - tt.T) * np.hstack((z.reshape(-1), [1.0]))
            SDWz.append(deltaWz)
            tempZ = np.delete(self.Wz.T, len(self.Wz[0]) - 1, 0)
            deltaZ = np.dot(tempZ, (Q - tt.T))
            deltaWy = deltaZ * (z * (1 - z)) * np.hstack((y.reshape(-1), 1.0))
            SDWy.append(deltaWy)
            # tempy = np.delete(self.Wy.T, len(self.Wy[0]) - 1, 0)
            deltaY = np.delete(self.Wy.T, num - 1, 0).T * (z * (1 - z))
            deltaWx = np.dot(deltaY.T, deltaZ) * (y * (1 - y)) * np.hstack((k[j], 1.0))
            SDWx.append(deltaWx)
            # exit(1)
            self.Wy -= self.alpha * (np.mean(np.array(SDWy), axis=0))
            self.Wx -= self.alpha * (np.mean(np.array(SDWx), axis=0))
            self.Wz -= self.alpha * (np.mean(np.array(SDWz), axis=0))

    def func(self, inPut):
        Iy = np.dot(self.Wx, np.array([[inPut], [1]]))
        y = sigmoid(Iy)
        Iz = np.dot(self.Wy, np.vstack((y, np.array([[1]]))))
        z = sigmoid(Iz)
        Iq = np.dot(self.Wz, np.vstack((z, np.array([[1]]))))
        Q = softMax(Iq)
        return np.array(Q), Iq, z, Iz, y, Iy


with open('model.pickle', mode='rb') as p:  # with構文でファイルパスとバイナリ読み来みモードを設定
    y = pickle.load(p)
with open('x.pickle', mode='rb') as p:  # with構文でファイルパスとバイナリ読み来みモードを設定
    x = pickle.load(p)

T = np.zeros((150, 3))
j = 0
for i in y:
    T[j][i] = 1
    j += 1

T = np.array(T)
X = np.array(x)
m = Model()

m.fit()
plt.figure()
plt.plot(X, y, 'o')

TEST = X
re = []
for i in TEST:
    pre = (m.func(i)[0])
    print(pre)
    print(f"入力{i}:予測:{np.argmax(pre)},確率:{np.max(pre)}")
    re.append(np.argmax(pre))
plt.plot(TEST, re, marker="*", color='r', linestyle='None', markersize=15)
plt.figure()
a = np.arange(0, 15, 0.1)
b = []
for i in a:
    b.append(np.ravel(m.func(i)[0]))
plt.plot(a, b)
plt.show()
print(m.Wx, m.Wy, m.Wz)
