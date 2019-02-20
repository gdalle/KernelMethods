import numpy as np


class Kernel:

    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __str__(self):
        return "Kernel {} - Parameters {}".format(self.name, str(self.params))


class LinearKernel(Kernel):

    def __init__(self):
        super().__init__("linear", {})

    def apply(self, X, Y):
        return X.dot(Y.T)


class GaussianKernel(Kernel):

    def __init__(self, sigma):
        super().__init__("gaussian", {"sigma": sigma})

    def apply(self, X, Y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if len(Y.shape) == 1:
            Y = Y.reshape((1, -1))
        norm_X = np.linalg.norm(X, axis=1).reshape((-1, 1))
        norm_Y = np.linalg.norm(Y, axis=1).reshape((-1, 1))
        scal = X.dot(Y.T)
        result = np.exp(
            -(norm_X**2 - 2*scal + norm_Y.T**2) /
            (2 * self.params["sigma"]**2)
        )
        return result


class CSVKernel:

    def __init__(self, path):
        self.ma
