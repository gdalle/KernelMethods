import numpy as np

def lin(X, Y):
    return X.dot(Y.T)

def gauss(sigma):
    def kernel(X, Y):
        if len(X.shape) == 1:
            X = X.reshape((1,-1))
        if len(Y.shape) == 1:
            Y = Y.reshape((1,-1))
        norm_X = np.linalg.norm(X, axis=1).reshape((-1,1))
        norm_Y = np.linalg.norm(Y, axis=1).reshape((-1,1))
        scal = X.dot(Y.T)
        result = np.exp(-(- 2*scal + norm_Y.T**2 + norm_X**2)/(2*sigma**2))
        return result
    return kernel