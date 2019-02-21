import copy

import numpy as np
import pandas as pd

import backend


class Kernel:

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.dataset_index = -1
        self.need_dataset = True
        self.train_matrix = None
        self.text_matrix = None
        self.Ytr = None
    
    def apply(self, X, Y):
        """ Return the matrix K(X, Y) """
        raise NotImplementedError()
    
    def load(self, suffix, indices):
        """Given a data suffix, computes the kernel matrices for each dataset"""
        kernels = []
        datasets = backend.build_datasets(suffix, indices)
        for d, data in enumerate(datasets):
            self.dataset_index = d
            Xtr, Ytr, Xte = data
            k = copy.deepcopy(self)
            if self.need_dataset:
                k.train_matrix = k.apply(Xtr, Xtr)
                k.test_matrix = k.apply(Xte, Xtr)
            else:
                k.train_matrix = Xtr
                k.test_matrix = Xte
            k.Ytr = Ytr
            kernels.append(k)
    
        return kernels
       
    def split_train_validation(self, train_idx, val_idx):
        """ Splits train_matrix for the cross_validation into train and test matrices """
        k = Kernel("Sub-kernel", {})
        k.train_matrix = self.train_matrix[train_idx, :][:, train_idx]
        k.test_matrix  = self.train_matrix[val_idx, :][:, train_idx]
        k.Ytr = self.Ytr[train_idx]
        return k
    
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


class CSVKernel(Kernel):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.need_dataset = False