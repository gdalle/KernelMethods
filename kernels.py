import copy

import numpy as np
import pandas as pd

import backend
from classification import kernel_svm, multiple_kernel_svm, kernel_logreg, kernel_boosting


class Kernel:

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.dataset_index = -1
        self.csv_storage = "data"  # or "features" or "gram"
        self.train_matrix = None # K(Xtr, Xtr)
        self.test_matrix = None # K(Xte, Xtr)
        self.test_test_matrix = None # K(Xte, Xte)
        self.Ytr = None

    def __str__(self):
        return "Kernel {} - Parameters {}".format(self.name, str(self.params))

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
            if self.csv_storage == "data":
                m = Xtr.mean(axis=0)
                s = Xtr.std(axis=0)
                s[s < 1e-10] = 1
                Xtr = (Xtr - m) / s
                Xte = (Xte - m) / s
                k.train_matrix = k.apply(Xtr, Xtr)
                k.test_matrix = k.apply(Xte, Xtr)
            elif self.csv_storage == "features":
                m = Xtr.mean(axis=0)
                s = Xtr.std(axis=0)
                s[s < 1e-10] = 1
                Xtr = (Xtr - m) / s
                Xte = (Xte - m) / s
                k.train_matrix = LinearKernel().apply(Xtr, Xtr)
                k.test_matrix = LinearKernel().apply(Xte, Xtr)
            elif self.csv_storage == "gram":
                k.train_matrix = Xtr
                k.test_matrix = Xte
            k.Ytr = Ytr
            k.params["suffix"] = suffix
            kernels.append(k)

        return kernels

    def split_train_validation(self, train_idx, val_idx):
        """ Splits train_matrix for the cross_validation into train and test matrices """
        k = Kernel("Sub-kernel", {})
        k.train_matrix = self.train_matrix[train_idx, :][:, train_idx]
        k.test_matrix = self.train_matrix[val_idx, :][:, train_idx]
        k.Ytr = self.Ytr[train_idx]
        return k

    def compute_prediction(
        self, lambd,
        method="svm", solver="qp",
        center=True
    ):
        Ytr = self.Ytr

        K = self.train_matrix
        Kc = backend.center_K(K, Ytr) if center else K
        Kc += 1e-7 * np.eye(Ytr.shape[0])

        if method == "svm":
            alpha = kernel_svm(Kc, Ytr, lambd, solver=solver)
        elif method == "logreg":
            alpha = kernel_logreg(Kc, Ytr, lambd)

        def predictor(Kx):
            f = Kx.dot(alpha)
            if center:
                f += backend.correct_prediction_centering(K, Kx, Ytr, alpha)
            return np.sign(f)

        K_test = self.test_matrix

        return predictor(K), predictor(K_test)


class LinearKernel(Kernel):

    def __init__(self):
        super().__init__("LinearKernel", {})

    def apply(self, X, Y):
        return X.dot(Y.T)


class GaussianKernel(Kernel):

    def __init__(self, sigma):
        super().__init__("GaussianKernel", {"sigma": sigma})

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

class CauchyKernel(Kernel):

    def __init__(self, sigma):
        super().__init__("CauchyKernel", {"sigma": sigma})

    def apply(self, X, Y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if len(Y.shape) == 1:
            Y = Y.reshape((1, -1))
        norm_X = np.linalg.norm(X, axis=1).reshape((-1, 1))
        norm_Y = np.linalg.norm(Y, axis=1).reshape((-1, 1))
        scal = X.dot(Y.T)
        result = 1 / (1 +
            (norm_X**2 - 2*scal + norm_Y.T**2) * self.params["sigma"]**2
        )
        return result


class GramCSVKernel(Kernel):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.csv_storage = "gram"


class FeatureCSVKernel(Kernel):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.csv_storage = "features"


class MultipleKernel(Kernel):

    def __init__(self, loaded_kernels, grad_step=1, iterations=10, entropic=0):
        params = {
            "kernel_names": [kernel.name for kernel in loaded_kernels],
            "kernel_params": [kernel.params for kernel in loaded_kernels],
            "grad_step": grad_step,
            "iterations": iterations,
            "entropic": entropic,
        }
        super().__init__("MultipleKernel", params)
        self.grad_step = grad_step
        self.iterations = iterations
        self.entropic = entropic
        self.dataset_index = loaded_kernels[0].dataset_index
        self.Ytr = loaded_kernels[0].Ytr
        self.M = len(loaded_kernels)
        self.kernels = loaded_kernels
        self.train_list = [kernel.train_matrix for kernel in self.kernels]
        self.test_list = [kernel.test_matrix for kernel in self.kernels]
        # Temporary scaling until I learn how to center this shit
        for i in range(self.M):
            scale = np.abs(self.train_list[i]).mean()
            self.train_list[i] /= scale
            self.test_list[i] /= scale

    def split_train_validation(self, train_idx, val_idx):
        """ Splits train_matrix for the cross_validation into train and test matrices """
        subkernels = []
        for kernel in self.kernels:
            k = Kernel("Sub-kernel", {})
            k.train_matrix = kernel.train_matrix[train_idx, :][:, train_idx]
            k.test_matrix = kernel.train_matrix[val_idx, :][:, train_idx]
            k.Ytr = kernel.Ytr[train_idx]
            subkernels.append(k)
        return MultipleKernel(
            subkernels,
            grad_step=self.grad_step, iterations=self.iterations,
            entropic=self.entropic
        )

    def compute_prediction(
        self, lambd,
        method="svm", solver="qp",
    ):
        eta, alpha = multiple_kernel_svm(
            self.train_list, self.Ytr,
            lambd,
            grad_step=self.grad_step, iterations=self.iterations,
            entropic=self.entropic,
            solver=solver
        )
        self.params["kernel_weights"] = eta
        train_matrix = sum(eta[i] * self.train_list[i] for i in range(self.M))
        test_matrix = sum(eta[i] * self.test_list[i] for i in range(self.M))

        def predictor(Kx):
            return np.sign(Kx.dot(alpha))

        return predictor(train_matrix), predictor(test_matrix)

class BoostingKernel(Kernel):
    
    def __init__(self, vector_kernel=LinearKernel(), iterations=20):
        self.vector_kernel = vector_kernel
        self.iterations = iterations
        self.csv_storage = "features"
        params = {
            "vector_kernel": vector_kernel.name,
            "iterations": iterations,
        }
        super().__init__("BoostingKernel", params)
    
    def load(self, suffix, indices):
        """Given a data suffix, computes the kernel matrices for each dataset"""
        kernels = []
        datasets = backend.build_datasets(suffix, indices)
        for d, data in enumerate(datasets):
            self.dataset_index = d
            Xtr, Ytr, Xte = data
            k = copy.deepcopy(self)
            m = Xtr.mean(axis=0)
            s = Xtr.std(axis=0)
            s[s < 1e-10] = 1
            Xtr = (Xtr - m) / s
            Xte = (Xte - m) / s
            Ktr, Kte = kernel_boosting(self.vector_kernel, Xtr, Ytr, Xte, self.iterations)
            k.train_matrix = Ktr
            k.test_matrix = Kte
            k.Ytr = Ytr
            k.params["suffix"] = suffix
            kernels.append(k)
        return kernels
