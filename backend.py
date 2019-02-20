import os
import datetime
import itertools
import tqdm

import numpy as np
import scipy.sparse as sp
import pandas as pd

import matplotlib.pyplot as plt

from classification import kernel_svm, kernel_logreg


def read_data_mat(dataset="tr0", suffix="mat100"):
    folder = "data"
    features_file = "X" + dataset + "_" + suffix + ".csv"
    labels_file = "Y" + dataset + ".csv"

    X = pd.read_csv(
        os.path.join(folder, features_file),
        sep=" ",
        header=None
    )
    if "te" in dataset:
        return np.array(X)

    elif "tr" in dataset:
        Y = pd.read_csv(
            os.path.join(folder, labels_file),
            sep=",",
            index_col=0,
        )
        return np.array(X), 2 * np.array(Y.iloc[:, 0]) - 1


def build_datasets(suffix="mat100"):
    datasets = []
    for k in [0, 1, 2]:
        Xtr, Ytr = read_data_mat(dataset="tr" + str(k), suffix=suffix)
        Xte = read_data_mat("te" + str(k), suffix=suffix)
        datasets.append([Xtr, Ytr, Xte])
    return datasets


def compute_predictor(
    Xtr, Ytr,
    kernel, lambd,
    method="svm", solver="qp",
    center=True
):
    n = len(Xtr)
    n_plus = np.sum(Ytr == 1)
    n_minus = np.sum(Ytr == -1)

    K = kernel.apply(Xtr, Xtr)

    if center:
        gamma = (
            (Ytr == 1).astype(int) * (0.5 / n_plus) +
            (Ytr == -1).astype(int) * (0.5 / n_minus)
        )
        u = K.dot(gamma)
        v = np.tile(u, (n, 1))
        w = gamma.dot(u)
        Kc = K - v - v.T + w
    else:
        Kc = K
    Kc += 1e-10 * np.eye(n)

    if method == "svm":
        alpha = kernel_svm(Kc, Ytr, lambd, solver=solver)

    elif method == "logreg":
        alpha = kernel_logreg(Kc, Ytr, lambd)

    def predictor(x_new):
        Kx = kernel.apply(x_new, Xtr)
        f = Kx.dot(alpha)
        if center:
            f += (
                - alpha.sum() * (gamma.reshape((1, -1)) * Kx).sum(axis=1)
                - alpha.reshape((1, -1)).dot(K).dot(gamma)
                + alpha.sum() * gamma.reshape((1, -1)).dot(K).dot(gamma)
            )
        return np.sign(f)

    return predictor


def cross_validate(
    X, Y,
    kernel, lambd,
    method="svm", solver="qp",
    kfold=5, shuffle=True
):
    acc_train, acc_val = np.zeros(kfold), np.zeros(kfold)

    # jointly shuffle input datasets X, Y
    n = X.shape[0]
    if shuffle:
        perm = np.random.permutation(n)
        X, Y = X[perm], Y[perm]
    idx = np.arange(n)
    for k in range(kfold):
        # split the datasets
        val_idx = idx[k::kfold]
        train_idx = np.delete(idx, val_idx)

        Xtr = X[train_idx]
        Ytr = Y[train_idx]
        Xte = X[val_idx]
        Yte = Y[val_idx]

        # fit the predictor
        f = compute_predictor(
            Xtr, Ytr,
            kernel, lambd,
            method, solver
        )

        Ytr_pred = f(Xtr).reshape(-1)
        Yte_pred = f(Xte).reshape(-1)

        # compute precision
        acc_train[k] = np.mean(Ytr == Ytr_pred)
        acc_val[k] = np.mean(Yte == Yte_pred)
    return acc_train, acc_val


def lineplotCI(x, y, train=True, log=False):
    mean = y.mean(axis=1)
    up = mean + y.std(axis=1)
    low = mean - y.std(axis=1)
    if train:
        label = "train"
        color = "r"
    else:
        label = "val"
        color = "b"
    if log:
        plt.xscale("log")
    plt.plot(x, mean, lw=2, color=color, alpha=1, label=label)
    plt.fill_between(x, low, up, color=color, alpha=0.2)
    for i, param in enumerate(x):
        plt.scatter([param] * y.shape[1], y[i], color=color, alpha=0.5)


def plot_CV_results(acc_train, acc_val, param_range, param_name, title):
    plt.figure()
    lineplotCI(
        param_range, acc_train,
        train=True, log=True
    )
    lineplotCI(
        param_range, acc_val,
        train=False, log=True
    )
    plt.legend()
    plt.xlabel("Value of {}".format(param_name))
    plt.ylabel("Accuracy")
    plt.title("CV - {}".format(title))
    plt.show()


def tune_parameters(
    suffix,
    kernels, lambdas,
    method="svm", solver="qp",
    kfold=5, shuffle=True,
    plot=False, all_stats=False
):
    datasets = build_datasets(suffix)

    kernels_lambdas = list(itertools.product(kernels, lambdas))
    acc_train = np.empty((3, len(kernels), len(lambdas), kfold))
    acc_val = np.empty((3, len(kernels), len(lambdas), kfold))

    best_kernels = [None for d in range(3)]
    best_lambdas = [None for d in range(3)]

    for d, data in enumerate(datasets):
        Xtr, Ytr, Xte = data
        for i, kernel in enumerate(kernels):
            for j in tqdm.trange(
                len(lambdas),
                desc="Tuning lambda on dataset {} with kernel {} and params {}".format(
                    d, kernel.name, kernel.params
                )
            ):
                lambd = lambdas[j]
                acc_train[d, i, j], acc_val[d, i, j] = cross_validate(
                    Xtr, Ytr,
                    kernel, lambd,
                    method=method, solver=solver,
                    kfold=kfold, shuffle=shuffle
                )

            if plot:
                plot_CV_results(
                    acc_train[d, i], acc_val[d, i],
                    lambdas, "lambda",
                    "dataset {} with kernel {} and params {}".format(
                        d, kernel.name, kernel.params
                    )
                )

        acc = acc_val[d].mean(axis=-1)
        i_max, j_max = np.unravel_index(
            np.ndarray.argmax(acc), acc.shape
        )
        best_kernels[d] = kernels[i_max]
        best_lambdas[d] = lambdas[j_max]

    if all_stats:
        return best_kernels, best_lambdas, acc_train, acc_val
    else:
        return best_kernels, best_lambdas


def final_prediction(
    suffix,
    best_kernel, best_lambd,
    method="svm", solver="qp"
):
    datasets = build_datasets(suffix=suffix)

    Ypred = []
    training_precisions = []

    for d in [0, 1, 2]:
        print("DATASET {}".format(d))

        Xtr, Ytr, Xte = datasets[d]

        f = compute_predictor(
            Xtr, Ytr,
            best_kernel[d], best_lambd[d],
            method, solver
        )

        training_precisions.append(np.mean(Ytr == f(Xtr)))

        Yte = f(Xte)
        Ypred.extend(list(((Yte + 1) / 2).astype(int)))

    Ypred = pd.Series(
        index=np.arange(len(Ypred)),
        data=Ypred
    )
    Ypred.index.name = "Id"
    Ypred.name = "Bound"

    date = str(datetime.datetime.now())
    date = date[5:-10]
    date2 = "_".join(date.split())
    date2 = "-".join(date2.split(":"))

    Ypred.to_csv(
        os.path.join("predictions", date2 + "__Ypred.csv"),
        header=True
    )

    with open(os.path.join("predictions", date2 + "__params.txt"), "w") as file:
        file.write("PREDICTION LOG - {}\n".format(date))
        file.write("Suffix: {}\n".format(suffix))
        for d in range(3):
            file.write("Dataset " + str(d) + "\n")
            file.write("    " + str(best_kernel[d]) + "\n")
            file.write("    " + "Lambda " + str(best_lambd[d]) + "\n")
            file.write("    " + "Training precision " + str(training_precisions[d]) + "\n")
