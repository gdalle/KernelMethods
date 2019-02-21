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


def build_datasets(suffix="mat100", indices=[0,1,2]):
    datasets = []
    for k in indices:
        Xtr, Ytr = read_data_mat(dataset="tr" + str(k), suffix=suffix)
        Xte = read_data_mat("te" + str(k), suffix=suffix)
        datasets.append([Xtr, Ytr, Xte])
    return datasets


def compute_prediction(
    kernel, lambd,
    method="svm", solver="qp",
    center=True
):
    Ytr = kernel.Ytr
    n = Ytr.shape[0]
    n_plus = np.sum(Ytr == 1)
    n_minus = np.sum(Ytr == -1)

    K = kernel.train_matrix

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

    def predictor(Kx):
        f = Kx.dot(alpha)
        if center:
            f += (
                - alpha.sum() * (gamma.reshape((1, -1)) * Kx).sum(axis=1)
                - alpha.reshape((1, -1)).dot(K).dot(gamma)
                + alpha.sum() * gamma.reshape((1, -1)).dot(K).dot(gamma)
            )
        return np.sign(f)

    K_test = kernel.test_matrix

    return predictor(K), predictor(K_test)


def cross_validate(
    kernel, lambd,
    method="svm", solver="qp",
    kfold=5, shuffle=True
):
    acc_train, acc_val = np.zeros(kfold), np.zeros(kfold)

    # jointly shuffle input datasets X, Y
    n = kernel.Ytr.shape[0]
    if shuffle:
        perm = np.random.permutation(n)
        kernel = kernel.split_train_validation(perm, [])
    idx = np.arange(n)
    for k in range(kfold):
        # split the datasets
        val_idx = idx[k::kfold]
        train_idx = np.delete(idx, val_idx)

        sub_kernel = kernel.split_train_validation(train_idx, val_idx)
        Ytr = sub_kernel.Ytr
        Yte = kernel.Ytr[val_idx]

        # fit the predictor
        Ytr_pred, Yte_pred = compute_prediction(
            sub_kernel,
            lambd, method, solver
        )

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

    kernels_lambdas = list(itertools.product(kernels, lambdas))
    acc_train = np.empty((len(kernels), len(lambdas), kfold))
    acc_val = np.empty((len(kernels), len(lambdas), kfold))

    for i, kernel in enumerate(kernels):
        for j in tqdm.trange(
            len(lambdas),
            desc="Tuning lambda on dataset {} with kernel {} and params {}".format(
                kernel.dataset_index, kernel.name, kernel.params
            )
        ):
            lambd = lambdas[j]
            acc_train[i, j], acc_val[i, j] = cross_validate(
                kernel, lambd,
                method=method, solver=solver,
                kfold=kfold, shuffle=shuffle
            )

        if plot:
            plot_CV_results(
                acc_train[i], acc_val[i],
                lambdas, "lambda",
                "dataset {} with kernel {} and params {}".format(
                    kernel.dataset_index, kernel.name, kernel.params
                )
            )

    acc = acc_val.mean(axis=-1)
    j_max = acc.argmax(axis=1)
    best_lambdas = lambdas[j_max]

    if all_stats:
        return best_lambdas, acc_train, acc_val
    else:
        return best_lambdas


def final_prediction(
    suffix,
    kernels, best_lambdas,
    method="svm", solver="qp"
):

    Ypred = []
    training_precisions = []

    for i in range(len(kernels)):
        d = kernels[i].dataset_index
        print("DATASET {}".format(d))

        # fit the predictor
        Ytr_pred, Yte_pred = compute_prediction(
            kernels[i], best_lambdas[i],
            method, solver
        )

        training_precisions.append(np.mean(kernels[i].Ytr == Ytr_pred))

        Ypred.extend(list(((Yte_pred + 1) / 2).astype(int)))

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
        for i in range(len(kernels)):
            file.write("Dataset " + str(kernels[i].dataset_index) + "\n")
            file.write("    " + str(kernels[i]) + "\n")
            file.write("    " + "Lambda " + str(best_lambdas[i]) + "\n")
            file.write("    " + "Training precision " + str(training_precisions[i]) + "\n")
