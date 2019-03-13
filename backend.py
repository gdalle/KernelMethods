import os
import datetime
import itertools
import tqdm

import numpy as np
import scipy.sparse as sp
import pandas as pd

import matplotlib.pyplot as plt


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


def build_datasets(suffix="mat100", indices=[0, 1, 2]):
    datasets = []
    for k in indices:
        Xtr, Ytr = read_data_mat(dataset="tr" + str(k), suffix=suffix)
        Xte = read_data_mat(dataset="te" + str(k), suffix=suffix)
        datasets.append([Xtr, Ytr, Xte])
    return datasets


def center_K(K):
    n = K.shape[0]
    v = K.dot(np.ones(n) / n)
    UK = np.tile(v, (n, 1))
    UKU = K.sum() / n**2
    Kc = K - UK - UK.T + UKU
    return Kc


def predict_with_centering(K, Kx, alpha):
    n = K.shape[0]
    return np.sign(
        + Kx.dot(alpha)
        - alpha.sum() * Kx.sum(axis=1) / n
        - (alpha[:, None] * K).sum() / n
        + alpha.sum() * K.sum() / n**2
    )


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
        Ytr_pred, Yte_pred = sub_kernel.compute_prediction(
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
    kernels, lambdas,
    method="svm", solver="qp",
    kfold=5, shuffle=True,
    plot=False, result="best_lambdas"
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
                "dataset {} with {}".format(
                    kernel.dataset_index, kernel.name
                )
            )

    acc = acc_val.mean(axis=-1)

    if result == "best_kernel_lambda":
        (i_max, j_max) = np.unravel_index(np.argmax(acc), acc.shape)
        best_kernel = kernels[i_max]
        best_lambd = lambdas[j_max]
        return (best_kernel, best_lambd)
    elif result == "best_lambdas":
        j_max = acc.argmax(axis=1)
        best_lambdas = lambdas[j_max]
        return best_lambdas
    elif result == "all_stats":
        return acc_train, acc_val


def final_prediction(
    three_kernels, three_lambdas,
    method="svm", solver="qp"
):

    Ypred = []
    training_precisions = []

    for d in range(3):
        print("DATASET {}".format(d))

        # fit the predictor
        Ytr_pred, Yte_pred = three_kernels[d].compute_prediction(
            three_lambdas[d],
            method, solver
        )

        training_precisions.append(np.mean(three_kernels[d].Ytr == Ytr_pred))

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
        for d in range(3):
            file.write("Dataset " + str(three_kernels[d].dataset_index) + "\n")
            file.write("    " + "Kernel name: " + str(three_kernels[d].name) + "\n")
            file.write("    " + "Kernel params: " + str(three_kernels[d].params) + "\n")
            file.write("    " + "Lambda " + str(three_lambdas[d]) + "\n")
            file.write("    " + "Training precision " + str(training_precisions[d]) + "\n")
