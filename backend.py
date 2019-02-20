import os
import datetime
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


def compute_predictor(
    Xtr, Ytr,
    kernel, lambd,
    method="svm", solver="qp"
):
    n = len(Xtr)

    K = kernel.apply(Xtr, Xtr) + 1e-10 * np.eye(n)

    if method == "svm":
        alpha = kernel_svm(K, Ytr, lambd, solver=solver)

    elif method == "logreg":
        alpha = kernel_logreg(K, Ytr, lambd)

    def predictor(x_new):
        return np.sign(alpha.dot(kernel.apply(Xtr, x_new)))
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


def plot_CV_results(acc_train, acc_val, param_range, param_name):
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
    plt.title("Cross-validation results")
    plt.show()


def final_prediction(
    dataset,
    best_kernel, best_lambd,
    method="svm", solver="qp"
):
    Ypred = []
    training_precisions = []

    for d in [0, 1, 2]:
        print("DATASET {}".format(d+1))

        Xtr, Ytr, Xte = dataset[d]

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
        file.write("PREDICTION LOG : {}\n".format(date))
        for d in range(3):
            file.write("Dataset " + str(d+1) + "\n")
            file.write("    " + str(best_kernel[d]) + "\n")
            file.write("    " + "Lambda " + str(best_lambd[d]) + "\n")
            file.write("    " + "Training precision " + str(training_precisions[d]) + "\n")
