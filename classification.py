import tqdm

import numpy as np
from scipy import linalg
import scipy.sparse as sp

import cvxpy as cp
import osqp
from qpsolvers import solve_qp


def kernel_svm(K, Ytr, lambd, solver="qp"):
    n = K.shape[0]

    if solver == "qp":
        P = K
        q = - Ytr.astype(float)
        # Sparse G
        G = sp.vstack([
            -sp.diags(Ytr),
            sp.diags(Ytr)
        ]).tocsc().astype(float)
        h = np.hstack([
            np.zeros(n),
            np.ones(n) / (2 * lambd * n)
        ]).astype(float)

        alpha_opt = solve_qp(P=P, q=q, G=G, h=h, solver="cvxopt")

    elif solver == "cvxpy":
        alpha = cp.Variable(n)

        constraints = [
            cp.multiply(Ytr, alpha) >= np.zeros(n),
            cp.multiply(Ytr, alpha) <= np.ones(n) / (2 * lambd * n)
        ]

        objective = cp.Minimize(
            - 2 * (Ytr * alpha)
            + cp.quad_form(alpha, K)
        )

        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.OSQP, verbose=False)
        alpha_opt = alpha.value

    return alpha_opt


def projection_simplex(y):
    D = len(y)
    u = np.sort(y)[::-1]
    U = np.cumsum(u)
    rho = None
    for j in range(1, D+1):
        if u[j-1] + (1 - U[j-1]) / j > 0:
            rho = j
    lambd = (1 / rho) * (1 - U[rho-1])
    return np.clip(y + lambd, a_min=0, a_max=None)


def multiple_kernel_svm(
    K_list, Ytr,
    lambd,
    grad_step=1, iterations=10,
    entropic=0,
    solver="qp"
):
    M = len(K_list)
    if M == 1:
        return np.ones(1), kernel_svm(K_list[0], Ytr, lambd, solver)
    eta = np.ones(M) / M
    if type(grad_step) == type(lambda x: 1):
        steps = [grad_step(it) for it in range(iterations)]
    else:
        steps = grad_step * np.ones(iterations)
    for it in range(iterations):
        K = sum(eta[i] * K_list[i] for i in range(M))
        alpha = kernel_svm(K, Ytr, lambd, solver)
        for i in range(M):
            grad_eta_i = - lambd * alpha.reshape((1, -1)).dot(K_list[i]).dot(alpha)
            grad_eta_i += entropic * np.log(eta[i]+1e-10)  # Entropic regularization
            eta[i] -= steps[it] * grad_eta_i
        eta = projection_simplex(eta)
        print("Eta", eta)
    K = sum(eta[i] * K_list[i] for i in range(M))
    alpha = kernel_svm(K, Ytr, lambd, solver)
    return eta, alpha


def sigma(x):
    return 1 / (1 + np.exp(-x))


def solve_WKRR(K, w, z, lambd):
    n = K.shape[0]
    Wsqrt = np.sqrt(np.diag(w))
    b = Wsqrt.dot(z)
    A = Wsqrt.dot(K).dot(Wsqrt) + lambd * n * np.eye(n)
    alpha = Wsqrt.dot(np.linalg.solve(A, b))
    return alpha


def kernel_logreg(K, Ytr, lambd, iterations=10):
    n = K.shape[0]
    alpha = np.zeros(n)
    for it in tqdm.trange(iterations):
        m = K.dot(alpha)
        p = - sigma(-Ytr * m)
        w = sigma(m) * sigma(-m)
        z = m + Ytr / sigma(-Ytr * m)
        alpha = solve_WKRR(K, w, z, lambd)
    return alpha


def base_kernel_learner(K0, D, Xtr, Ytr, Xte):
    # A base kernel is basically K_w = w.dot(w.T) with w a vector
    # The kernel is thus K(x, y) = (x.w) (y.w)
    A = K0.test_matrix.T
    B = (Ytr * Ytr.T) * D
    K = K0.test_test_matrix
    m = K.shape[0]-1
    eg = linalg.eigh(A.T.dot(B).dot(A), K, eigvals=(m, m))
    # Generalized eigenvector with the largest eigenvalue
    v = eg[1][:, 0]
    w = (v * Xte.T).T.sum(axis=0)
    w = (w / np.linalg.norm(w))
    return w


def kernel_boosting(K0, Xtr, Ytr, Xte, steps):
    """
    Xtr and Xte are vectorized representations of the dataset
    K0 is an empty vector kernel
    """
    K0.train_matrix = K0.apply(Xtr, Xtr) + 1e-10*np.eye(Xtr.shape[0])
    K0.test_matrix = K0.apply(Xte, Xtr)
    K0.test_test_matrix = K0.apply(Xte, Xte) + 1e-10*np.eye(Xte.shape[0])
    Ktr = 0
    Kte = 0
    Ytr = Ytr.reshape((-1, 1))
    for t in tqdm.tqdm(range(steps)):
        # Compute a distribution over the weights
        D = np.exp(-(Ytr * Ytr.T) * Ktr)  # ExpLoss
        # Compute the next update for train and test
        w = base_kernel_learner(K0, D, Xtr, Ytr, Xte)
        Xtrw = Xtr.dot(w).reshape((-1, 1))
        Ktr_t = Xtrw.dot(Xtrw.T)
        Xtew = Xte.dot(w).reshape((-1, 1))
        Kte_t = Xtew.dot(Xtrw.T)
        # Compute the update rate
        Sp = (Ytr * Ytr.T) * Ktr_t > 0
        Sm = (Ytr * Ytr.T) * Ktr_t < 0
        Wp = np.sum(D[Sp] * np.abs(Ktr_t[Sp]))
        Wm = np.sum(D[Sm] * np.abs(Ktr_t[Sm]))
        alpha = 0.5 * np.log(Wp / Wm)
        # Execute the update
        Ktr += alpha * Ktr_t + 1e-10*np.eye(Xtr.shape[0])
        Kte += alpha * Kte_t
        M = Ktr.max()
        Ktr /= M
        Kte /= M
        #print(linalg.eigh(Ktr, eigvals=(0,0))[0])
    return Ktr, Kte
