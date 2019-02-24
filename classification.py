import tqdm

import numpy as np
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
    solver="qp"
):
    M = len(K_list)
    if M == 1:
        return 1, kernel_svm(K_list[0], Ytr, lambd, solver)
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
            eta[i] -= steps[i] * grad_eta_i
        eta = projection_simplex(eta)
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
