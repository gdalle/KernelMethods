
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import OneHotEncoder


# In[2]:


np.random.seed(63)

states = np.arange(4)
observations = np.arange(4)

pi = np.random.random(4)
pi /= pi.sum()

A = np.random.random((4, 4))
i_zero = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
j_zero = [0, 2, 3, 0, 1, 3, 1, 2, 1, 2]
A[i_zero, j_zero] = 0
A /= A.sum(axis=1)[:, None]

B = np.random.random((4, 4))
B /= B.sum(axis=1)[:, None]


def generate(A, B, pi, T):
    x = []
    y = []
    for t in range(T):
        if t == 0:
            x.append(np.random.choice(states, p=pi))
        else:
            x.append(np.random.choice(states, p=A[x[-1]]))
        y.append(np.random.choice(observations, p=B[x[-1]]))
    x = np.array(x)
    y = np.array(y)
    return x, y


x, y = generate(A, B, pi, 100)

# print(A)
# print(B)


# In[3]:


def loglikelihood(y, A, B, pi):
    T = len(y)
    n_s = A.shape[0]

    norm = np.empty(T)

    alpha = np.empty((T, n_s))
    alpha[0] = pi * B[:, y[0]]
    norm[0] = 1/alpha[0].sum()
    for t in range(T-1):
        alpha[t+1] = B[:, y[t+1]] * A.T.dot(alpha[t])
        norm[t+1] = 1/alpha[t+1].sum()
        alpha[t+1] *= norm[t+1]

    loglike = alpha[T-1].sum() - np.log(norm).sum()
    return loglike


# In[4]:


def forward_backward(y, A, B, pi):
    T = len(y)
    n_s = A.shape[0]

    norm = np.empty(T)

    alpha = np.empty((T, n_s))
    alpha[0] = pi * B[:, y[0]]
    norm[0] = 1/alpha[0].sum()
    for t in range(T-1):
        alpha[t+1] = B[:, y[t+1]] * A.T.dot(alpha[t])
        norm[t+1] = 1/alpha[t+1].sum()
        alpha[t+1] *= norm[t+1]

    beta = np.empty((T, n_s))
    beta[T-1] = norm[T-1] * 1
    for t in range(T-2, -1, -1):
        beta[t] = A.dot(beta[t+1] * B[:, y[t+1]])
        beta[t] *= norm[t]

    return alpha, beta, norm


# In[5]:


def update(y, A, B, pi, alpha, beta, norm):
    T = len(y)
    n_s = A.shape[0]
    n_o = B.shape[1]

    new_pi = np.empty(n_s)
    new_A = np.empty((n_s, n_s))
    new_B = np.empty((n_s, n_o))

    for i in range(n_s):
        new_pi[i] = alpha[0, i] * beta[0, i] / alpha[0].dot(beta[0])

    for i in range(n_s):
        for j in range(n_s):
            new_A[i, j] = sum(
                [alpha[t, i] * A[i, j] * B[j, y[t+1]] * beta[t+1, j] for t in range(T-1)]
            ) / sum(
                [alpha[t, i] * beta[t, i] / norm[t] for t in range(T-1)]
            )
    for j in range(n_s):
        for k in range(n_o):
            new_B[j, k] = sum(
                [alpha[t, j] * beta[t, j] / norm[t] for t in range(T) if y[t] == k]
            ) / sum(
                [alpha[t, j] * beta[t, j] / norm[t] for t in range(T)]
            )

    return new_A, new_B, new_pi


# In[6]:


def baum_welch_step(y, A, B, pi):
    alpha, beta, norm = forward_backward(y, A, B, pi)
    return update(y, A, B, pi, alpha, beta, norm)


def baum_welch(y, A0, B0, pi0, iterations=100):
    A, B, pi = A0, B0, pi0
    for it in tqdm.trange(iterations):
        A, B, pi = baum_welch_step(y, A, B, pi)
    return A, B, pi


# In[7]:


def baum_welch_step_dataset2(dataset, A, B, pi):

    n_s = A.shape[0]
    n_o = B.shape[1]

    new_A_num = np.zeros((n_s, n_s))
    new_A_den = np.zeros((n_s, n_s))
    new_B_num = np.zeros((n_s, n_o))
    new_B_den = np.zeros((n_s, n_o))
    new_pi_num = np.zeros(n_s)
    new_pi_den = np.zeros(n_s)

    for y in dataset:
        T = len(y)
        alpha, beta, norm = forward_backward(y, A, B, pi)

        for i in range(n_s):
            new_pi_num[i] += alpha[t, i] * beta[t, i] / norm[t]
            new_pi_den[i] += 1

        for i in range(n_s):
            for j in range(n_s):
                new_A_num[i, j] += sum(
                    [alpha[t, i] * A[i, j] * B[j, y[t+1]] * beta[t+1, j] for t in range(T-1)]
                )
                new_A_den[i, j] += sum(
                    [alpha[t, i] * beta[t, i] / norm[t] for t in range(T-1)]
                )
        for j in range(n_s):
            for k in range(n_o):
                new_B_num[j, k] += sum(
                    [alpha[t, j] * beta[t, j] / norm[t] for t in range(T-1) if y[t] == k]
                )
                new_B_den[j, k] += sum(
                    [alpha[t, j] * beta[t, j] / norm[t] for t in range(T-1)]
                )

    return new_A_num/new_A_den, new_B_num/new_B_den, new_pi_num/new_pi_den


def baum_welch_dataset2(dataset, A0, B0, pi0, iterations=100):
    A, B, pi = A0, B0, pi0
    for it in tqdm.trange(iterations):
        A, B, pi = baum_welch_step_dataset2(dataset, A, B, pi)
    return A, B, pi


# In[8]:


def baum_welch_step_dataset(dataset, A, B, pi):

    n_s = A.shape[0]
    n_o = B.shape[1]

    new_A_num = np.zeros((n_s, n_s))
    new_A_den = np.zeros((n_s, n_s))
    new_B_num = np.zeros((n_s, n_o))
    new_B_den = np.zeros((n_s, n_o))
    new_pi_num = np.zeros(n_s)
    new_pi_den = np.zeros(n_s)

    for y in dataset:
        T = len(y)
        alpha, beta, norm = forward_backward(y, A, B, pi)

        new_A_num += (
            alpha[:-1, :, None] * A[None, :, :] * B[:, y[1:]].T[:, None, :] * beta[1:][:, None, :]
        ).sum(axis=0)
        new_A_den += (
            alpha[:-1, :, None] * beta[:-1, :, None] / norm[:-1, None, None]
        ).sum(axis=0)

        onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        y_binary = onehot_encoder.fit_transform(y[:, None]).astype(int)
        new_B_num += (
            y_binary[:-1, None, :] * alpha[:-1, :, None] *
            beta[:-1, :, None] / norm[:-1, None, None]
        ).sum(axis=0)
        new_B_den += (
            alpha[:-1, :, None] * beta[:-1, :, None] / norm[:-1, None, None]
        ).sum(axis=0)

        new_pi_num += alpha[0, :] * beta[0, :] / norm[0]
        new_pi_den += 1

    return new_A_num/new_A_den, new_B_num/new_B_den, new_pi_num/new_pi_den


def baum_welch_dataset(dataset, A0, B0, pi0, iterations=100):
    A, B, pi = A0, B0, pi0
    for it in tqdm.trange(iterations):
        A, B, pi = baum_welch_step_dataset(dataset, A, B, pi)
    return A, B, pi


# In[9]:


def baum_welch_dna(dataset, iterations=100):
    np.random.seed(63)
    A_guess = np.array([
        [0,   1,   0,   0, ],  # exon 1 => exon 2
        [0,   0,   1,   0, ],  # exon 2 => exon 3
        [0.5,   0,   0, 0.5, ],  # exon 3 => exon 1 or intron
        [0.5,   0,   0, 0.5, ],
    ])
    A0 = np.random.random((4, 4)) * A_guess
    A0 /= A0.sum(axis=1)[:, None]

    B0 = np.random.random((4, 4)) / 4
    B0 /= B0.sum(axis=1)[:, None]

    pi0 = np.random.random(4)
    pi0 /= pi0.sum()

    print("\nInitial values")
    print(A0)
    print(B0)
    print(pi0)

    A, B, pi = baum_welch_dataset(dataset, A0, B0, pi0, iterations)

    print("Estimation")
    print(A)
    print(B)
    print(pi)

    return A, B, pi


# In[10]:


def translate(dna_string):
    dna_string = dna_string.replace("A", "0")
    dna_string = dna_string.replace("T", "1")
    dna_string = dna_string.replace("G", "2")
    dna_string = dna_string.replace("C", "3")
    seq = list(dna_string)
    return np.array(seq).astype(int)


# In[12]:


n_samples = 2000
iterations = 100
for d in [0, 1]:
    dataset_string = pd.read_csv("data/Xtr{}.csv".format(d), index_col=0)
    dataset = [
        translate(dna_string)
        for dna_string in dataset_string.values[:, 0]
    ]
    new_A, new_B, new_pi = baum_welch_dna(dataset[:n_samples], iterations)
    pd.DataFrame(new_A).to_csv("data/HMM_{}_A.csv".format(d))
    pd.DataFrame(new_B).to_csv("data/HMM_{}_B.csv".format(d))
    pd.DataFrame(new_pi).to_csv("data/HMM_{}_pi.csv".format(d))
