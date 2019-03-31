
# coding: utf-8

# # Kernel methods for biological sequence classification
#
# MVA 2019 - Kernel methods for machine learning
#
# *Éloïse Berthier, Guillaume Dalle, Clément Mantoux*

import backend
from kernels import *

kernels1 = GaussianKernel(50).load("shapespectr4", indices=[0, 1, 2])
kernels2 = GaussianKernel(20).load("trans6", indices=[0, 1, 2])
kernels3 = GaussianKernel(50).load("tfidf6", indices=[0, 1, 2])

three_lambdas = [
    0.00025118864315095795,
    0.0001,
    0.0001
]
three_kernels = []

for dataset in [0, 1, 2]:
    kernels_to_combine = [
        kernels1[dataset],
        kernels2[dataset],
        kernels3[dataset]
    ]
    best_kernel = MultipleKernel(
        kernels_to_combine,
        grad_step=1, iterations=3, entropic=1
    )
    three_kernels.append(best_kernel)

backend.final_prediction(three_kernels, three_lambdas, simple_storage=True)
