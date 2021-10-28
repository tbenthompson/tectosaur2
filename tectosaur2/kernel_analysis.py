# flake8: noqa
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from tectosaur2.nb_config import setup

setup()

import matplotlib.pyplot as plt
import numpy as np

# %%
import sympy as sp

from tectosaur2.analyze import final_check, find_d_up, find_dcutoff_refine
from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)
from tectosaur2.mesh import gauss_rule, stage1_refine

# %%
kernel = double_layer
tol = 1e-13
nq = 12
max_curvature = 0.5

t = sp.var("t")
(circle,) = stage1_refine(
    [
        (t, sp.cos(sp.pi * t), sp.sin(sp.pi * t)),
    ],
    gauss_rule(nq),
    max_curvature=max_curvature,
    control_points=np.array([[1, 0, 0, 0.1]]),
)

# %%
circle.n_pts

# %%
# final_check(kernel, circle)
for K in [
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
]:
    d_up = find_d_up(K, nq, max_curvature, 0.05, tol, 1)
    d_qbx = find_d_up(K, nq, max_curvature, 0.05, tol, 3)
    d_cutoff, d_refine = find_dcutoff_refine(K, circle, tol)
    print("\n", K)
    print("d_cutoff =", d_cutoff)
    print("d_refine =", d_refine)
    print("d_up =", d_up)
    print("d_qbx =", d_qbx)

# %%
import sys

sys.exit()

# %%
# d_up = find_d_up(double_layer_matrix, tol, nq, max_curvature, d_refine)

# %%
# print(f"using d_cutoff={d_cutoff}")
# print(f"using kappa={kappa_qbx}")
# print(f"using d_up = {d_up}")

# %%
# d_up = find_d_up(K, tol, nq, max_curvature, kappa_qbx)

# %%
