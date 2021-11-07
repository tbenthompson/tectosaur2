# ---
# flake8: noqa
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''tectosaur2'': conda)'
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
from tectosaur2.mesh import gauss_rule, refine_surfaces

# %%
kernel = double_layer
tol = 1e-13
nq = 12
max_curvature = 0.5

t = sp.var("t")
(circle,) = refine_surfaces(
    [
        (t, sp.cos(sp.pi * t), sp.sin(sp.pi * t)),
    ],
    gauss_rule(nq),
    max_curvature=max_curvature,
    control_points=np.array([[1, 0, 0, 0.1]]),
)

# %%
Ms = []
for p in range(10, 30):
    Ms.append(
        hypersingular.integrate(
            circle.pts,
            circle,
            d_cutoff=2.0,
            tol=1e-25,
            max_p=p,
            d_refine=4.5,
            on_src_direction=1.0,
        )
    )

# %%
density = np.cos(circle.pts[:, 0])
vs = []
for i in range(len(Ms)):
    vs.append(Ms[i].dot(density))

# %%
for i in range(len(Ms) - 1):
    diff = vs[i + 1] - vs[i]
    print(np.max(np.abs(diff)))
    plt.plot(diff[:, 0], label=str(10 + i))
plt.legend()
plt.show()

# %%
circle.n_pts

# %%
# final_check(kernel, circle)
for K in [
    # single_layer,
    # double_layer,
    adjoint_double_layer,
    # hypersingular,
]:
    d_up = find_d_up(K, nq, max_curvature, 0.05, tol, 1)
    d_qbx = find_d_up(K, nq, max_curvature, 0.05, tol, 3)
    d_cutoff, d_refine = find_dcutoff_refine(K, circle, tol, plot=True)
    print("\n", type(K).__name__ + "(")
    print(f"    d_cutoff = {d_cutoff},")
    print(f"    d_refine = {d_refine},")
    print(f"    d_up = {d_up}")
    print(f"    d_qbx = {d_qbx}")
    print(")")

# %%

# final_check(kernel, circle)
for k_name, tol in [
    # ("single_layer", 1e-13),
    # ("double_layer", 1e-13),
    # ("adjoint_double_layer", 1e-13),
    ("hypersingular", 1e-13)
]:
    K = locals()[k_name]
    d_up = find_d_up(K, nq, max_curvature, 0.05, 1e-13, 1)
    d_qbx = find_d_up(K, nq, max_curvature, 0.05, 1e-13, 3)
    d_cutoff, d_refine = find_dcutoff_refine(K, circle, tol)
    print(f"\n{k_name} = {type(K).__name__}(")
    print(f"    d_cutoff = {d_cutoff},")
    print(f"    d_refine = {d_refine},")
    print(f"    d_up = {d_up},")
    print(f"    d_qbx = {d_qbx}")
    print(")")

# %%

# %%

# %%
