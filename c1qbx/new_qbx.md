---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## TODO:

- implement single_layer, adjoint_double_layer, hypersingular
- try a fault with a singular tip!
- implement interior evaluation.

```{code-cell} ipython3
:tags: [remove-cell]

from config import setup, import_and_display_fnc

setup()
```

```{code-cell} ipython3
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    single_layer_matrix,
    double_layer_matrix,
    adjoint_double_layer_matrix,
    hypersingular_matrix,
    stage1_refine,
    qbx_panel_setup,
    build_stage2_panel_surf,
    apply_interp_mat,
    qbx_expand_matrix,
)
```

```{code-cell} ipython3
def upsample(src, kappa):
    stage2_panels = np.empty((src.n_panels, 3))
    stage2_panels[:, 0] = np.arange(src.n_panels)
    stage2_panels[:, 1] = -1
    stage2_panels[:, 2] = 1
    src_refined, interp_mat = build_stage2_panel_surf(
        src, stage2_panels, *gauss_rule(src.panel_order * kappa)
    )
    return src_refined, interp_mat


def double_layer_expand(exp_centers, src_pts, src_normals, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j
    return (nw * (r[:, None] ** m) / ((2 * np.pi) * (w - z0) ** (m + 1)))[:, None, :]


def double_layer_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return (z - z0) ** m / (r ** m)



def global_qbx_self(src, p, direction=1, kappa=3):
    obs_pts = src.pts

    L = np.repeat(src.panel_length, src.panel_order)
    exp_centers = src.pts + direction * src.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    src_high, interp_mat_high = upsample(src, kappa)

    exp_terms = []
    for i in range(p):
        K = double_layer_expand(
            exp_centers, src_high.pts, src_high.normals, exp_rs, i
        )
        I = K * (
            src_high.quad_wts[None, None, :] * src_high.jacobians[None, None, :]
        )
        exp_terms.append(I)

    eval_terms = []
    for i in range(p):
        eval_terms.append(double_layer_eval(obs_pts, exp_centers, exp_rs, i))

    kernel_ndim = exp_terms[0].shape[1]
    out = np.zeros((obs_pts.shape[0], kernel_ndim, src_high.n_pts), dtype=np.float64)
    for i in range(p):
        out += np.real(exp_terms[i][:, :, :] * eval_terms[i][:, None, None])

    return apply_interp_mat(out, interp_mat_high)
```

```{code-cell} ipython3
%load_ext cython
```

```{code-cell} ipython3
def build_interp_wts(x):
    dist = x[:, None] - x[None, :]
    np.fill_diagonal(dist, 1.0)
    weights = 1.0 / np.prod(dist, axis=1)
    return weights

def barycentric_eval(eval_pts, interp_pts, interp_wts, fnc_vals):
    dist = eval_pts[:, None] - interp_pts[None, :]
    kernel = interp_wts[None, :] / dist
    return (kernel.dot(fnc_vals)) / np.sum(kernel, axis=1)
```

```{code-cell} ipython3
%%cython --compile-args=-fopenmp --link-args=-fopenmp --verbose --cplus
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport pi, fabs
from libcpp cimport bool
from libcpp.pair cimport pair

cdef extern from "<complex.h>" namespace "std" nogil:
    double real(double complex z)
    
cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef pair[int, bool] single_obs(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, 
    double[:,::1] orig_src_pts, double[:,::1] orig_src_normals,
    double[::1] orig_src_jacobians, double[::1] orig_src_panel_widths,
    int orig_src_panel_order,
    double[:,::1] exp_centers, double[::1] exp_rs,
    double[:,::1] quad_pts, double[:, ::1] quad_wts, int[::1] quad_orders,
    double[:] interp_wts, 
    double tol, int max_kappa, int max_p,
    qbx_panels, int obs_pt_idx, int exp_idx
) nogil:

    cdef double complex z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    cdef double complex z0 = exp_centers[exp_idx,0] + (exp_centers[exp_idx, 1] * I)
    cdef double r = exp_rs[exp_idx]
    cdef double complex zz0_div_r = (z - z0) / r
    
    cdef long[:] panels    
    cdef int n_panels
    cdef int n_orig_srcs
    cdef double[:] qbx_terms
    cdef int[:] kappa
    
    with gil:
        if len(qbx_panels) == 0:
            return -1, False
        panels = np.unique(
            np.array(qbx_panels) // src_panel_order
        )
        n_panels = panels.shape[0]
        n_orig_srcs = n_panels * orig_src_panel_order
        n_max_srcs = n_orig_srcs * max_kappa
        qbx_terms = np.zeros(n_orig_srcs)
        kappa = np.ones(n_panels)
    
    cdef double am, am_sum
    cdef double complex w, nw, inv_wz0
    
    cdef int pt_start, pt_end, panel_idx, j, j_orig, orig_src_pt_idx, src_pt_idx, m
    cdef double src_x, src_y, src_nx, src_ny, src_J
    cdef double interp_denom, qx_kappa, qx_orig, wt, dist
    
    cdef double am_sum_prev = 0.0
    cdef int divergences = 0
    for m in range(max_p):
        am_sum = 0
        
        for panel_idx in range(n_panels):
            pt_start = panels[panel_idx] * src_panel_order
            pt_end = (panels[panel_idx] + 1) * src_panel_order
            k = kappa[panel_idx]
            for j in range(k * src_panel_order):
                # Interpolate if k > 1!
                if k==1:
                    src_x = orig_src_pts[src_pt_idx,0]
                    src_y = orig_src_pts[src_pt_idx,1]
                    src_nx = orig_src_normals[src_pt_idx,0]
                    src_ny = orig_src_normals[src_pt_idx,1]
                    src_J = orig_src_jacobians[src_pt_idx]
                else:
                    qx_kappa = quad_pts[k - 1, j]
                    src_x = 0
                    src_y = 0
                    src_nx = 0
                    src_ny = 0
                    src_J = 0
                    interp_denom = 0
                    for j_orig in range(pt_end - pt_start):
                        orig_src_pt_idx = j_orig + pt_start
                        qx_orig = quad_pts[0, j_orig]
                        wt = interp_wts[0, j_orig]
                        dist = qx_kappa - qx_orig
                        kernel = wt / dist
                        src_x += kernel * orig_src_pts[orig_src_pt_idx,0]
                        src_y += kernel * orig_src_pts[orig_src_pt_idx,1]
                        src_nx += kernel * orig_src_normals[orig_src_pt_idx,0]
                        src_ny += kernel * orig_src_normals[orig_src_pt_idx,1]
                        src_J += kernel * orig_src_jacobians[orig_src_pt_idx]
                        interp_denom += kernel
                    src_x /= interp_denom
                    src_y /= interp_denom
                    src_nx /= interp_denom
                    src_ny /= interp_denom
                    src_J *= orig_src_panel_widths[panel_idx] / interp_denom
                w = 
        # We use the sum of the last two terms to avoid issues with
        # common sequences where every terms alternate in magnitude
        # See Klinteberg and Tornberg 2018 at the end of page 5
        if (fabs(am_sum) + fabs(am_sum_prev)) < 2 * tol * mag_a0:
            divergences = 0
            break
        if (fabs(am_sum) > fabs(am_sum_prev)):
            divergences += 1
        else:
            divergences = 0
        am_sum_prev = am_sum
        eval_t *= zz0_div_r
    
    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for j in range(pt_end - pt_start):
            qbx_mat[obs_pt_idx, 0, pt_start + j] = qbx_terms[panel_idx * src_panel_order + j]
    
    return pair[int, bool](m, divergences > 1)
    
def local_qbx_self_integrals(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, src, 
    double[:,::1] exp_centers, double[::1] exp_rs, 
    double[:,::1] quad_pts, double[:, ::1] quad_wts, int[::1] quad_orders,
    double[:] interp_wts,
    double tol, int max_kappa, int max_p, 
    qbx_panels
):
        
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_quad_wts = src.quad_wts
    cdef int src_panel_order = src.panel_order
    
    cdef int i
    cdef np.ndarray p = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef np.ndarray kappa_too_small = np.empty(obs_pts.shape[0], dtype=np.bool_)
    
    cdef pair[int,bool] result
    for i in range(obs_pts.shape[0]):
        this_panel_set = qbx_panels[i]
        result = single_obs(
            qbx_mat, obs_pts, 
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, quad_pts, quad_wts, quad_orders, interp_wts,
            tol, max_kappa, max_p, this_panel_set, i, i
        )
        p[i], kappa_too_small[i] = result
    return p, kappa_too_small
```

```{code-cell} ipython3
import scipy.spatial
from common import stage2_refine


def local_qbx_self(
    src, d_cutoff, tol, kappa=5, direction=1, max_p=50, return_report=False
):
    direct_mat = double_layer_matrix(src, src.pts)

    L = np.repeat(src.panel_length, src.panel_order)
    exp_centers = src.pts + direction * src.normals * L[:, None] * 0.5
    exp_rs = L * 0.5
    exp_L = L

    refined_src, interp_mat = stage2_refine(src, exp_centers, kappa=kappa)

    src_high_tree = scipy.spatial.KDTree(refined_src.pts)
    qbx_src_pts_lists = src_high_tree.query_ball_point(exp_centers, d_cutoff * exp_L)
    n_src_pts_per_center = np.array([len(pt_list) for pt_list in qbx_src_pts_lists])
    #     qbx_src_pts = np.concatenate(qbx_src_pts_lists)
    #     pt_list_starts = np.zeros(n_src_pts_per_center.shape[0] + 1)
    #     pt_list_starts[1:] = np.cumsum(n_src_pts_per_center)
    #     local_terms = np.empty(qbx_src_pts.shape[0])

    quad_orders = np.arange(1, max_kappa+1, 1)
    quad_pts = np.empty((max_kappa, max_kappa * src.panel_order))
    quad_wts = np.empty((max_kappa, max_kappa * src.panel_order))
    for k in range(1, max_kappa+1):
        order = k * src.panel_order
        qx, qw = gauss_rule(order)
        quad_pts[k - 1,:qx.shape[0]] = qx
        quad_wts[k - 1,:qx.shape[0]] = qw
        quad_orders[k-1] = order
    interp_wts = build_interp_wts(gauss_rule(src.panel_order))
    
    # TODO: This could be replaced by a sparse local matrix.
    qbx_mat = np.zeros((src.pts.shape[0], 1, refined_src.n_pts))
    p, kappa = local_qbx_self_integrals(
        qbx_mat,
        src.pts,
        refined_src,
        exp_centers,
        exp_rs,
        quad_pts, quad_wts, quad_orders,
        tol,
        kappa,
        max_p,
        qbx_src_pts_lists,
    )
    final_local_mat = apply_interp_mat(qbx_mat, interp_mat)
    nonzero = np.abs(final_local_mat.ravel()) > 0
    direct_mat.ravel()[nonzero] = final_local_mat.ravel()[nonzero]

    if return_report:
        report = dict()
        report["stage2_src"] = refined_src
        report["exp_centers"] = exp_centers
        report["exp_rs"] = exp_rs
        report["n_qbx_panels"] = np.sum(n_src_pts_per_center) // src.panel_order
        report["qbx_src_pts_lists"] = qbx_src_pts_lists
        report["p"] = p
        report["kappa"] = kappa
        return direct_mat, report
    else:
        return direct_mat
```

```{code-cell} ipython3
from dataclasses import dataclass
from typing import Callable

@dataclass()
class Kernel:
    direct: Callable
    expand: Callable
    evaluate: Callable
    global_qbx_self: Callable
    local_qbx_self: Callable

double_layer = Kernel(
    direct=double_layer_matrix,
    expand=double_layer_expand,
    evaluate=double_layer_eval,
    global_qbx_self=global_qbx_self,
    local_qbx_self=local_qbx_self
)
```

```{code-cell} ipython3
max_curvature = 0.5
t = sp.var("t")
(circle,) = stage1_refine(
    [
        (t, sp.cos(sp.pi * t), sp.sin(sp.pi * t)),
    ],
    gauss_rule(12),
    max_curvature=max_curvature,
)
```

```{code-cell} ipython3
circle.n_pts, circle.n_panels
```

```{code-cell} ipython3
nq = 12
obs_pts = circle.pts
src = circle
kernel = double_layer_matrix
```

```{code-cell} ipython3
density = np.cos(circle.pts[:,0] - circle.pts[:,1])
baseline = global_qbx_self(circle, p=50, kappa=10, direction=1.0)
baseline_v = baseline.dot(density)
tols = 10.0 ** np.arange(0, -15, -1)
```

```{code-cell} ipython3
%%time
local_baseline, report = local_qbx_self(
    circle,
    d_cutoff=1.0,
    tol=1e-8,
    kappa=4,
    direction=1.0,
    return_report=True,
)
```

```{code-cell} ipython3
local_baseline_v = local_baseline.dot(density)
err = np.max(np.abs(baseline_v - local_baseline_v))
print(err)
plt.plot(baseline_v)
plt.plot(local_baseline_v)
plt.show()
```

## Kernel exploration prep stage

The prep parameters will be fixed as a function of `(max_curvature, tolerance, kernel, nq)`:

- $d_{\textrm{up}}$ is the distance at which an observation point will require an upsampled source quadrature

- $d_{\textrm{qbx}}$ is the distance at which QBX is necessary.

- $d_{\textrm{cutoff}}$: When we compute an integral using QBX, we will only use QBX for those source panels that are within $d_{\textrm{cutoff}}L$ from the expansion center where $L$ is the length of the source panel that spawned the expansion center.

- $\kappa$ is the upsampling ratio

```{code-cell} ipython3
# prep step 1: find d_cutoff and kappa
# The goal is to estimate the error due to the QBX local patch
# The local surface will have singularities at the tips where it is cut off
# These singularities will cause error in the QBX expansion. We want to make
# the local patch large enough that these singularities are irrelevant.
# To isolate the QBX patch cutoff error, we will use a very high upsampling.
# We'll also choose p to be the minimum allowed value since that will result in
# the largest cutoff error. Increasing p will reduce the cutoff error guaranteeing that
# we never need to worry about cutoff error.
```

```{code-cell} ipython3
d_tol = 5e-14
density = np.ones_like(src.pts[:, 0])  # np.cos(src.pts[:,0] * src.pts[:,1])
plt.figure(figsize=(9, 13))

params = []
d_cutoffs = np.linspace(0.8, 2.4, 9)

ps = np.arange(1, 55, 3)
for di, direction in enumerate([-1.0, 1.0]):
    baseline = global_qbx_self(src, p=15, kappa=10, direction=direction)
    baseline_v = baseline[:, 0, :].dot(density)

    # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
    d_cutoff = 100.0
    local_baseline = local_qbx_self(
        src, d_cutoff=100.0, tol=d_tol, max_p=10, kappa=10, direction=direction
    )
    local_baseline_v = local_baseline.dot(density)
    assert np.max(np.abs(baseline_v - local_baseline_v)) < 5e-14

    n_qbx_panels = []
    kappa_optimal = []
    p_for_full_accuracy = []
    plt.subplot(3, 2, 1 + di)
    for i_d, d_cutoff in enumerate(d_cutoffs):
        errs = []
        for i_p, p in enumerate(ps):
            # print(p, d_cutoff)
            kappa_temp = 8
            test, report = local_qbx_self(
                src,
                d_cutoff,
                tol=d_tol,
                max_p=p,
                direction=direction,
                kappa=kappa_temp,
                return_report=True,
            )
            testv = test[:, 0, :].dot(density)
            err = np.max(np.abs(baseline_v - testv))
            errs.append(err)
            if err < d_tol:
                for kappa_decrease in range(1, kappa_temp + 1):
                    kappa_test, kappa_report = local_qbx_self(
                        src,
                        d_cutoff,
                        tol=d_tol * 0.8, # Increase d_tol to have a safety margin.
                        max_p=p + 20,  # Increase p here to have a kappa safety margin
                        direction=direction,
                        kappa=kappa_decrease,
                        return_report=True,
                    )
                    kappa_testv = kappa_test[:, 0, :].dot(density)
                    kappa_err = np.max(np.abs(baseline_v - kappa_testv))
                    if kappa_err < d_tol:
                        kappa_optimal.append(kappa_decrease)
                        n_qbx_panels.append(kappa_report["n_qbx_panels"])
                        p_for_full_accuracy.append(p)
                        break
                if len(n_qbx_panels) <= i_d:
                    print(f"Failed to find parameters for {d_cutoff}")
                    kappa_optimal.append(1000)
                    n_qbx_panels.append(1e6)
                    p_for_full_accuracy.append(1e3)
                break
        print(d_cutoff, errs)
        plt.plot(ps[: i_p + 1], np.log10(errs), label=str(d_cutoff))

    params.append((direction, n_qbx_panels, kappa_optimal, p_for_full_accuracy))

    plt.legend()
    plt.title("interior" if direction > 0 else "exterior")
    plt.xlabel(r"$p_{\textrm{max}}$")
    if di == 0:
        plt.ylabel(r"$\log_{10}(\textrm{error})$")
    plt.yticks(-np.arange(0, 16, 3))
    plt.xticks(np.arange(0, 61, 10))
    plt.ylim([-15, 0])

    plt.subplot(3, 2, 3 + di)
    plt.plot(d_cutoffs, np.array(n_qbx_panels) / src.n_pts, "k-*")
    plt.xlabel(r"$d_{\textrm{cutoff}}$")
    plt.ylim([0, 20])
    if di == 0:
        plt.ylabel("QBX panels per point")

    plt.subplot(3, 2, 5 + di)
    plt.plot(d_cutoffs, np.array(kappa_optimal), "k-*")
    plt.xlabel(r"$d_{\textrm{cutoff}}$")
    plt.ylim([0, 6])
    if di == 0:
        plt.ylabel(r"$\kappa_{\textrm{optimal}}$")
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
total_cost = 0
for i in [0, 1]:
    direction, n_qbx_panels, kappa_optimal, p_for_full_accuracy = params[i]
    appx_cost = (
        np.array(p_for_full_accuracy) * np.array(n_qbx_panels) * np.array(kappa_optimal)
    )
    print(direction, appx_cost)
    total_cost += appx_cost
plt.plot(d_cutoffs, total_cost, "k-o")
plt.show()
```

```{code-cell} ipython3
best_idx = np.argmin(total_cost)
d_cutoff = d_cutoffs[best_idx]
kappa_qbx = kappa_optimal[best_idx]
```

```{code-cell} ipython3
from common import panelize_symbolic_surface

# prep step 2: find the minimum distance at which integrals are computed
# to the required tolerance for each kappa in [1, kappa_qbx]
def find_safe_direct_distance(nq, max_curvature, start_d, tol, kappa):
    t = sp.var("t")

    n_panels = 2
    while True:
        panel_edges = np.linspace(-1, 1, n_panels + 1)
        panel_bounds = np.stack((panel_edges[:-1], panel_edges[1:]), axis=1)
        circle = panelize_symbolic_surface(
            t, sp.cos(sp.pi * t), sp.sin(sp.pi * t), panel_bounds, *gauss_rule(nq)
        )
        n_panels_new = np.max(circle.panel_length / max_curvature * circle.panel_radius)
        if n_panels_new <= n_panels:
            break
        n_panels = np.ceil(n_panels_new).astype(int)
    #print(f"\nusing {n_panels} panels with max_curvature={max_curvature}")

    L = np.repeat(circle.panel_length, circle.panel_order)

    circle_high, interp_mat_high = upsample(circle, kappa)
    circle_higher, interp_mat_higher = upsample(circle, 8)
    # test_density = np.cos(circle.pts[:,0] * circle.pts[:,1])
    test_density = np.ones_like(circle.pts[:, 0])
    d = start_d
    for i in range(50):
        dist = L * d
        # In actuality, we only need to test interior points because the curvature
        # of the surface ensures that more source panels are near the observation
        # points and, as a result, the error will be higher for any given value of d.
        test_pts = np.concatenate(
            (
                circle.pts + circle.normals * dist[:, None],
                circle.pts - circle.normals * dist[:, None],
            )
        )

        # Check to make sure that the closest distance to a source point is truly `dist`.
        # This check might fail if the interior test_pts are crossing over into the other half of the circle.
        min_src_dist = np.min(
            np.linalg.norm((test_pts[:, None] - circle.pts[None, :]), axis=2), axis=1
        )
        if not np.allclose(min_src_dist, np.concatenate((dist, dist))):
            return False, d

        higher_mat = apply_interp_mat(
            kernel(circle_higher, test_pts), interp_mat_higher
        )
        high_mat = apply_interp_mat(kernel(circle_high, test_pts), interp_mat_high)

        # Use the absolute value of the matrix coefficients in order to compute an upper bound on the error
        err = np.max(np.abs(higher_mat - high_mat).dot(test_density))
        if err < tol:
            return True, d
        d *= 1.2


d_up = np.zeros(kappa_qbx)
for k in range(kappa_qbx, 0, -1):
    max_iter = 20
    d_up[k - 1] = d_up[k] if k < kappa_qbx else 0.05
    for i in range(max_iter):
        result = find_safe_direct_distance(
            nq, max_curvature * (0.8) ** i, d_up[k - 1], d_tol, k
        )
        d_up[k - 1] = result[1]
        if result[0]:
            print('done', k, d_up[k-1])
            break
```

```{code-cell} ipython3
print(f"using d_cutoff={d_cutoff}")
print(f"using kappa={kappa_qbx}")
print(f"using d_up = {d_up}")
```

```{code-cell} ipython3
import time
```

```{code-cell} ipython3
# %load_ext line_profiler
# %lprun -f local_qbx_self local_qbx_self(circle, d_cutoff=d_cutoff, tol=tol, kappa=kappa_qbx, direction=1.0)
```

```{code-cell} ipython3
%%time
density = np.ones_like(circle.pts[:, 0])  # np.cos(source.pts[:,0] * src.pts[:,1])
baseline = global_qbx_self(circle, p=50, kappa=10, direction=1.0)
baseline_v = baseline.dot(density)
tols = 10.0 ** np.arange(0, -15, -1)
errs = []
runtimes = []
for tol in tols:
    runs = []
    for i in range(10):
        start = time.time()
        local_baseline, report = local_qbx_self(
            circle,
            d_cutoff=d_cutoff,
            tol=tol,
            kappa=kappa_qbx,
            direction=1.0,
            return_report=True,
        )
        runs.append(time.time() - start)
    runtimes.append(np.min(runs))
    local_baseline_v = local_baseline.dot(density)
    errs.append(np.max(np.abs(baseline_v - local_baseline_v)))
    # print(tol, errs[-1], runtime)
    # assert(np.max(np.abs(baseline_v-local_baseline_v)) < 5e-14)
plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.plot(-np.log10(tols), np.log10(errs))
plt.subplot(1, 2, 2)
plt.plot(-np.log10(tols), runtimes)
plt.tight_layout()
plt.show()
```

## Interior Evaluation

```{code-cell} ipython3
from common import pts_grid
nobs = 50
zoomx = [0.75, 1.25]
zoomy = [0.15, 0.65]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
obs_pts.shape
```

```{code-cell} ipython3
# part 1: direct evaluation
# part 2: identify nearfield for each kappa
# part 3: identify qbx points
```

```{code-cell} ipython3
direct = kernel(source, obs_pts)
```
