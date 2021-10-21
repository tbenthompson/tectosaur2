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
def upsample(source, kappa):
    stage2_panels = np.empty((source.n_panels, 3))
    stage2_panels[:, 0] = np.arange(source.n_panels)
    stage2_panels[:, 1] = -1
    stage2_panels[:, 2] = 1
    source_refined, interp_mat = build_stage2_panel_surf(
        source, stage2_panels, *gauss_rule(source.panel_order * kappa)
    )
    return source_refined, interp_mat

def double_layer_expand(exp_centers, src_pts, src_normals, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j
    return (nw * (r[:, None] ** m) / ((2 * np.pi) * (w - z0) ** (m + 1)))[:, None, :]


def double_layer_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return (z - z0) ** m / (r ** m)


def global_qbx_self(source, p, direction=1, kappa=3):
    obs_pts = source.pts

    L = np.repeat(source.panel_length, source.panel_order)
    exp_centers = source.pts + direction * source.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    source_high, interp_mat_high = upsample(source, kappa)

    exp_terms = []
    for i in range(p):
        K = double_layer_expand(
            exp_centers, source_high.pts, source_high.normals, exp_rs, i
        )
        I = K * (
            source_high.quad_wts[None, None, :] * source_high.jacobians[None, None, :]
        )
        exp_terms.append(I)

    eval_terms = []
    for i in range(p):
        eval_terms.append(double_layer_eval(obs_pts, exp_centers, exp_rs, i))

    kernel_ndim = exp_terms[0].shape[1]
    out = np.zeros((obs_pts.shape[0], kernel_ndim, source_high.n_pts), dtype=np.float64)
    for i in range(p):
        out += np.real(exp_terms[i][:, :, :] * eval_terms[i][:, None, None])

    return apply_interp_mat(out, interp_mat_high)
```

```{code-cell} ipython3
%load_ext cython
```

```{code-cell} ipython3
%%cython --compile-args=-fopenmp --link-args=-fopenmp --verbose --cplus
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport pi, fabs
from cython.parallel import prange

cdef extern from "<complex.h>" namespace "std" nogil:
    double real(double complex z)
    #double complex I
    
cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef int single_obs(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, 
    double[:,::1] src_pts, double[:,::1] src_normals,
    double[::1] src_jacobians, double[::1] src_quad_wts,
    int src_panel_order,
    double[:,::1] exp_centers, double[::1] exp_rs, 
    int max_p, double tol,
    qbx_panels, int obs_pt_idx
) nogil:

    cdef double complex z0 = exp_centers[obs_pt_idx,0] + (exp_centers[obs_pt_idx,1] * I)
    cdef double complex z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    cdef double r = exp_rs[obs_pt_idx]
    cdef double complex zz0_div_r = (z - z0) / r
    
    cdef long[:] panels
    with gil:
        if len(qbx_panels[obs_pt_idx]) == 0:
            return -1
        panels = np.unique(
            np.array(qbx_panels[obs_pt_idx]) // src_panel_order
        )
    
    cdef int n_panels = panels.shape[0]
    cdef int n_srcs = n_panels * src_panel_order
    cdef double complex[:] r_inv_wz0
    cdef double complex[:] exp_t
    cdef double[:] qbx_terms
    with gil:
        r_inv_wz0 = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        exp_t = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        qbx_terms = np.zeros(n_panels * src_panel_order)
    
    
    cdef double am, am_sum
    cdef double complex w, nw, inv_wz0

    cdef double mag_a0 = 0
    cdef double complex eval_t = 1.0
    
    cdef int pt_start, pt_end, pt_idx, panel_idx, src_pt_idx, j, m
    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = panel_idx * src_panel_order + pt_idx
            w = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
            nw = src_normals[src_pt_idx,0] + src_normals[src_pt_idx,1] * I
            inv_wz0 = 1.0 / (w - z0)
            
            r_inv_wz0[j] = r * inv_wz0
            exp_t[j] = nw * inv_wz0 * C * src_quad_wts[src_pt_idx] * src_jacobians[src_pt_idx]
            qbx_terms[j] = 0.0
            mag_a0 += fabs(real(exp_t[j] * eval_t))
    
    cdef double am_sum_last = 0.0
    for m in range(max_p):
        am_sum = 0
        for j in range(n_srcs):
            am = real(exp_t[j] * eval_t)
            am_sum += am
            qbx_terms[j] += am
            exp_t[j] *= r_inv_wz0[j]
        # We use the sum of the last two terms to avoid issues with
        # common sequences where every terms alternate in magnitude
        # See Klinteberg and Tornberg 2018 at the end of page 5
        if (fabs(am_sum) + fabs(am_sum_last)) < 2 * tol * mag_a0:
            break
        am_sum_last = am_sum
        eval_t *= zz0_div_r
    
    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = panel_idx * src_panel_order + pt_idx
            qbx_mat[obs_pt_idx, 0, src_pt_idx] = qbx_terms[j]
    return m
    
def local_qbx_self_integrals(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, src, 
    double[:,::1] exp_centers, double[::1] exp_rs, 
    int max_p, double tol,
    qbx_panels
):
        
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_quad_wts = src.quad_wts
    cdef int src_panel_order = src.panel_order
    
    cdef int i
    cdef np.ndarray p = np.empty(obs_pts.shape[0], dtype=np.int32)
    for i in range(obs_pts.shape[0]):#TODO: PRANGE, nogil=True):
        p[i] = single_obs(
            qbx_mat, obs_pts, 
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, max_p, tol, qbx_panels, i
        )
    return p
```

```{code-cell} ipython3
import scipy.spatial
from common import stage2_refine

def local_qbx_self(source, d_cutoff, tol, direction=1, max_p=50, kappa=5, return_report=False):
    direct_mat = double_layer_matrix(source, source.pts)

    L = np.repeat(source.panel_length, source.panel_order)
    exp_centers = source.pts + direction * source.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    refined_source, interp_mat = stage2_refine(source, exp_centers, kappa=kappa)

    src_high_tree = scipy.spatial.KDTree(refined_source.pts)
    qbx_src_pts_lists = src_high_tree.query_ball_point(exp_centers, d_cutoff * L)
    n_src_pts_per_center = np.array([len(pt_list) for pt_list in qbx_src_pts_lists])
#     qbx_src_pts = np.concatenate(qbx_src_pts_lists)
#     pt_list_starts = np.zeros(n_src_pts_per_center.shape[0] + 1)
#     pt_list_starts[1:] = np.cumsum(n_src_pts_per_center)
#     local_terms = np.empty(qbx_src_pts.shape[0])
    
    # TODO: This could be replaced by a sparse local matrix.
    qbx_mat = np.zeros((source.pts.shape[0], 1, refined_source.n_pts))
    p = local_qbx_self_integrals(
        qbx_mat,
        source.pts,
        refined_source,
        exp_centers,
        exp_rs,
        max_p,
        tol,
        qbx_src_pts_lists
    )
    final_local_mat = apply_interp_mat(qbx_mat, interp_mat)
    nonzero = np.abs(final_local_mat.ravel()) > 0
    direct_mat.ravel()[nonzero] = final_local_mat.ravel()[nonzero]
    
    if return_report:
        report = dict()
        report['stage2_src'] = refined_source
        report['exp_centers'] = exp_centers
        report['exp_rs'] = exp_rs
        report['n_qbx_panels'] = np.sum(n_src_pts_per_center) // source.panel_order
        report['qbx_src_pts_lists'] = qbx_src_pts_lists
        report['p'] = p
        return direct_mat, report
    else:
        return direct_mat
```

```{code-cell} ipython3
t = sp.var("t")
(circle,) = stage1_refine(
    [
        (t, sp.cos(sp.pi * t), sp.sin(sp.pi * t)),
    ],
    gauss_rule(12),
    max_radius_ratio=1.0,
)
```

```{code-cell} ipython3
circle.n_pts, circle.n_panels
```

```{code-cell} ipython3
import numpy as np
density = np.ones_like(circle.pts[:,0])#np.cos(source.pts[:,0] * source.pts[:,1])
baseline = global_qbx_self(circle, p=50, kappa=10, direction=1.0)
baseline_v = baseline.dot(density)
```

```{code-cell} ipython3
import time
```

```{code-cell} ipython3
%%time
tols = 10.0 ** np.arange(0, -15, -1)
errs = []
for tol in tols:
    start = time.time()
    # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
    local_baseline, report = local_qbx_self(
        circle, d_cutoff=1.5, tol=tol, kappa=4, direction=1.0, return_report=True
    )
    local_baseline_v = local_baseline.dot(density)
    errs.append(np.max(np.abs(baseline_v - local_baseline_v)))
    runtime = time.time() - start
    print(tol, errs[-1], runtime)
    #assert(np.max(np.abs(baseline_v-local_baseline_v)) < 5e-14)
plt.plot(-np.log10(tols), np.log10(errs))
plt.show()
```

```{code-cell} ipython3
tol = 1e-7
nq = 12
obs_pts = circle.pts
source = circle
kernel = double_layer_matrix
```

## Prep stage

The prep parameters will be fixed as a function of `(max_curvature, tolerance, kernel, nq)`:

- $d_{\textrm{up}}$ is the distance at which an observation point will require an upsampled source quadrature

- $d_{\textrm{qbx}}$ is the distance at which QBX is necessary.

- $d_{\textrm{cutoff}}$: When we compute an integral using QBX, we will only use QBX for those source panels that are within $d_{\textrm{cutoff}}L$ from the expansion center where $L$ is the length of the source panel that spawned the expansion center.

```{code-cell} ipython3
# prep step 1: find d_up
# Here, we find the distance at which the error in comparison to a
# panel at double the distance is less than some tiny tolerance
def find_safe_direct_distance(nq, max_curvature, start_d, tol, kappa):
    t = sp.var("t")
    
    n_panels = 2
    while True:
        circle = panelize_symbolic_surface(
            t, 
            sp.cos(sp.pi * t), 
            sp.sin(sp.pi * t),
            np.linspace(-1, 1, n_panels + 1),
            *gauss_rule(nq)
        )
        n_panels_new = 2 * max_curvature * circle.panel_radius / circle.panel_length
        print(n_panels_new)
        if n_panels_new <= n_panels:
            break
    
    L = np.repeat(source.panel_length, source.panel_order)

    source_high, interp_mat_high = upsample(source, kappa)
    source_higher, interp_mat_higher = upsample(source, 8)
    #test_density = np.cos(source.pts[:,0] * source.pts[:,1])
    test_density = np.ones_like(source.pts[:,0])
    d = start_d
    for i in range(50):
        dist = L * d
        # In actuality, we only need to test interior points because the curvature 
        # of the surface ensures that more source panels are near the observation
        # points and, as a result, the error will be higher for any given value of d.
        test_pts = np.concatenate((
            source.pts + source.normals * dist[:, None],
            source.pts - source.normals * dist[:, None]
        ))
        
        # Check to make sure that the closest distance to a source point is truly `dist`. 
        # This check might fail if the interior test_pts are crossing over into the other half of the circle.
        min_src_dist = np.min(np.linalg.norm((test_pts[:, None] - source.pts[None,:]), axis=2), axis=1)
        np.testing.assert_allclose(min_src_dist, np.concatenate((dist, dist)))
        
        higher_mat = apply_interp_mat(
            kernel(source_higher, test_pts), interp_mat_higher
        )
        high_mat = apply_interp_mat(kernel(source_high, test_pts), interp_mat_high)

        higher_vals = np.abs(higher_mat).dot(test_density)
        high_vals = np.abs(high_mat).dot(test_density)

        err = np.max(np.abs(higher_vals - high_vals))

        print(d, err)
        if err < tol:
            return d
        d *= 1.2

d_tol = 5e-14
d_qbx = find_safe_direct_distance(nq, 0.2, d_tol, 2)
d_up = find_safe_direct_distance(nq, d_qbx * 1.5, d_tol, 1)
print(f"using d_up = {d_up}")
print(f"using d_qbx = {d_qbx}")
```

```{code-cell} ipython3
# prep step 2: find d_cutoff
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
density = np.ones_like(source.pts[:, 0])  # np.cos(source.pts[:,0] * source.pts[:,1])
kappa = 2
plt.figure(figsize=(9,9))
for di, direction in enumerate([-1.0, 1.0]):
    baseline = global_qbx_self(source, p=10, kappa=10, direction=direction)
    baseline_v = baseline[:, 0, :].dot(density)

    # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
    d_cutoff = 100.0
    local_baseline = local_qbx_self(
        source, d_cutoff=100.0, tol=d_tol, max_p=10, kappa=10, direction=direction
    )
    local_baseline_v = local_baseline.dot(density)
    assert(np.max(np.abs(baseline_v - local_baseline_v)) < 5e-14)

    d_cutoffs = [0.9, 1.2, 1.5, 1.8, 2.0]
    ps = np.arange(3, 55, 5)
    
    n_qbx_panels = []
    p_for_full_accuracy = []
    plt.subplot(2,2,1+di)
    for d_cutoff in d_cutoffs:
        errs = []
        for i_p, p in enumerate(ps):
            # print(p, d_cutoff)
            kappa_temp = kappa + p // 10
            test, report = local_qbx_self(
                source, d_cutoff, tol=d_tol, max_p=p, direction=direction, kappa=kappa_temp, return_report=True
            )
            if p == ps[0]:
                n_qbx_panels.append(report['n_qbx_panels'])
            testv = test[:, 0, :].dot(density)
            err = np.max(np.abs(baseline_v - testv))
            errs.append(err)
            if err < d_tol:
                p_for_full_accuracy.append(p)
                break
        #print(errs)
        plt.plot(ps[:i_p+1], np.log10(errs), label=str(d_cutoff))
    plt.legend()
    plt.title('interior' if direction > 0 else 'exterior')
    plt.xlabel(r'$p_{\textrm{max}}$')
    if di == 0:
        plt.ylabel(r'$\log_{10}(\textrm{error})$')
    plt.yticks(-np.arange(0, 16, 3))
    plt.xticks(np.arange(0,61,10))
    plt.ylim([-15, 0])
    
    plt.subplot(2,2,3+di)
    plt.plot(d_cutoffs, np.array(n_qbx_panels) / source.n_pts, 'k-*')
    plt.xlabel(r'$d_{\textrm{cutoff}}$')
    if di == 0:
        plt.ylabel('QBX panels per point')
    scaled_expense = np.array(n_qbx_panels) / source.n_pts
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
np.array(p_for_full_accuracy) * np.array(n_qbx_panels)
```

```{code-cell} ipython3
# prep step 2: find upsampling kappa for use when d_up < d < d_qbx
upsampling_tol = d_tol
test_pts = source.pts + source.normals * L_per_source_pt[:, None] * d_qbx
source_high, interp_mat_high = upsample(10)
high_mat = apply_interp_mat(kernel(source_high, test_pts), interp_mat_high)
for i in range(2, 10):
    source_low, interp_mat_low = upsample(i)
    low_mat = apply_interp_mat(kernel(source_low, test_pts), interp_mat_low)

    high_vals = high_mat.dot(test_density)
    low_vals = low_mat.dot(test_density)

    if (
        np.max(np.abs(high_vals - low_vals)) / np.max(np.abs(high_vals))
        < upsampling_tol
    ):
        break
upsample_kappa = i
print(f"using upsample_kappa = {upsample_kappa}")
```

```{code-cell} ipython3
d_up
```

```{code-cell} ipython3
# step 0: build the farfield!

farfield = kernel(source, obs_pts)

# step 1: find the observation points and source points that need to use QBX
import scipy.spatial

src_tree = scipy.spatial.KDTree(source.pts)

source_dist, closest_source_pt = src_tree.query(obs_pts)
L = source.panel_length[closest_source_pt // source.panel_order]
obs_pts_using_qbx = source_dist < d_up * L

source_pts_using_qbx = closest_source_pt[obs_pts_using_qbx]
```

```{code-cell} ipython3
# step 2: find the expansion distance
(expansions,) = qbx_panel_setup([source], directions=[1], singularities=singularities)
```

```{code-cell} ipython3
# step 3: find the source panels that need to be upsampled
upsampled_panels = src_tree.query_ball_point(expansions.pts, d_qbx)
qbx_panels = src_tree.query_ball_point(expansions.pts, d_up * L_per_source_pt)
this_center_qbx_panels = np.unique(np.array(qbx_panels[0]) // source.panel_order)
this_center_upsampled_panels = np.unique(np.array(upsampled[0]) // source.panel_order)
```

```{code-cell} ipython3
this_center_qbx_panels
```

```{code-cell} ipython3
this_center_upsampled_panels
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# step 4: Compute upsampled quadrature for panels that need it!
```
