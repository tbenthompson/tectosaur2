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
    exp_centers = source.pts - direction * source.normals * L[:, None] * 0.5
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
%%cython --verbose --cplus
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from libc.math cimport pi, fabs
from cython.parallel import prange

cdef extern from "<complex.h>" namespace "std" nogil:
    double real(double complex z)
    #double complex I
    
cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef void single_obs(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, 
    double[:,::1] src_pts, double[:,::1] src_normals,
    double[::1] src_jacobians, double[::1] src_quad_wts,
    int src_panel_order,
    double[:,::1] exp_centers, double[::1] exp_rs, 
    int min_p, int max_p, int max_kappa, double tol,
    qbx_panels, int obs_pt_idx
) nogil:
    
    cdef int pt_start, pt_end, panel_idx, src_pt_idx, m
    
    cdef double r, rm, qbx_term, JW, am
    cdef double complex z0, z, w, nw, exp_t, eval_t
    #cdef double complex zz0m[20]
    cdef double complex zz0_div_r, inv_wz0, r_inv_wz0
    cdef long[:] this_center_qbx_panels
    
    with gil:
        this_center_qbx_panels = np.unique(
            np.array(qbx_panels[obs_pt_idx]) // src_panel_order
        )

    z0 = exp_centers[obs_pt_idx,0] + (exp_centers[obs_pt_idx,1] * I)
    z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    r = exp_rs[obs_pt_idx]
    zz0_div_r = (z - z0) / r

    for panel_idx in range(this_center_qbx_panels.shape[0]):
        pt_start = this_center_qbx_panels[panel_idx] * src_panel_order
        pt_end = (this_center_qbx_panels[panel_idx] + 1) * src_panel_order
        for src_pt_idx in range(pt_start, pt_end):
            w = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
            nw = src_normals[src_pt_idx,0] + src_normals[src_pt_idx,1] * I
            JW = src_quad_wts[src_pt_idx] * src_jacobians[src_pt_idx]
            
            inv_wz0 = 1.0 / (w - z0)
            r_inv_wz0 = r * inv_wz0
            
            qbx_term = 0
            exp_t = nw * inv_wz0
            eval_t = C * JW
            am = real(exp_t * eval_t)
            mag_a0 = fabs(am) # TODO: should I use a relative tolerance?
            for m in range(max_p):
                if src_pt_idx == 15 and obs_pt_idx == 15:
                    with gil:
                        print(m, am)
                qbx_term += am
                if fabs(am) < tol * mag_a0:
                    break
                exp_t *= r_inv_wz0
                eval_t *= zz0_div_r
                am = real(exp_t * eval_t)
                
            qbx_mat[obs_pt_idx, 0, src_pt_idx] = qbx_term
    
def local_qbx_self_integrals(
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, src, 
    double[:,::1] exp_centers, double[::1] exp_rs, 
    int min_p, int max_p, int max_kappa, double tol,
    qbx_panels
):
        
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_quad_wts = src.quad_wts
    cdef int src_panel_order = src.panel_order
    
    cdef int i
    
    for i in range(obs_pts.shape[0]):#, nogil=True):
        single_obs(
            qbx_mat, obs_pts, 
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, min_p, max_p, max_kappa, tol, qbx_panels, i
        )
```

```{code-cell} ipython3
from common import stage2_refine


def local_qbx_self(source, d_cutoff, tol, direction=1, min_p=2, max_p=30, max_kappa=5):
    obs_pts = source.pts

    L = np.repeat(source.panel_length, source.panel_order)
    exp_centers = source.pts - direction * source.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    refined_source, interp_mat = stage2_refine(source, exp_centers, kappa=max_kappa)
    print(f"initial source n_pts={source.n_pts}")
    print(f"stage2 refinement n_pts={refined_source.n_pts}")

    import scipy.spatial

    qbx_mat = double_layer_matrix(refined_source, obs_pts)

    src_high_tree = scipy.spatial.KDTree(refined_source.pts)
    qbx_src_pts_lists = src_high_tree.query_ball_point(exp_centers, d_cutoff * L)

    local_qbx_self_integrals(
        qbx_mat,
        obs_pts,
        refined_source,
        exp_centers,
        exp_rs,
        min_p,
        max_p,
        max_kappa,
        tol,
        qbx_src_pts_lists,
    )
    return apply_interp_mat(qbx_mat, interp_mat)
```

```{code-cell} ipython3
density = np.ones_like(source.pts[:,0])#np.cos(source.pts[:,0] * source.pts[:,1])
baseline = global_qbx_self(source, p=50, kappa=10, direction=-1.0)
baseline_v = baseline.dot(density)

# Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
local_baseline = local_qbx_self(
    source, d_cutoff=100.0, tol=1e-10, max_p=50, max_kappa=10, direction=-1.0
)
local_baseline_v = local_baseline.dot(density)
print(np.max(np.abs(baseline_v - local_baseline_v)))
#assert(np.max(np.abs(baseline_v-local_baseline_v)) < 5e-14)
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
local_qbx_self(circle, 1.5, direction=1.0)
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
tol = 1e-5
obs_pts = circle.pts
source = circle
kernel = double_layer_matrix
```

$d_{\textrm{up}}$ is the distance at which an observation point will require an upsampled source quadrature

$d_{\textrm{qbx}}$ is the distance at which QBX is necessary.

```{code-cell} ipython3
# prep step 1: find d_up
# Here, we find the distance at which the error in comparison to a
# panel at double the distance is less than some tiny tolerance
def upsample(source, kappa):
    stage2_panels = np.empty((source.n_panels, 3))
    stage2_panels[:, 0] = np.arange(source.n_panels)
    stage2_panels[:, 1] = -1
    stage2_panels[:, 2] = 1
    source_refined, interp_mat = build_stage2_panel_surf(
        source, stage2_panels, *gauss_rule(source.panel_order * kappa)
    )
    return source_refined, interp_mat


def find_safe_dist(source, start_d, tol, kappa_base):
    L = np.repeat(source.panel_length, source.panel_order)

    source_high, interp_mat_high = upsample(source, kappa_base)
    source_higher, interp_mat_higher = upsample(source, kappa_base + 1)
    test_density = np.cos(1 * np.pi * source.pts[:, 0])
    d = start_d
    for i in range(20):
        test_pts = source.pts - source.normals * L[:, None] * d
        higher_mat = apply_interp_mat(
            kernel(source_higher, test_pts), interp_mat_higher
        )
        high_mat = apply_interp_mat(kernel(source_high, test_pts), interp_mat_high)

        higher_vals = higher_mat.dot(test_density)
        high_vals = high_mat.dot(test_density)

        err = np.max(np.abs(higher_vals - high_vals))
        maxv = np.max(np.abs(higher_vals))

        rel_err = err / maxv

        print(d, rel_err)
        if rel_err < tol:
            return d
        d *= 1.2


d_tol = 1e-13
d_qbx = find_safe_dist(source, 0.05, d_tol, 2)
d_up = find_safe_dist(source, d_qbx * 2, d_tol, 1)
print(f"using d_up = {d_up}")
print(f"using d_qbx = {d_qbx}")
```

When we compute an integral using QBX, we will only use QBX for those source panels that are within $d_{\textrm{cutoff}}L$ from the expansion center where $L$ is the length of the source panel that spawned the expansion center.

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
for direction in [-1.0, 1.0]:
    baseline = global_qbx_self(source, p=50, kappa=10, direction=direction)
    baseline_v = baseline[:, 0, :].dot(density)

    # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
    d_cutoff = 100.0
    local_baseline = local_qbx_self(
        source, d_cutoff=100.0, max_p=50, max_kappa=10, direction=direction
    )
    local_baseline_v = local_baseline.dot(density)
    print(np.max(np.abs(baseline_v - local_baseline_v)))
    assert(np.max(np.abs(baseline_v-local_baseline_v)) < 5e-14)

    d_cutoffs = [0.9, 1.2, 1.5, 2.0]
    ps = np.arange(3, 55, 5)
    for d_cutoff in d_cutoffs:
        errs = []
        for p in ps:
            # print(p, d_cutoff)
            kappa_temp = kappa + p // 10
            test = local_qbx_self(
                source, d_cutoff, max_p=p, direction=direction, max_kappa=kappa_temp
            )
            testv = test[:, 0, :].dot(density)
            errs.append(np.max(np.abs(baseline_v - testv)))
        print(errs)
        plt.plot(ps, np.log10(errs), label=str(d_cutoff))
    plt.legend()
    plt.ylim([-15, 0])
    plt.show()
```

```{code-cell} ipython3
errs
```

```{code-cell} ipython3
# prep step 2: find upsampling kappa for use when d_up < d < d_qbx
upsampling_tol = 1e-10
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
