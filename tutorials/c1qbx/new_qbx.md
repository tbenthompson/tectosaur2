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

- local qbx using single_layer, adjoint_double_layer, hypersingular
- make sure this works for fault-surface intersection and the like where historically i've needed to make sure the expansion centers were exactly the same.
- try a fault with a singular tip!
- optimize! later!

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
    build_stage2_panel_surf,
    apply_interp_mat,
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
    control_points=np.array([[1,0,0,0.1]])
)
```

```{code-cell} ipython3
density=np.cos(circle.pts[:,0])
obs_pts=np.array([[0.1,0.1]])
exp_c = np.array([[0.1,0.0]])
exp_r = np.array([1.0])
p = 5
```

```{code-cell} ipython3
def base_expand(exp_centers, src_pts, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    if m == 0:
        return np.log(w - z0) / (2 * np.pi)
    else:
        return -(r ** m) / (m * (2 * np.pi) * (w - z0) ** m)

def deriv_expand(exp_centers, src_pts, src_normals, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j
    return (nw * (r[:, None] ** m) / ((2 * np.pi) * (w - z0) ** (m + 1)))

def base_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return (z - z0) ** m / (r ** m)

def deriv_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return -m * (z - z0) ** (m - 1) / (r ** m)
```

```{code-cell} ipython3
hypersingular_matrix(circle, obs_pts).dot(density)
```

```{code-cell} ipython3
out = np.zeros((1,2,circle.n_pts))
for m in range(p):
    exp_t = deriv_expand(exp_c, circle.pts, circle.normals, exp_r, m)
    eval_t = deriv_eval(obs_pts, exp_c, exp_r, m)
    out[:,0,:] += np.real(exp_t * eval_t * circle.jacobians * circle.quad_wts)
    out[:,1,:] -= np.imag(exp_t * eval_t * circle.jacobians * circle.quad_wts)
out.dot(density)
```

```{code-cell} ipython3
single_layer_matrix(circle, obs_pts).dot(density)
```

```{code-cell} ipython3
out = 0
for m in range(p):
    exp_t = base_expand(exp_c, circle.pts, exp_r, m)
    eval_t = base_eval(obs_pts, exp_c, exp_r, m)
    out += np.real(exp_t * eval_t * circle.jacobians * circle.quad_wts)
out.dot(density)
```

```{code-cell} ipython3
adjoint_double_layer_matrix(circle, obs_pts).dot(density)
```

```{code-cell} ipython3
out = np.zeros((1,2,circle.n_pts))
for m in range(p):
    exp_t = base_expand(exp_c, circle.pts, exp_r, m)
    eval_t = deriv_eval(obs_pts, exp_c, exp_r, m)
    out[:,0,:] += np.real(exp_t * eval_t * circle.jacobians * circle.quad_wts)
    out[:,1,:] -= np.imag(exp_t * eval_t * circle.jacobians * circle.quad_wts)
out.dot(density)
```

```{code-cell} ipython3
d_cutoff=1.6
kappa=4
d_up = [3.31236863, 0.53496603, 0.30958682, 0.21499085]
```

```{code-cell} ipython3
circle.panel_length
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

## Quick test

```{code-cell} ipython3
density = np.ones_like(circle.pts[:,0])#np.cos(circle.pts[:,0] - circle.pts[:,1])
baseline = global_qbx_self(circle, p=50, kappa=10, direction=1.0)
baseline_v = baseline.dot(density)
tols = 10.0 ** np.arange(0, -15, -1)
```

```{code-cell} ipython3
%%time
local_baseline, report = local_qbx(
    circle.pts,
    circle,
    tol=1e-8,
    d_cutoff=1.6,
    kappa=4,
    d_up = [3.31236863, 0.53496603, 0.30958682, 0.21499085],
    on_src_direction=1.0,
    return_report=True,
)
```

```{code-cell} ipython3
local_baseline_v = local_baseline.dot(density)
err = np.max(np.abs(baseline_v - local_baseline_v))
print(err)
#plt.plot(baseline_v, 'k-')
#plt.plot(local_baseline_v, 'r-')
plt.plot(np.repeat(circle.panel_length, circle.panel_order),'k-')
plt.show()
plt.plot(local_baseline_v, 'r-')
plt.show()
```

## Kernel exploration prep stage

The prep parameters will be fixed as a function of `(max_curvature, tolerance, kernel, nq)`:

- $d_{\textrm{up}}$ is the distance at which an observation point will require an upsampled source quadrature

- $d_{\textrm{qbx}}$ is the distance at which QBX is necessary.

- $d_{\textrm{cutoff}}$: When we compute an integral using QBX, we will only use QBX for those source panels that are within $d_{\textrm{cutoff}}L$ from the expansion center where $L$ is the length of the source panel that spawned the expansion center.

- $\kappa$ is the upsampling ratio

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

```

## Interior Evaluation

```{code-cell} ipython3
from common import pts_grid
nobs = 100
zoomx = [0.75, 1.25]
zoomy = [0.0, 0.5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
obs_pts.shape
```

```{code-cell} ipython3
%%time
upsampled_src, interp = upsample(src, 10)
high = double_layer_matrix(upsampled_src, obs_pts)
mat_upsampled = apply_interp_mat(high, interp)
```

```{code-cell} ipython3
mat_upsampled.shape
```

```{code-cell} ipython3
%load_ext line_profiler
```

```{code-cell} ipython3
%lprun -f local_qbx local_qbx(obs_pts,circle,tol = 1e-13,d_cutoff = 1.6,kappa = 4,d_up = [3.31236863, 0.53496603, 0.30958682, 0.21499085],on_src_direction=1.0,return_report=True)
```

```{code-cell} ipython3
%%time
mat, report = local_qbx(
    obs_pts,
    circle,
    tol = 1e-8,
    d_cutoff = 1.6,
    kappa = 4,
    d_up = [3.31236863, 0.53496603, 0.30958682, 0.21499085],
    on_src_direction=1.0,
    return_report=True,
)
```

```{code-cell} ipython3
plt.plot(report['exp_centers'][:,0], report['exp_centers'][:,1], 'k.')
plt.plot(src.pts[:,0], src.pts[:,1],'k-')
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.show()

density = np.cos(src.pts[:,0]**2)
check = mat_upsampled.dot(density)

est = mat.dot(density)

logerror = np.log10(np.abs(est - check)).reshape((nobs, nobs))
logerror[np.isinf(logerror)] = -17.0
error_levels = np.linspace(-15, -3, 5)
cntf = plt.contourf(xs, ys, logerror, levels=error_levels, extend="both")
plt.contour(
    xs,
    ys,
    logerror,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=error_levels,
    extend="both",
)
plt.colorbar(cntf)
plt.show()
```
