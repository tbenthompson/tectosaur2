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

Things to do here:
- show the difference between kappa=1 and kappa=2
- look at the error estimates from one of the relevant papers. how does the estimate vary with distance from a singularity? with the order of the singularity? what if only the derivatives are singular?
- maybe stage2 refinement should be modified near a singularity?

Fault tips:
- Identify or specify singularities and then make sure that the QBX and quadrature account for the singularities. This would be helpful for avoiding the need to have the sigmoid transition.
- *Would it be useful to use an interpolation that includes the end points so that I can easily make sure that slip goes to zero at a fault tip?* --> I should test this!

```{code-cell} ipython3
from config import setup, import_and_display_fnc

setup()
```

```{code-cell} ipython3
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    qbx_matrix2,
    single_layer_matrix,
    double_layer_matrix,
    adjoint_double_layer_matrix,
    hypersingular_matrix,
    stage1_refine,
    qbx_panel_setup,
    stage2_refine,
    pts_grid,
)
```

```{code-cell} ipython3
import quadpy

def clencurt(n1):
    """Computes the Clenshaw Curtis quadrature nodes and weights"""
    C = quadpy.c1.clenshaw_curtis(n1)
    return (C.points, C.weights)
```

```{code-cell} ipython3
log(np.sqrt(2) * 0.001) / log(0.03125)
```

```{code-cell} ipython3
panel_width = 0.125
nq = 6
t = sp.var("t")
fault, = stage1_refine([(t, t * 0, t)], gauss_rule(nq), control_points=[(0, 0, 1.0, panel_width)])
fault_expansions, = qbx_panel_setup(
    [fault], directions=[1], mult=0.5, singularities=np.array([[0,-1], [0,1]])
)
print(fault_expansions.pts[:,0])

print(fault.n_panels, fault.n_pts)
K = hypersingular_matrix
#K = double_layer_matrix
#K = single_layer_matrix
M = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=4)
M2 = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=5)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.log10(np.abs((M - M2) / M))[:,0,:])
plt.colorbar()
plt.subplot(1,2,2)
slip = np.cos(np.pi * 0.5 * fault.pts[:,1])
slip_err = M.dot(slip) - M2.dot(slip)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'b-', label='cos')

y = fault.pts[:,1]

slip = np.ones_like(fault.pts[:,1])
slip_err = M.dot(slip) - M2.dot(slip)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'r-', label='one')

slip = 1 + np.cos(np.pi * fault.pts[:,1])
slip_err = M.dot(slip) - M2.dot(slip)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'k-', label='1+cos')

plt.legend()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
panel_width = 0.125
nq = 6
t = sp.var("t")
fault, = stage1_refine([(t, t * 0, t)], gauss_rule(nq), control_points=[(0, 0, 1.0, panel_width)])
fault_expansions, = qbx_panel_setup(
    [fault], directions=[1], mult=0.5, singularities=np.array([[0,-1], [0,1]])
)
print(fault_expansions.pts[:,0])

print(fault.n_panels, fault.n_pts)
K = hypersingular_matrix
#K = double_layer_matrix
#K = single_layer_matrix
M = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=4, kappa=10)
M2 = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=5, kappa=10)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(np.log10(np.abs((M - M2) / M))[:,0,:])
plt.colorbar()
plt.subplot(2,2,2)
slip_cos = np.cos(np.pi * 0.5 * fault.pts[:,1])
slip_err = M.dot(slip_cos) - M2.dot(slip_cos)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'b-', label='cos')

y = fault.pts[:,1]

slip_ones = np.ones_like(fault.pts[:,1])
slip_err = M.dot(slip_ones) - M2.dot(slip_ones)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'r-', label='one')

slip_1cos = 1 + np.cos(np.pi * fault.pts[:,1])
slip_err = M.dot(slip_1cos) - M2.dot(slip_1cos)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'k-', label='1+cos')
plt.legend()

plt.subplot(2,2,3)
plt.plot(y, M2.dot(slip_ones)[:,0], 'r-', label='one')
plt.plot(y, M2.dot(slip_cos)[:,0], 'b-', label='cos')
plt.plot(y, M2.dot(slip_1cos)[:,0], 'k-', label='1+cos')
plt.ylim([-1, 2])
plt.legend()

plt.tight_layout()
plt.show()
```

## Convergence with r

```{code-cell} ipython3
panel_width = 0.75
nq = 16
t = sp.var("t")
fault, = stage1_refine([(t, t * 0, t)], gauss_rule(nq), control_points=[(0, 0, 1.0, panel_width)])
fault_expansions, = qbx_panel_setup(
    [fault], directions=[1], mult=0.5, singularities=np.array([[0,-1], [0,1]])
)
#print(fault_expansions.pts[:,0])

print(fault.n_panels, fault.n_pts)
K = hypersingular_matrix
#K = double_layer_matrix
#K = single_layer_matrix
Ms = []

M2 = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=20, kappa=3)

Ms = []
for p in range(4, 20, 2):
    M = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=p, kappa=3)
    Ms.append(M)
```

```{code-cell} ipython3
slip_errs = []
svs = []
for i in range(len(Ms)):
    slip = np.ones_like(fault.pts[:,1])
    #slip = np.cos(0.5 * np.pi * fault.pts[:,1])
    #slip = 0.5 + 0.5 * np.cos(np.pi * fault.pts[:,1])
    slip_err = Ms[i].dot(slip) - M2.dot(slip)
    svs.append(Ms[i].dot(slip))
    slip_errs.append(slip_err)
    plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), label=str(4 + 2 * i))
#plt.xlim([-1.1, -0.7])
plt.legend(loc='right')
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
np.array(svs)[:,-1,0]
```

```{code-cell} ipython3
np.array(slip_errs)[:,-1,0]
```

```{code-cell} ipython3
np.array(svs)[:,-1,0]
```

```{code-cell} ipython3
np.array(slip_errs)[:,-1,0]
```

## What if I use clenshaw-curtis and just set the endpoints to zero?

```{code-cell} ipython3
panel_width = 0.125
nq = 6
t = sp.var("t")
qx, qw = clencurt(nq)
fault, = stage1_refine([(t, t * 0, t)], (qx, qw), control_points=[(0, 0, 1.0, panel_width)])
fault_expansions, = qbx_panel_setup(
    [fault], directions=[1], mult=0.5, singularities=np.array([[0,-1], [0,1]])
)
print(fault_expansions.pts[:,0])

print(fault.n_panels, fault.n_pts)
K = hypersingular_matrix
#K = double_layer_matrix
#K = single_layer_matrix
M = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=8, kappa=10)
M2 = qbx_matrix2(K, fault, fault.pts, fault_expansions, p=9, kappa=10)
```

```{code-cell} ipython3
fault.panel_bounds
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(np.log10(np.abs((M - M2) / M))[:,0,:])
plt.colorbar()
plt.subplot(2,2,2)
slip_cos = np.cos(np.pi * 0.5 * fault.pts[:,1])
slip_err = M.dot(slip_cos) - M2.dot(slip_cos)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'b-', label='cos')

y = fault.pts[:,1]

slip_ones = np.ones_like(fault.pts[:,1])
slip_ones[:nq] = 1 + (fault.pts[:nq,1] - fault.panel_bounds[0,1]) / (fault.panel_bounds[0, 1] - fault.panel_bounds[0,0])
slip_ones[-nq:] = 1 - (fault.pts[-nq:,1] - fault.panel_bounds[-1,0]) / (fault.panel_bounds[-1, 1] - fault.panel_bounds[-1,0])
slip_err = M.dot(slip_ones) - M2.dot(slip_ones)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'r-', label='one')

def sigmoid(x0, W):
    return 1.0 / (1 + np.exp((fault.pts[:, 1] - x0) / W))

#slip_1cos = sigmoid(0.5, 0.05) - sigmoid(-0.5, 0.05)
slip_1cos = 0.5 + 0.5 * np.cos(np.pi * fault.pts[:,1])
slip_err = M.dot(slip_1cos) - M2.dot(slip_1cos)
plt.plot(fault.pts[:,1], np.log10(np.abs(slip_err[:,0])), 'k-', label='1+cos')
plt.legend()

plt.subplot(2,2,3)
plt.plot(y, M2.dot(slip_ones)[:,0], 'r-', label='one')
plt.plot(y, M2.dot(slip_cos)[:,0], 'b-', label='cos')
plt.plot(y, M2.dot(slip_1cos)[:,0], 'k-', label='1+cos')
plt.ylim([-1, 2])
plt.legend()
plt.subplot(2,2,4)
plt.plot(y, slip_ones, 'r-o', markersize=4.0, label='one')
plt.plot(y, slip_cos, 'b-o', markersize=4.0, label='cos')
plt.plot(y, slip_1cos, 'k-o', markersize=4.0, label='1+cos')
plt.legend()

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3














```

```{code-cell} ipython3
nq = 256
panel_width = 4.0

qx, qw = gauss_rule(nq)
#qx, qw = clencurt(nq)

def trial(qx, qw, panel_width, f):
    t = sp.var("t")

    cp = [(0, 0, 1.0, panel_width)]
    fault, = stage1_refine([(t, t * 0, t)], (qx, qw), control_points=cp)

    fault_expansions, = qbx_panel_setup([fault], directions=[0], p=10)

    fault_slip_to_fault_stress = qbx_matrix2(
        hypersingular_matrix, fault, fault.pts, fault_expansions
    )

#     from common import build_interpolator, interpolate_fnc
#     slip = 1 - np.abs(qx)
#     #slip[0] = 0
#     #slip[-1] = 0
#     evalx = np.linspace(-1, 1, 1000)
#     evalslip = interpolate_fnc(build_interpolator(qx), slip, evalx)
#     plt.plot(evalx, evalslip, 'k-')
#     plt.show()

    fy = fault.pts[:,1]
    slip = f(fault.pts[:,1])#np.ones(fault.n_pts)
#     slip[0] = 0
#     slip[-1] = 0
#     plt.plot(fy, slip)
#     plt.show()
    stress = fault_slip_to_fault_stress.dot(slip)
    plt.plot(fy, stress[:,0], 'r-')
    plt.plot(fy, stress[:,1], 'b-')
    plt.show()
```

```{code-cell} ipython3
def f(y):
    return np.cos(y * np.pi * 0.5)
```

```{code-cell} ipython3
trial(*gauss_rule(256), 4.0, f)

trial(*gauss_rule(64), 4.0, f)

trial(*gauss_rule(128), 1.0, f)

trial(*gauss_rule(8), 1.0 / 8.0, f)

trial(*gauss_rule(8), 1.0 / 16.0, f)
```

```{code-cell} ipython3
def approach_test(obs_pts, slip):
    panel_width = 0.24
    nq = 16
    fault, = stage1_refine([(t, t * 0, t)], gauss_rule(nq), control_points=[(0, 0, 1.0, panel_width)])
    obs_pts.shape
    V1 = hypersingular_matrix(fault, obs_pts).dot(slip(fault.pts[:,1]))[:,0]

    panel_width = 0.12
    nq = 32
    fault, = stage1_refine([(t, t * 0, t)], gauss_rule(nq), control_points=[(0, 0, 1.0, panel_width)])
    obs_pts.shape
    V2 = hypersingular_matrix(fault, obs_pts).dot(slip(fault.pts[:,1]))[:,0]

#     plt.plot(fault.pts[:,1], slip(fault.pts[:,1]))
#     plt.show()

    return V1 - V2
```

```{code-cell} ipython3

seq1 = []
yvs = np.linspace(-1.1, 1.1, 23)
for yv in yvs:
    dist = 2.0 ** -np.arange(10)
    obs_pts = np.stack((dist, np.full_like(dist, yv)), axis=1)
    #print(approach_test(obs_pts, lambda x: np.cos(x * np.pi * 0.5)))
    err = approach_test(obs_pts, lambda x: np.ones_like(x))
    seq1.append(err[6])
plt.plot(yvs, np.log10(np.abs(seq1)))
plt.show()
```

```{code-cell} ipython3
seq = []
yvs = np.linspace(-1.1, 1.1, 23)
for yv in yvs:
    dist = 2.0 ** -np.arange(10)
    obs_pts = np.stack((dist, np.full_like(dist, yv)), axis=1)
    #print(approach_test(obs_pts, lambda x: np.cos(x * np.pi * 0.5)))
    err = approach_test(obs_pts, lambda x: np.cos(x * np.pi * 0.5))
    seq.append(err[6])
plt.plot(yvs, np.log10(np.abs(seq)), 'r-')
plt.plot(yvs, np.log10(np.abs(seq1)), 'k-')
plt.show()
```
