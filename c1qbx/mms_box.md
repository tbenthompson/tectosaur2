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
%load_ext autoreload
%autoreload 2
```

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import common
import sympy as sp

%config InlineBackend.figure_format='retina'

n_q = 20
qx, qw = common.gauss_rule(n_q)

t = sp.symbols("t")
t01 = (t + 1) / 2
# counter-clockwise from the bottom left
bottom = common.symbolic_surface(t, t01, 0)
right = common.symbolic_surface(t, 1, t01)
top = common.symbolic_surface(t, 1 - t01, 1)
left = common.symbolic_surface(t, 0, 1 - t01)
sides = [bottom, right, top, left]
```

```{code-cell} ipython3
symbolic_combined = [common.symbolic_eval(t, qx, s) for s in sides]
```

```{code-cell} ipython3
box = [np.concatenate(f) for f in (zip(*symbolic_combined))]
box_quad_rule = (np.concatenate([qx] * 4), np.concatenate([qw / 2] * 4))
```

```{code-cell} ipython3
np.sum(box_quad_rule[1])
```

## Body force quadrature

```{code-cell} ipython3
nq_volume = 20
q_vol, qw_vol = common.gauss_rule(nq_volume)
q_vol = (q_vol + 1) / 2
qw_vol /= 2

qx_vol, qy_vol = np.meshgrid(q_vol, q_vol)
q2d_vol = np.array([qx_vol.flatten(), qy_vol.flatten()]).T.copy()
q2d_vol_wts = (qw_vol[:, None] * qw_vol[None, :]).flatten()
```

```{code-cell} ipython3
np.sum(q2d_vol_wts)
```

```{code-cell} ipython3
correct = 0.9460830703671830
correct - np.sum(np.cos(q2d_vol[:,0] * q2d_vol[:,1]) * q2d_vol_wts)
```

```{code-cell} ipython3
plt.plot(box[0], box[1])
plt.quiver(box[0], box[1], box[2], box[3])
plt.plot(q2d_vol[:, 0], q2d_vol[:, 1], "r.")
plt.xlim([-0.3, 1.3])
plt.ylim([-0.3, 1.3])
plt.show()
```

```{code-cell} ipython3
nobs = 200
offset = 0
zoomx = [offset, 1.0 - offset]
zoomy = [offset, 1.0 - offset]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obs2d = np.array([obsx.flatten(), obsy.flatten()]).T.copy()
```

```{code-cell} ipython3
freq_factor = 1.0
def soln_fnc(x, y):
    return np.sin(freq_factor * 2 * np.pi * x) * np.sin(freq_factor * 2 * np.pi * y)

def laplacian_fnc(x, y):
    return (
        -(2 * ((2 * freq_factor) ** 2))
        * np.pi ** 2
        * np.sin(freq_factor * 2 * np.pi * x)
        * np.sin(freq_factor * 2 * np.pi * y)
    )

# Ae = 25
# def soln_fnc(x, y):
#     return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) * Ae)

# def laplacian_fnc(x, y):
#     return 4 * Ae * soln_fnc(x, y) * (Ae * ((x - 0.5) ** 2 + (y - 0.5) ** 2) - 1)
```

```{code-cell} ipython3
fxy = laplacian_fnc(q2d_vol[:,0], q2d_vol[:,1])
correct = soln_fnc(obsx, obsy)
fxy_obs = laplacian_fnc(obsx, obsy)
```

```{code-cell} ipython3
plt.contourf(qx_vol, qy_vol, eterm.reshape(qx_vol.shape))
plt.colorbar()
plt.show()
```

```{code-cell} ipython3
plt.contourf(qx_vol, qy_vol, fxy.reshape(qx_vol.shape))
plt.colorbar()
plt.show()
```

```{code-cell} ipython3
hx = xs[1] - xs[0]
hy = ys[1] - ys[0]
hx, hy
```

```{code-cell} ipython3
dx2 = (correct[2:] - 2*correct[1:-1] + correct[:-2]) / (hx ** 2)
dy2 = (correct[:, 2:] - 2*correct[:, 1:-1] + correct[:, :-2]) / (hy ** 2)
```

```{code-cell} ipython3
laplacian = np.zeros_like(fxy_obs)
laplacian[1:-1] += dx2
laplacian[:,1:-1] += dy2
```

```{code-cell} ipython3
plt.figure(figsize = (12,5))
plt.subplot(1,3,1)
levels = np.linspace(np.min(fxy_obs), np.max(fxy_obs), 21)
cntf = plt.contourf(obsx, obsy, fxy_obs, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    fxy_obs,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.subplot(1,3,2)
levels = np.linspace(np.min(laplacian), np.max(laplacian), 21)
cntf = plt.contourf(obsx, obsy, laplacian, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    laplacian,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.subplot(1,3,3)
err = np.log10(np.abs(laplacian - fxy_obs))
levels = np.linspace(-5, 0, 11)
cntf = plt.contourf(obsx, obsy, err, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    err,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
def fundamental_soln_matrix(obsx, obsy, src_pts, src_wts):
    dx = obsx[:, None] - src_pts[None, :, 0]
    dy = obsy[:, None] - src_pts[None, :, 1]
    r2 = (dx ** 2) + (dy ** 2)
    r = np.sqrt(r2)
    G = (1.0 / (2 * np.pi)) * np.log(r) * src_wts[None, :]
    return G[:, None, :]
```

```{code-cell} ipython3
u_body_force = (
    fundamental_soln_matrix(obs2d[:, 0], obs2d[:, 1], q2d_vol, q2d_vol_wts)
    .dot(fxy)
    .reshape(obsx.shape)
)
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
levels = np.linspace(np.min(u_body_force), np.max(u_body_force), 21)
cntf = plt.contourf(obsx, obsy, u_body_force, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    u_body_force,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.subplot(1,3,2)
levels = np.linspace(np.min(correct), np.max(correct), 21)
cntf = plt.contourf(obsx, obsy, correct, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    correct,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.subplot(1,3,3)
err = np.log10(np.abs(u_body_force - correct))
levels = np.linspace(-3, 0, 21)
cntf = plt.contourf(obsx, obsy, err, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    err,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.show()
```

## Direct to surface eval

```{code-cell} ipython3
surf_vals = fundamental_soln_matrix(box[0], box[1], q2d_vol, q2d_vol_wts).dot(fxy)[:, 0]

A, _ = common.interaction_matrix(
    common.double_layer_matrix, box, box_quad_rule, box, box_quad_rule
)
A = A[:, 0, :]
lhs = -A - 0.5 * np.eye(A.shape[0])
surf_field = np.linalg.solve(lhs, surf_vals)
```

```{code-cell} ipython3
plt.plot(surf_field)
plt.show()
```

```{code-cell} ipython3
u_box_rough = (
    common.double_layer_matrix(box, box_quad_rule, obsx.flatten(), obsy.flatten())
    .dot(surf_field)
    .reshape(obsx.shape)
)
```

```{code-cell} ipython3
qx3, qw3 = common.gauss_rule(n_q * 3)
box_quad_rule3 = (np.concatenate([qx3] * 4), np.concatenate([qw3 / 2] * 4))

start_idx = 0
surf_refined = []
slip_refined = []
for i in range(len(symbolic_combined)):
    print(i)
    surf_refined.append(common.interp_surface(symbolic_combined[i], qx, qx3))
    end_idx = start_idx + symbolic_combined[i][0].shape[0]
    slip_refined.append(common.interp_fnc(surf_field[start_idx:end_idx], qx, qx3))
    start_idx = end_idx
```

```{code-cell} ipython3
box_refined = [np.concatenate(f) for f in (zip(*surf_refined))]
slip_refined = np.concatenate(slip_refined)
```

```{code-cell} ipython3
dx = box[0][1:] - box[0][:-1]
dy = box[1][1:] - box[1][:-1]
L = np.sqrt(dx ** 2 + dy ** 2)
Lc = np.zeros_like(box[0])
Lc[1:] = np.cumsum(L)

dx = box_refined[0][1:] - box_refined[0][:-1]
dy = box_refined[1][1:] - box_refined[1][:-1]
L = np.sqrt(dx ** 2 + dy ** 2)
Lc_refined = np.zeros_like(box_refined[0])
Lc_refined[1:] = np.cumsum(L)

plt.plot(Lc, surf_field)
plt.plot(Lc_refined, slip_refined)
plt.figure()
```

```{code-cell} ipython3
u_box = common.interior_eval(
    common.double_layer_matrix,
    box,
    box_quad_rule,
    surf_field,
    obsx.flatten(),
    obsy.flatten(),
    offset_mult=5,
    kappa=3,
    qbx_p=10,
    quad_rule_qbx=box_quad_rule3,
    surface_qbx=box_refined,
    slip_qbx=slip_refined,
    visualize_centers=True,
).reshape(obsx.shape)
```

## Full solution!

```{code-cell} ipython3
u_full = u_box + u_body_force
#u_full = u_box_rough + u_body_force
```

```{code-cell} ipython3
# levels = np.linspace(np.min(u_full), np.max(u_full), 21)
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([u_box_rough, u_box, u_body_force]):
    plt.subplot(1,3,1+i)
    levels = np.linspace(-1, 1, 21)
    cntf = plt.contourf(obsx, obsy, to_plot, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        to_plot,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.plot(box[0], box[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([correct, u_full]):
    plt.subplot(1,3,1+i)
    levels = np.linspace(-1, 1, 21)
    cntf = plt.contourf(obsx, obsy, to_plot, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        to_plot,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.plot(box[0], box[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    
plt.subplot(1,3,3)
to_plot = correct - u_full
levels = np.linspace(-0.1, 0.1, 21)
cntf = plt.contourf(obsx, obsy, to_plot, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    to_plot,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(box[0], box[1], "k-", linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.show()
```

## Trying a QBX to surface eval

```{code-cell} ipython3
qbx_center_x, qbx_center_y, qbx_r = common.qbx_choose_centers(box, box_quad_rule)
```

```{code-cell} ipython3
plt.plot(box[0], box[1])
plt.plot(qbx_center_x, qbx_center_y, "ro")
```

```{code-cell} ipython3
qbx_p = 10
```

```{code-cell} ipython3
qbx_nq = 2 * qbx_p + 1
qbx_qx, qbx_qw = common.trapezoidal_rule(qbx_nq)
qbx_qw *= np.pi
qbx_theta = np.pi * (qbx_qx + 1)

# The coefficient integral points will have shape (number of expansions,
# number of quadrature points).
qbx_eval_r = qbx_r * 0.5
qbx_x = qbx_center_x[:, None] + qbx_eval_r[:, None] * np.cos(qbx_theta)[None, :]
qbx_y = qbx_center_x[:, None] + qbx_eval_r[:, None] * np.sin(qbx_theta)[None, :]
```

```{code-cell} ipython3
Keval = fundamental_soln_matrix(qbx_x.flatten(), qbx_y.flatten(), q2d_vol, q2d_vol_wts)
kernel_ndim = Keval.shape[1]
qbx_u_matrix = Keval.reshape((*qbx_x.shape, kernel_ndim, q2d_vol.shape[0]))
```

```{code-cell} ipython3
# Compute the expansion coefficients in matrix form.
alpha = np.empty(
    (qbx_center_x.shape[0], kernel_ndim, qbx_p, q2d_vol.shape[0]), dtype=np.complex128
)
for L in range(qbx_p):
    C = 1.0 / (np.pi * (qbx_eval_r ** L))
    if L == 0:
        C /= 2.0
    oscillatory = qbx_qw[None, :, None] * np.exp(-1j * L * qbx_theta)[None, :, None]
    alpha[:, :, L, :] = C[:, None, None] * np.sum(
        qbx_u_matrix * oscillatory[:, :, None], axis=1
    )
QBX_EXPAND = alpha
```

```{code-cell} ipython3
QBX_EXPAND.shape
```

```{code-cell} ipython3
qbx_center_x.shape
```

```{code-cell} ipython3
box[0][:, None].shape
```

```{code-cell} ipython3
QBX_EVAL = common.qbx_eval_matrix(
    box[0][None, :], box[1][None, :], qbx_center_x, qbx_center_y, qbx_p=qbx_p
)[0]
```

```{code-cell} ipython3
body_force_matrix = np.real(np.sum(QBX_EVAL[:, None, :, None] * QBX_EXPAND, axis=2))[
    :, 0, :
]
```

```{code-cell} ipython3
plt.plot(body_force_matrix.dot(fxy))
plt.show()
```

```{code-cell} ipython3

```
