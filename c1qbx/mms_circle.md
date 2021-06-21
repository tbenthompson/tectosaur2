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

## TODO: 

* Deal with the homogeneous solution.
* Add back in the body force.
* The remaining fundamental issue is the singularity in the volume integral. Set up a way of testing the accuracy of this integral.

+++

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import common
import sympy as sp

%config InlineBackend.figure_format='retina'

n_q = 100
circle_rule = list(common.trapezoidal_rule(n_q))

t = sp.symbols("t")
theta = sp.pi * (t + 1)
circle_rule[1] *= np.pi

sym_circle = common.symbolic_surface(t, sp.cos(theta), sp.sin(theta))
circle = common.symbolic_eval(t, circle_rule[0], sym_circle)
```

```{code-cell} ipython3
np.sum(circle_rule[1])
```

## Body force quadrature

```{code-cell} ipython3
nq_r = 20
nq_theta = 40
qgauss, qgauss_w = common.gauss_rule(nq_r)
qtrap, qtrap_w = common.trapezoidal_rule(nq_theta)
r = 0.5 * (qgauss + 1)
r_w = qgauss_w * 0.5
theta = (qtrap + 1) * np.pi
theta_w = qtrap_w * np.pi

r_vol, theta_vol = [x.flatten() for x in np.meshgrid(r, theta)]
qx_vol = np.cos(theta_vol) * r_vol
qy_vol = np.sin(theta_vol) * r_vol

q2d_vol = np.array([qx_vol, qy_vol]).T.copy()
q2d_vol_wts = (r_w[None, :] * theta_w[:, None] * r[None, :]).flatten()
```

```{code-cell} ipython3
np.sum(q2d_vol_wts) - np.pi, np.sum(np.cos(theta_vol) * q2d_vol_wts), np.sum(r_vol * q2d_vol_wts) - 2.0943951023931954
```

```{code-cell} ipython3
plt.plot(circle[0], circle[1])
plt.quiver(circle[0], circle[1], circle[2], circle[3])
plt.plot(q2d_vol[:, 0], q2d_vol[:, 1], "r.")
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.show()
```

```{code-cell} ipython3
nobs = 100
offset = -0.1
zoomx = [-1.0 + offset, 1.0 - offset]
zoomy = [-1.0 + offset, 1.0 - offset]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obs2d = np.array([obsx.flatten(), obsy.flatten()]).T.copy()
obs2d_mask = np.sqrt(obs2d[:,0] ** 2 + obs2d[:,1] ** 2) <= 1
obs2d_mask_sq = obs2d_mask.reshape(obsx.shape)
```

```{code-cell} ipython3
freq_factor = 1.0
def soln_fnc(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return 2 + x + y + r * np.sin(freq_factor * np.pi * r)

def laplacian_fnc(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    T1 = 3 * np.pi * freq_factor * np.cos(np.pi * freq_factor * r)
    T2 = ((np.pi * freq_factor * r) ** 2 - 1) * np.sin(np.pi * freq_factor * r) / r
    return (T1 - T2)
```

```{code-cell} ipython3
fxy = laplacian_fnc(q2d_vol[:,0], q2d_vol[:,1])

correct = soln_fnc(obsx, obsy)
fxy_obs = laplacian_fnc(obsx, obsy)
```

```{code-cell} ipython3
plotx = q2d_vol[:,0].reshape((nq_theta, nq_r))
plt.contourf(plotx, q2d_vol[:,1].reshape(plotx.shape), soln_fnc(q2d_vol[:,0], q2d_vol[:,1]).reshape(plotx.shape))
plt.colorbar()
plt.show()
plotx = q2d_vol[:,0].reshape((nq_theta, nq_r))
plt.contourf(plotx, q2d_vol[:,1].reshape(plotx.shape), laplacian_fnc(q2d_vol[:,0], q2d_vol[:,1]).reshape(plotx.shape))
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
err = correct - u_body_force
levels = np.linspace(0, 4, 20)
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

+++

For the Poisson equation with Dirichlet boundary conditions:
\begin{split}
\nabla u &= f  ~~ \textrm{in} ~~ \Omega\\
u &= g ~~ \textrm{on} ~~ \partial \Omega
\end{split}
`u_body_force` is the integral:

\begin{equation}
v(x) = \int_{\Omega} G(x,y) f(y) dy
\end{equation}

which satisfies equation 1 but not 2.

Then, compute homogeneous solution with appropriate boundary conditions:

\begin{split}
\nabla u^H &= 0 ~~ \textrm{in} ~~ \Omega \\
u^H &= g - v|_{\partial \Omega}  ~~ \textrm{on} ~~ \partial \Omega
\end{split}

So, first, I need to compute $g - v|_{\partial \Omega}$

```{code-cell} ipython3
## This is g
bcs = soln_fnc(circle[0], circle[1])
```

```{code-cell} ipython3
## This is v|_{\partial \Omega}
surf_vals = fundamental_soln_matrix(circle[0], circle[1], q2d_vol, q2d_vol_wts).dot(fxy)[:, 0]
```

```{code-cell} ipython3
plt.plot(bcs, 'r-')
plt.plot(surf_vals, 'k-')
```

```{code-cell} ipython3
kappa = 3
refined_circle_rule = list(common.trapezoidal_rule(kappa * circle_rule[0].shape[0]))
refined_circle_rule[1] *= np.pi
refined_circle = common.symbolic_eval(t, refined_circle_rule[0], sym_circle)

qbx_p = 10
qbx_center_x, qbx_center_y, qbx_r = common.qbx_choose_centers(circle, circle_rule, direction=-1.0)
qbx_expand = common.qbx_expand_matrix(
    common.double_layer_matrix,
    refined_circle,
    refined_circle_rule,
    qbx_center_x,
    qbx_center_y,
    qbx_r,
    qbx_p=qbx_p,
)
qbx_eval = common.qbx_eval_matrix(
    circle[0][None, :],
    circle[1][None, :],
    qbx_center_x,
    qbx_center_y,
    qbx_p=qbx_p,
)[0]
A_raw = np.real(np.sum(qbx_eval[:, None, :, None] * qbx_expand, axis=2))
A_raw = A_raw[:, 0, :]
```

```{code-cell} ipython3
interp_matrix = np.zeros((A_raw.shape[1], A_raw.shape[0]))

for i in range(refined_circle_rule[0].shape[0]):
    offset = i % kappa
    if offset == 0:
        match = i // kappa
        interp_matrix[i, match] = 1.0
    else:
        below = i // kappa
        above = below + 1
        if above == interp_matrix.shape[1]:
            above = 0
        interp_matrix[i, below] = (kappa - offset) / kappa
        interp_matrix[i, above] = offset / kappa

plt.plot(circle_rule[0], bcs)
plt.plot(refined_circle_rule[0], interp_matrix.dot(bcs))
```

```{code-cell} ipython3
A = A_raw.dot(interp_matrix)

A2, _ = common.interaction_matrix(
    common.double_layer_matrix, circle, circle_rule, circle, circle_rule
)
A2 = A2[:,0,:]

print(A[:5,0], A2[:5,0])
print(np.max(np.abs(A - A2)), np.max(np.abs(A)))
```

```{code-cell} ipython3
lhs = A - 0.5 * np.eye(A.shape[0])
surf_field = np.linalg.solve(lhs, bcs - surf_vals)
```

```{code-cell} ipython3
plt.plot(bcs)
plt.plot(surf_field)
plt.show()
```

```{code-cell} ipython3
u_box_rough = (
    common.double_layer_matrix(circle, circle_rule, obsx.flatten(), obsy.flatten())
    .dot(surf_field)
    .reshape(obsx.shape)
)
```

```{code-cell} ipython3
np.max(u_box_rough[obs2d_mask_sq]), np.min(u_box_rough[obs2d_mask_sq])
```

```{code-cell} ipython3
np.max(bcs), np.min(bcs)
```

```{code-cell} ipython3
np.max(correct[obs2d_mask_sq]), np.min(correct[obs2d_mask_sq])
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([u_box_rough, u_box_rough, correct - u_body_force]):
    plt.subplot(1,3,1+i)
    if i == 0:
        levels = np.linspace(np.min(to_plot), np.max(to_plot), 21)
    else:
        levels = np.linspace(np.min(correct - u_body_force), np.max(correct - u_body_force), 21)
    
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
    plt.plot(circle[0], circle[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.title(['u\_box\_rough', 'u\_box\_rough', 'correct - u\_body\_force'][i])
plt.figure()
to_plot = np.abs(correct - u_body_force - u_box_rough)
levels = np.linspace(0, 0.25, 21)
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
plt.plot(circle[0], circle[1], "k-", linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.title('err(u\_box\_rough)')
plt.show()
```

```{code-cell} ipython3
# def trig_interp(f, N_out):
#     Npad_right = (N_out - f.shape[0]) // 2
#     Npad_left = N_out - f.shape[0] - Npad_right
#     F = np.fft.fft(f)
#     F = np.fft.fftshift(F)
#     F = np.concatenate((np.zeros(Npad_left), F, np.zeros(Npad_right)))
#     F = np.fft.ifftshift(F)
#     fi = np.fft.ifft(F) * F.shape[0] / f.shape[0]
#     return np.real(fi)

# np.random.seed(0)
# f = np.random.rand(21)
# fi = trig_interp(f, 180)
# x = np.linspace(1, fi.shape[0] + 1, f.shape[0] + 1)
# x = x[:-1]
# plt.plot(x, f)
# xi = np.arange(1, fi.shape[0] + 1)
# plt.plot(xi, fi)
# plt.show()

# circle_rule_qbx = common.trapezoidal_rule(3 * qx.shape[0])
# circle_qbx = []
# # So far, we've defined surfaces as five element tuples consisting of:
# # (x, y, normal_x, normal_y, jacobian)
# for f in circle[:5]:
#     circle_qbx.append(trig_interp(f, circle_rule_qbx[0].shape[0]))
# surf_field_qbx = trig_interp(surf_field, circle_rule_qbx[0].shape[0])

# circle_qbx[0]

# u_box = common.interior_eval(
#     common.double_layer_matrix,
#     circle,
#     (qx, qw),
#     surf_field,
#     obsx.flatten(),
#     obsy.flatten(),
#     offset_mult=5,
#     kappa=3,
#     qbx_p=10,
#     quad_rule_qbx=circle_rule_qbx,
#     surface_qbx=circle_qbx,
#     slip_qbx=surf_field_qbx,
#     visualize_centers=True,
# ).reshape(obsx.shape)
```

## Full solution!

```{code-cell} ipython3
#u_full = u_box + u_body_force
u_full = u_box_rough + u_body_force
```

```{code-cell} ipython3
# levels = np.linspace(np.min(u_full), np.max(u_full), 21)
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([u_box_rough, 0 * u_box_rough, correct - u_body_force]):
    plt.subplot(1,3,1+i)
    levels = np.linspace(np.min(correct - u_body_force), np.max(correct - u_body_force), 21)
#     if i <= 1:
#         levels = np.linspace(np.min(to_plot), np.max(to_plot), 21)
#     else:
#         levels = np.linspace(-1, 1, 21)
    
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
    plt.plot(circle[0], circle[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.title(['u\_box\_rough', 'u\_box', 'correct - u\_body\_force'][i])
    
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([correct, u_full]):
    plt.subplot(1,3,1+i)
    levels = np.linspace(np.min(correct), np.max(correct), 21)
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
    plt.plot(circle[0], circle[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.title(['correct', 'u\_full'][i])
    
plt.subplot(1,3,3)
to_plot = np.abs(correct - u_full)
levels = np.linspace(0, 1.0, 21)
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
plt.plot(circle[0], circle[1], "k-", linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.title('error')
plt.show()
```

```{code-cell} ipython3
A = np.linalg.norm(correct[obs2d_mask_sq] - u_full[obs2d_mask_sq])
B = np.linalg.norm(correct[obs2d_mask_sq])
A, B, A/B
```

```{code-cell} ipython3

```
