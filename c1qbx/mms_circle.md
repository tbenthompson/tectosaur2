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

# [DRAFT] Body forces

+++

## TODO: 

* The remaining fundamental issue is the singularity in the volume integral. Set up a way of testing the accuracy of this integral.
* There's something preventing convergence in the L2 and Linf norms that isn't preventing convergence in the L1 norm. That suggests that the problem is some kind of outlier. The problem might be the scalloping in the QBX zone!!! 
* Fix the scalloping issue!
* Measure the error just inside r < 0.99.
* Getting an exterior solution would be nice!

+++

For the Poisson equation with Dirichlet boundary conditions:
\begin{split}
\nabla u &= f  ~~ \textrm{in} ~~ \Omega\\
u &= g ~~ \textrm{on} ~~ \partial \Omega
\end{split}
`u_particular` is the integral:

\begin{equation}
v(x) = \int_{\Omega} G(x,y) f(y) dy
\end{equation}

which satisfies equation 1 but not 2.

Then, compute homogeneous solution, `u_homog` with appropriate boundary conditions:

\begin{split}
\nabla u^H &= 0 ~~ \textrm{in} ~~ \Omega \\
u^H &= g - v|_{\partial \Omega}  ~~ \textrm{on} ~~ \partial \Omega
\end{split}

So, first, I need to compute $g - v|_{\partial \Omega}$

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

sym_circle = common.symbolic_suvrface(t, sp.cos(theta), sp.sin(theta))
circle = common.symbolic_eval(t, circle_rule[0], sym_circle)
```

```{code-cell} ipython3
np.sum(circle_rule[1])
```

We will solve for the function

```{code-cell} ipython3
x, y = sp.symbols('x, y')
sym_soln = 2 + x + y + x**2 + y*sp.cos(6*x) + x*sp.sin(6*y)

sym_laplacian = (
    sp.diff(sp.diff(sym_soln, x), x) + 
    sp.diff(sp.diff(sym_soln, y), y)
)
soln_fnc = sp.lambdify((x, y), sym_soln, "numpy")
laplacian_fnc = sp.lambdify((x, y), sym_laplacian, "numpy")

sym_soln
```

```{code-cell} ipython3
sym_laplacian
```

## Body force quadrature

```{code-cell} ipython3
nq_r = 40
nq_theta = 80
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
nobs = 200
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
def fundamental_soln_matrix(obsx, obsy, src_pts, src_wts):
    dx = obsx[:, None] - src_pts[None, :, 0]
    dy = obsy[:, None] - src_pts[None, :, 1]
    r2 = (dx ** 2) + (dy ** 2)
    r = np.sqrt(r2)
    G = (1.0 / (2 * np.pi)) * np.log(r) * src_wts[None, :]
    return G[:, None, :]
```

```{code-cell} ipython3
u_particular = (
    fundamental_soln_matrix(obs2d[:, 0], obs2d[:, 1], q2d_vol, q2d_vol_wts)
    .dot(fxy)
    .reshape(obsx.shape)
)
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
levels = np.linspace(np.min(u_particular), np.max(u_particular), 21)
cntf = plt.contourf(obsx, obsy, u_particular, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    u_particular,
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
err = correct - u_particular
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
qbx_p = 8
mult = 1.0

if kappa != 1:
    refined_circle_rule = list(common.trapezoidal_rule(kappa * circle_rule[0].shape[0]))
    refined_circle_rule[1] *= np.pi
    refined_circle = common.symbolic_eval(t, refined_circle_rule[0], sym_circle)
else:
    refined_circle_rule = circle_rule
    refined_circle = circle

qbx_center_x, qbx_center_y, qbx_r = common.qbx_choose_centers(
    circle, circle_rule, mult=mult, direction=-1.0
)
```

```{code-cell} ipython3
plt.plot(circle[0], circle[1])
plt.quiver(circle[0], circle[1], circle[2], circle[3])
plt.plot(qbx_center_x, qbx_center_y, "ro")
```

```{code-cell} ipython3
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
surf_density = np.linalg.solve(A, bcs - surf_vals)
```

```{code-cell} ipython3
plt.plot(bcs)
plt.plot(surf_density)
plt.show()
```

```{code-cell} ipython3
u_homog_rough = (
    common.double_layer_matrix(circle, circle_rule, obsx.flatten(), obsy.flatten())
    .dot(surf_density)
    .reshape(obsx.shape)
)
```

```{code-cell} ipython3
refined_density = interp_matrix.dot(surf_density)
```

```{code-cell} ipython3
u_homog = common.interior_eval(
    common.double_layer_matrix,
    circle,
    circle_rule,
    surf_density,
    obsx.flatten(),
    obsy.flatten(),
    kappa=kappa,
    offset_mult=mult,
    qbx_p=qbx_p,
    quad_rule_qbx=refined_circle_rule,
    surface_qbx=refined_circle,
    slip_qbx=refined_density,
    visualize_centers=True,
).reshape(obsx.shape)
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
for i, to_plot in enumerate([u_homog, u_homog, correct - u_particular]):
    plt.subplot(1,3,1+i)
    if i == 0:
        levels = np.linspace(np.min(to_plot), np.max(to_plot), 21)
    else:
        levels = np.linspace(np.min(correct - u_particular), np.max(correct - u_particular), 21)
    
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
    plt.title(['$u^H$', '$u^H$', 'correct - $u^p$'][i])
plt.figure()
to_plot = np.abs(correct - u_particular - u_homog)
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
plt.title('err($u^H$)')
plt.show()
```

## Full solution!

```{code-cell} ipython3
#u_full = u_box + u_particular
u_full = u_homog + u_particular
```

```{code-cell} ipython3
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
    plt.title(['correct', 'u'][i])
    
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
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
for norm in [np.linalg.norm, lambda x: np.linalg.norm(x, ord=1), lambda x: np.linalg.norm(x, ord=np.inf)]:
    A = norm(correct[obs2d_mask_sq] - u_full[obs2d_mask_sq])
    B = norm(correct[obs2d_mask_sq])
    print(A, B, A/B)
```

```{code-cell} ipython3
E = correct[obs2d_mask_sq] - u_full[obs2d_mask_sq]
plt.plot(np.log10(sorted(np.abs(E).tolist()))[-50:])
plt.show()
```
