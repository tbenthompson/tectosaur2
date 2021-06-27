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

# [DRAFT] A box code for volumetric integration

+++

### Next steps:

* Cite the Ethridge and Greengard paper. 
* Build the singular integrals and do a single box test for those. 
* Do a nine box grid with singular integrals. 
* Split into nearfield and far-field.

+++

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import common
import sympy as sp

%config InlineBackend.figure_format='retina'
```

```{code-cell} ipython3
x, y = sp.symbols("x, y")
```

```{code-cell} ipython3
# sym_soln = 2 + x + y + x ** 2 + y * sp.cos(6 * x) + x * sp.sin(6 * y)
```

```{code-cell} ipython3
gaussian_centers = np.array([[0.1, 0.1], [0, 0], [-0.15, 0.1]])
alpha = 250
ethridge_sym_soln = 0
for i in range(3):
    r2_i = (x - gaussian_centers[i, 0]) ** 2 + (y - gaussian_centers[i, 1]) ** 2
    ethridge_sym_soln += sp.exp(-alpha * r2_i)
```

```{code-cell} ipython3
ethridge_sym_laplacian = sp.diff(sp.diff(ethridge_sym_soln, x), x) + sp.diff(
    sp.diff(ethridge_sym_soln, y), y
)
ethridge_soln_fnc = sp.lambdify((x, y), ethridge_sym_soln, "numpy")
ethridge_laplacian_fnc = sp.lambdify((x, y), ethridge_sym_laplacian, "numpy")
```

```{code-cell} ipython3
def fundamental_soln_matrix(obs_pts, src_pts):
    dx = obs_pts[:, None, 0] - src_pts[None, :, 0]
    dy = obs_pts[:, None, 1] - src_pts[None, :, 1]
    r2 = (dx ** 2) + (dy ** 2)
    r = np.sqrt(r2)
    G = (1.0 / (2 * np.pi)) * np.log(r)
    return G[:, None, :]
```

```{code-cell} ipython3
import pickle

with open("test_integral.pkl", "rb") as f:
    xy_integral = pickle.load(f)
```

```{code-cell} ipython3
ox, oy = sp.symbols("ox, oy")
xy_soln_fnc = sp.lambdify((ox, oy), xy_integral, "numpy")


def xy_laplacian_fnc(x, y):
    return (1 - x ** 2) * (1 - y ** 2)
```

## Singular quadrature, poor convergence.

```{code-cell} ipython3
def run(obs_pt, nq):
    q_vol, qw_vol = common.gauss_rule(nq)

    qx_vol, qy_vol = np.meshgrid(q_vol, q_vol)
    q2d_vol = np.array([qx_vol.flatten(), qy_vol.flatten()]).T.copy()
    q2d_vol_wts = (qw_vol[:, None] * qw_vol[None, :]).flatten()
    fxy = xy_laplacian_fnc(q2d_vol[:, 0], q2d_vol[:, 1])

    u_particular = (
        (fundamental_soln_matrix(obs_pt, q2d_vol) * q2d_vol_wts[None, None, :])
        .dot(fxy)
        .reshape(obs_pt.shape[0])
    )
    return u_particular
```

```{code-cell} ipython3
obs_pt = np.array([[-0.9, -0.9]])
correct = xy_soln_fnc(obs_pt[0, 0], obs_pt[0, 1])

steps = np.arange(2, 500, 6)
ests = np.array([run(obs_pt, s) for s in steps])

difference = np.linalg.norm(ests - correct, axis=1) / np.linalg.norm(correct)

plt.plot(steps, np.log10(difference))
plt.show()
```

## Precomputing singular integrals

### MOVE STUFF UP HERE

```{code-cell} ipython3
def clencurt(n1):
    """Computes the Clenshaw Curtis quadrature nodes and weights"""
    C = quadpy.c1.clenshaw_curtis(n1)
    return (C.points, C.weights)


# TODO: is there a quadpy function that does tensor products?
def tensor_product(x, w):
    rect_x, rect_y = np.meshgrid(x, x)
    rect_pts = np.array([rect_x.flatten(), rect_y.flatten()]).T
    rect_w = np.outer(w, w).flatten()
    return rect_pts, rect_w


def clencurt_2d(n):
    return tensor_product(*clencurt(n))


def cheblob(n):
    """Computes the chebyshev lobatto."""
    pts = clencurt(n)[0]
    wts = (-1) ** np.arange(n).astype(np.float64)
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return pts, wts  # tensor_product(pts, wts)
```

## Two-dimensional interpolation

```{code-cell} ipython3
import quadpy
```

```{code-cell} ipython3


eps = np.finfo(float).eps


def barycentric_tensor_product(evalx, evaly, interp_pts, interp_wts, fnc_vals):
    """
    eval_pts is (N, 2)
    interp_pts is (Q,)
    interp_wts is (Q,)
    fnc_vals is (P, Q^2)
    """

    dx = evalx[:, None] - interp_pts
    dy = evaly[:, None] - interp_pts

    idx0, idx1 = np.where(dx == 0)
    dx[idx0, idx1] = eps
    idx0, idx1 = np.where(dy == 0)
    dy[idx0, idx1] = eps

    kernelX = interp_wts[None, :] / dx
    kernelY = interp_wts[None, :] / dy
    kernel = (kernelX[:, None, :] * kernelY[:, :, None]).reshape(
        (-1, fnc_vals.shape[1])
    )
    return (
        np.sum(kernel[None, :] * fnc_vals[:, None, :], axis=2)
        / np.sum(kernel, axis=1)[None, :]
    )
```

```{code-cell} ipython3
nobs = 200
zoomx = np.array([-1, 1])
zoomy = np.array([-1, 1])
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obsx_flat = obsx.flatten()
obsy_flat = obsy.flatten()


nI = 3
Ix, Iwts = cheblob(nI)
Ipts, Iwts2d = tensor_product(Ix, Iwts)
F = xy_laplacian_fnc(Ipts[:, 0], Ipts[:, 1])
F_interp = barycentric_tensor_product(obsx_flat, obsy_flat, Ix, Iwts, np.array([F]))
F_interp2d = F_interp.reshape(obsx.shape)
F_correct = xy_laplacian_fnc(obsx_flat, obsy_flat).reshape(obsx.shape)
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
levels = np.linspace(np.min(F_correct), np.max(F_correct), 7)
cntf = plt.contourf(
    Ipts[:, 0].reshape((nI, nI)),
    Ipts[:, 1].reshape((nI, nI)),
    F.reshape((nI, nI)),
    levels=levels,
    extend="both",
)
plt.contour(
    Ipts[:, 0].reshape((nI, nI)),
    Ipts[:, 1].reshape((nI, nI)),
    F.reshape((nI, nI)),
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(1,3,2)
levels = np.linspace(np.min(F_correct), np.max(F_correct), 7)
cntf = plt.contourf(obsx, obsy, F_interp2d, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    F_interp2d,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(1, 3, 3)
levels = np.linspace(-5, 1, 7)
err = np.log10(np.abs(F_correct - F_interp2d))
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
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.show()
```

## A box code

### TODO: Restrict adjacency sizes

+++

### TODO: Compute a set of leaves to do integration much faster?

```{code-cell} ipython3
q1 = clencurt_2d(4)
q2 = clencurt_2d(7)
interp1 = cheblob(4)
interp2 = cheblob(7)
```

```{code-cell} ipython3
import matplotlib.patches as patches

from dataclasses import dataclass


@dataclass()
class TreeLevel:
    fhigh: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    parents: np.ndarray
    is_leaf: np.ndarray


def build_box_tree(f, start_centers, start_sizes, max_levels, tol):
    parents = np.zeros(start_centers.shape[0])
    centers = start_centers
    sizes = start_sizes
    levels = []
    for i in range(max_levels):
        box_low_pts = q1[0][None, :] * 0.5 * sizes[:, None, :] + centers[:, None, :]
        box_high_pts = q2[0][None, :] * 0.5 * sizes[:, None, :] + centers[:, None, :]
        box_quad_wts = q2[1][None, :] * 0.25 * sizes[:, 0, None] * sizes[:, 1, None]

        f_high = f(
            box_high_pts[:, :, 0].ravel(), box_high_pts[:, :, 1].ravel()
        ).reshape((box_low_pts.shape[0], interp2[0].shape[0], interp2[0].shape[0]))

        # Because the Chebyshev Lobatto/Clenshaw Curtis points are nested, a 2N - 1 point
        # rule contains an N point rule inside it.
        f_low = f_high[:, ::2, ::2]

        f_high_flat = f_high.reshape((centers.shape[0], -1))
        f_low_flat = f_low.reshape((centers.shape[0], -1))

        # Interpolate to get an error estimate. Note that this error estimate will be
        # very conservative because it's estimating the error in the N low accuracy points
        # but we will end up using the 2N - 1 high accuracy points.
        f_high_interp = barycentric_tensor_product(
            q2[0][:, 0], q2[0][:, 1], interp1[0], interp1[1], f_low_flat
        )
        err = np.linalg.norm(f_high_interp - f_high_flat, axis=1)

        # Don't refine if we're at the last level.
        if i == max_levels - 1:
            refine_boxes = np.array([], dtype=np.int64)
        else:
            refine_boxes = np.where(err > tol)[0]

        is_leaf = np.ones(centers.shape[0], dtype=bool)
        is_leaf[refine_boxes] = False

        levels.append(TreeLevel(f_high_flat, centers, sizes, parents, is_leaf))
        if refine_boxes.shape[0] == 0:
            break

        refine_centers = centers[refine_boxes]
        bump = sizes[refine_boxes] / 4

        parents = np.repeat(np.arange(centers.shape[0])[refine_boxes], 4)
        centers = np.concatenate(
            [
                refine_centers + np.array([bump[:, 0], bump[:, 1]]).T,
                refine_centers + np.array([-bump[:, 0], bump[:, 1]]).T,
                refine_centers + np.array([bump[:, 0], -bump[:, 1]]).T,
                refine_centers + np.array([-bump[:, 0], -bump[:, 1]]).T,
            ]
        )
        sizes = np.repeat(sizes[refine_boxes] / 2, 4, axis=0)
    return levels
```

```{code-cell} ipython3
%%time
tree = build_box_tree(laplacian_fnc, np.array([[0, 0]]), np.array([[1, 1]]), 10, 0.1)
```

```{code-cell} ipython3
tree = build_box_tree(
    lambda x, y: np.cos(np.exp(7 * x)), np.array([[0, 0]]), np.array([[1, 1]]), 10, 0.1
)

for level in tree:
    centers = level.centers
    sizes = level.sizes
    for i in range(centers.shape[0]):
        plt.gca().add_patch(
            patches.Rectangle(
                (centers[i][0] - sizes[i][0] / 2, centers[i][1] - sizes[i][1] / 2),
                sizes[i][0],
                sizes[i][1],
                edgecolor="k",
                facecolor="none",
            )
        )
plt.xlim([-0.55, 0.55])
plt.ylim([-0.55, 0.55])
plt.show()

S = 0
for level in tree:
    if not np.any(level.is_leaf):
        continue
    leaf_sizes = level.sizes[level.is_leaf]
    leaf_centers = level.sizes[level.is_leaf]
    leaf_f = level.fhigh[level.is_leaf]

    box_high_pts = (
        q2[0][None, :] * 0.5 * leaf_sizes[:, None, :] + leaf_centers[:, None, :]
    )
    box_quad_wts = (
        q2[1][None, :] * 0.25 * leaf_sizes[:, 0, None] * leaf_sizes[:, 1, None]
    )

    S += np.sum(box_quad_wts.ravel() * leaf_f.ravel())
S
```

## Volumetric Green's function integrals

```{code-cell} ipython3
obs_test = np.array([[-0.10, -0.10]])
```

```{code-cell} ipython3
testF = lambda x, y: laplacian_fnc(x, y)
```

```{code-cell} ipython3
tree = build_box_tree(testF, np.array([[0, 0]]), np.array([[1, 1]]), 15, 0.000001)
S = np.zeros_like(obsx_test)
for level in tree:
    leaf_sizes = level.sizes[level.is_leaf]
    leaf_centers = level.centers[level.is_leaf]
    leaf_f = level.fhigh[level.is_leaf]

    box_high_pts = (
        q2[0][None, :] * 0.5 * leaf_sizes[:, None, :] + leaf_centers[:, None, :]
    )
    box_quad_wts = (
        q2[1][None, :] * 0.25 * leaf_sizes[:, 0, None] * leaf_sizes[:, 1, None]
    )

    G = (
        fundamental_soln_matrix(obs_test, box_high_pts.reshape((-1, 2)))[:, 0, :]
        * box_quad_wts.ravel()[None, :]
    )
    S += G.dot(leaf_f.ravel())
S
```

```{code-cell} ipython3
len(tree)
```

```{code-cell} ipython3
correct = soln_fnc(obsx_test, obsy_test)
```

```{code-cell} ipython3
np.linalg.norm(S - correct) / np.linalg.norm(correct)
```

## Pre-computing near-field coefficients

```{code-cell} ipython3
q2 = clencurt_2d(7)
```

```{code-cell} ipython3
obs_pt = q2[0][0]
obs_pt
```

What is the implied polynomial when `f(src_pt) = 1` and `f(the other pts) = 0`? 

```{code-cell} ipython3
idx = 22
single_nonzero = np.zeros((1, q2[0].shape[0]))
single_nonzero[0, idx] = 1
src_pt = q2[0][idx]
src_pt
```

```{code-cell} ipython3
out = barycentric_tensor_product(
    obsx_flat * 2, obsy_flat * 2, interp2[0], interp2[1], single_nonzero
).reshape(obsx.shape)
```

```{code-cell} ipython3
levels = np.linspace(np.min(out), np.max(out), 7)
cntf = plt.contourf(2 * obsx, 2 * obsy, out, levels=levels, extend="both")
plt.contour(
    2 * obsx,
    2 * obsy,
    out,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(src_pt[0:1], src_pt[1:2], "ro")
plt.colorbar(cntf)
plt.xlim(2 * zoomx)
plt.ylim(2 * zoomy)
plt.show()
```

```{code-cell} ipython3
import tanh_sinh

n_digits = 7
tol = 10 ** -(n_digits + 1)
tol


def integrand(x, ys):
    xs = np.full_like(ys, x)
    interp_val = barycentric_tensor_product(
        xs, ys, interp2[0], interp2[1], single_nonzero.reshape((1, -1))
    )[0, :]
    src_pts = np.array([xs, ys]).T.copy()
    out = (
        fundamental_soln_matrix(obs_pt.reshape((1, -1)), src_pts)[0, 0, :] * interp_val
    )
    return out


def inner(xs):
    out = np.empty_like(xs)
    for i, x in enumerate(xs):
        f = lambda y: integrand(x, y)
        I1, error_est = tanh_sinh.integrate(
            f,
            -1.0,
            obs_pt[1],
            tol,
        )
        I2, error_est = tanh_sinh.integrate(
            f,
            obs_pt[1],
            1.0,
            tol,
        )
        out[i] = I1 + I2
    return out


f = lambda x: inner(obs_pt, x)
I1, error_est = tanh_sinh.integrate(inner, 0, obs_pt[0], tol)
I2, error_est = tanh_sinh.integrate(inner, obs_pt[0], 1.0, tol)
result = I1 + I2
```

```{code-cell} ipython3
result
```

```{code-cell} ipython3
import mpmath as mp
```

```{code-cell} ipython3
n_digits = 20
mp.dps = n_digits
tol = 10 ** -(n_digits + 1)
mp_eps = tol * 10
```

```{code-cell} ipython3
obs_pt
```

```{code-cell} ipython3
grid = np.array(grid)
```

```{code-cell} ipython3
np.save("obs_pt0_grid.npy", grid)
```

```{code-cell} ipython3
def mp_barycentric_tensor_product(evalx, evaly, interp_pts, interp_wts, fnc_vals):
    dx = evalx - interp_pts
    dy = evaly - interp_pts

    idx0 = np.where(dx == 0)
    dx[idx0] = mp_eps
    idx0 = np.where(dy == 0)
    dy[idx0] = mp_eps

    kernelX = interp_wts / dx
    kernelY = interp_wts / dy
    kernel = (kernelX[None, :] * kernelY[:, None]).reshape(fnc_vals[0].shape)
    return kernel.dot(fnc_vals[0]) / np.sum(kernel)


def mp_fundamental_soln(obsx, obsy, srcx, srcy):
    dx = obsx - srcx
    dy = obsy - srcy
    r2 = (dx ** 2) + (dy ** 2)
    r = mp.sqrt(r2)
    return (1.0 / (2 * mp.pi)) * mp.log(r)


def mp_integral(n, m, x, y):
    #     interp_val = mp_barycentric_tensor_product(
    #         x, y, interp2[0], interp2[1], single_nonzero
    #     )
    interp_val = mp.chebyt(n, x) * mp.chebyt(m, y)
    out = mp_fundamental_soln(obs_pt[0], obs_pt[1], x, y) * interp_val
    return out


grid = []
for n in range(10):
    for m in range(10):
        result = mp.quad(
            lambda x, y: mp_integral(n, m, x, y),
            [-1, obs_pt[0], 1],
            [-1, obs_pt[1], 1],
            error=True,
            verbose=True,
        )
        grid.append((n, m, result))
        print(grid[-1])
        print(len(grid))
```

```{code-cell} ipython3
def fun(x, y):
    return (
        fundamental_soln_matrix(obsx, obsy, np.array([[x, y]]), np.array([1.0])).dot(
            laplacian_fnc(np.array([x]), np.array([y]))
        )
    )[0]
```
