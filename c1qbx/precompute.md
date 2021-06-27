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

# [DRAFT] Precomputing near-field volumetric integrals.

+++

Thoughts:
* mpmath is slow!
* do I really need 35 digits of precision? nope. 30 should be enough.
* yes, I want to make sure that these integrals are 100% solid and undeniably correct.
* but I could run for a small Chebyshev order, maybe 5 terms total?
* and I should test on a smaller problem!

+++

## Setting up a test problem

```{code-cell} ipython3
# New packages!
import sympy as sp
from mpmath import mp
n_digits = 30
mp.dps = n_digits

%config InlineBackend.figure_format='retina'
```

```{code-cell} ipython3
import pickle

with open("test_integral.pkl", "rb") as f:
    xy_integral = pickle.load(f)
```

```{code-cell} ipython3
ox, oy = sp.symbols("ox, oy")
xy_soln_fnc = sp.lambdify((ox, oy), xy_integral, "mpmath")

def xy_laplacian_fnc(x, y):
    return (1 - x) * (1 - y)
```

## Pre-computing near-field coefficients

```{code-cell} ipython3
mp_C = mp.mpf('0.5') / mp.pi
def fundamental_soln(obsx, obsy, srcx, srcy):
    dx = obsx - srcx
    dy = obsy - srcy
    r2 = (dx ** 2) + (dy ** 2)
    r = mp.sqrt(r2)
    return mp_C * mp.log(r)

def volume_integral(obsx, obsy, n, m, x, y):
    interp_val = mp.chebyt(n, x) * mp.chebyt(m, y)
    out = fundamental_soln(obsx, obsy, x, y) * interp_val
    return out
```

```{code-cell} ipython3
:tags: []

def compute_for_pt(n_chebyshev_terms, obsx, obsy, verbose=False):
    mp.dps = n_digits
    print('Computing integrals for point:', obsx, obsy)
    grid = []
    for n in range(n_chebyshev_terms):
        for m in range(n_chebyshev_terms):
            integral, error_estimate = mp.quad(
                lambda x, y: volume_integral(obsx, obsy, n, m, x, y),
                [mp.mpf('-1'), obsx, mp.mpf('1')],
                [mp.mpf('-1'), obsy, mp.mpf('1')],
                error=True,
                verbose=verbose
            )
            py_integral = float(integral)
            grid.append((n, m, py_integral, integral, error_estimate))
            if verbose:
                print(n, m, py_integral, integral, error_estimate)
    return grid
```

```{code-cell} ipython3
grid = compute_for_pt(2, mp.mpf('-0.99'), mp.mpf('-0.99'), verbose=True)
```

```{code-cell} ipython3
est = grid[0][3] - grid[1][3] - grid[2][3] + grid[3][3]
true = xy_soln_fnc(mp.mpf('-0.99'), mp.mpf('-0.99'))
est, true, est - true
```

```{code-cell} ipython3
:tags: []

n_chebyshev_terms = 5
chebyshev_pts = [mp.cos(mp.pi * mp.mpf(i) / (n_chebyshev_terms - 1)) for i in range(n_chebyshev_terms)]
chebyshev_pts = np.array(chebyshev_pts)
```

```{code-cell} ipython3
inputs = []
for i in range(n_chebyshev_terms):
    for j in range(n_chebyshev_terms):
        inputs.append((n_chebyshev_terms, chebyshev_pts[i], chebyshev_pts[j]))
```

```{code-cell} ipython3
:tags: []

import multiprocessing
p = multiprocessing.Pool()
full_grid = np.array(p.starmap(compute_for_pt, inputs))
```

```{code-cell} ipython3
:tags: []

np.save("coincident_grid_new.npy", full_grid)
```

```{code-cell} ipython3
full_grid = np.load("coincident_grid_new.npy", allow_pickle=True)
```

```{code-cell} ipython3
integrals = np.array(np.array(full_grid)[:,:,2], dtype=np.float64)
```

```{code-cell} ipython3
nobs = 200
zoomx = np.array([-0.99, 0.99])
zoomy = np.array([-0.99, 0.99])
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obsx_flat = obsx.flatten()
obsy_flat = obsy.flatten()
```

```{code-cell} ipython3
est = integrals[:,0] - integrals[:,1] - integrals[:,5] + integrals[:,6]
```

```{code-cell} ipython3
est
```

```{code-cell} ipython3
for i in range(1,n_chebyshev_terms-1):
    for j in range(1,n_chebyshev_terms-1):
        err = est[i * n_chebyshev_terms + j] - xy_soln_fnc(chebyshev_pts[i], chebyshev_pts[j])
        print(err)
```

```{code-cell} ipython3
def compute_grid(multx, multy, offsetx, offsety):
    X, Y = np.meshgrid(multx * chebyshev_pts + offsetx, multy * chebyshev_pts + offsety)
    grid_input = zip(
        [n_chebyshev_terms] * n_chebyshev_terms ** 2, 
        X.ravel(), 
        Y.ravel()
    )
    print(list(grid_input)[0])
#     p = multiprocessing.Pool()
#     return np.array(p.starmap(compute_for_pt, grid_input))
```

```{code-cell} ipython3
np.save("coincident_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('0.0'), mp.mpf('0.0')))
np.save("adj1_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('2.0'), mp.mpf('2.0')))
np.save("adj2_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('0.0'), mp.mpf('2.0')))
np.save("adj3_grid.npy", compute_grid(mp.mpf('2.0'), mp.mpf('2.0'), mp.mpf('3.0'), mp.mpf('3.0')))
np.save("adj4_grid.npy", compute_grid(mp.mpf('2.0'), mp.mpf('2.0'), mp.mpf('1.0'), mp.mpf('3.0')))
```

```{code-cell} ipython3
import quadpy

def clencurt(n1):
    """Computes the Clenshaw Curtis quadrature nodes and weights"""
    C = quadpy.c1.clenshaw_curtis(n1)
    return (C.points[::-1], C.weights[::-1])

# TODO: is there a quadpy function that does tensor products?
def tensor_product(x, w):
    rect_x, rect_y = np.meshgrid(x, x)
    rect_pts = np.array([rect_x.flatten(), rect_y.flatten()]).T
    rect_w = np.outer(w, w).flatten()
    return rect_pts, rect_w

def cheblob(n):
    """Computes the chebyshev lobatto."""
    pts = clencurt(n)[0]
    wts = (-1) ** np.arange(n).astype(np.float64)
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return pts, wts  # tensor_product(pts, wts)

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
Ix, Iwts = cheblob(n_chebyshev_terms)
Ipts, Iwts2d = tensor_product(chebyshev_pts, Iwts)
F = est
F_interp = barycentric_tensor_product(obsx_flat, obsy_flat, Ix, Iwts, np.array([F]))
F_interp2d = F_interp.reshape(obsx.shape)
xy_soln_fnc_np = sp.lambdify((ox, oy), xy_integral, "numpy")
F_correct = xy_soln_fnc_np(obsx_flat, obsy_flat).reshape(obsx.shape)
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
nI = n_chebyshev_terms
plt.figure(figsize=(8.5,8.5))
plt.subplot(2,2,1)
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
plt.plot(Ipts[:,0], Ipts[:,1], 'ro', markersize=0.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(2,2,2)
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


plt.subplot(2,2,3)
levels = np.linspace(np.min(F_correct), np.max(F_correct), 7)
cntf = plt.contourf(obsx, obsy, F_correct, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    F_correct,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(2, 2, 4)
err = F_correct - F_interp2d
levels = np.linspace(-0.003, 0.003, 7)
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
plt.plot(Ipts[:,0], Ipts[:,1], 'ro', markersize=4.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.tight_layout()
plt.show()
```

There are two sources of error here:
1. The integration error for the values of the integrals at the interpolation points. 
2. The interpolation error for the values of the integrals away from the interpolation points.

Based on the previous demonstration, the first type of error is nonexistant. The red points in the error figure above show the location of the interpolation points. You can see that the error oscillates around these points crossing zero at the points themselves. 

```{code-cell} ipython3

```
