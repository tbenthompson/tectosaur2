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

## Setting up a test problem

```{code-cell} ipython3
import matplotlib.pyplot as plt
import sympy as sp

%config InlineBackend.figure_format='retina'
```

```{code-cell} ipython3
import pickle

with open("data/constant_test_integral.pkl", "rb") as f:
    symbolic_coincident = pickle.load(f)

ox, oy = sp.symbols("ox, oy")
soln_coincident = sp.lambdify((ox, oy), symbolic_coincident, "numpy")
```

### Lagrange basis functions

```{code-cell} ipython3
:tags: []

N = 5
chebyshev_pts = [sp.cos(sp.pi * i / (N - 1)) for i in range(N)][::-1]
x = sp.var("x")
```

```{code-cell} ipython3
basis_functions = []
for i in range(N):
    xi = chebyshev_pts[i]
    prod = 1
    # The definition of the Lagrange interpolating polynomial.
    # In a numerical context, this definition is troublesome
    # and it's better to use the barycentric Lagrange formulas.
    # But this simple definition works fantastically well in
    # a symbolic setting.
    for j in range(N):
        if j == i:
            continue
        xj = chebyshev_pts[j]
        prod *= (x - xj) / (xi - xj)
    basis_functions.append(prod.simplify().expand())
```

```{code-cell} ipython3
basis_functions
```

### Polar integration

```{code-cell} ipython3
C = 1.0 / (4 * np.pi)


def fundamental_solution(obsx, obsy, srcx, srcy):
    r2 = ((obsx - srcx) ** 2) + ((obsy - srcy) ** 2)
    return C * np.log(r2)
```

```{code-cell} ipython3
sx, sy = sp.var("sx, sy")
```

```{code-cell} ipython3
import scipy.integrate
def to_corner(ox, oy, cx, cy):
    t = np.arctan2(cy - oy, cx - ox)
    r = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
    return (t, r)


def compute_pair(obsx, obsy, basis):
    tol = 1e-16
    
    def F(srcR, srcT):
        if srcR == 0:
            return 0
        srcx = obsx + np.cos(srcT) * srcR
        srcy = obsy + np.sin(srcT) * srcR
        out = srcR * basis(srcx, srcy) * fundamental_solution(obsx, obsy, srcx, srcy)
        return out
    
    corner_vecs = [
        to_corner(obsx, obsy, 1, 1),
        to_corner(obsx, obsy, -1, 1),
        to_corner(obsx, obsy, -1, -1),
        to_corner(obsx, obsy, 1, -1)
    ]]
    # Normally the theta value for corner idx 2 is negative because it 
    # is greater than Pi and the output range of arctan2 is [-pi,pi]
    # But, if the observation point is on the bottom edge of the domain (y=-1)
    # then it's possible for the the theta value to be exactly pi. If this is the
    # case it will be positive and will mess up the integration domains for
    # integrals 2 and 3. So, if it's positive here, we loop around and make 
    # it negative.
    if corner_vecs[2][0] > 0:
        corner_vecs[2][0] -= 2 * np.pi

    subdomain = [
        [corner_vecs[0][0], corner_vecs[1][0], lambda t: (1.0 - obsy) / np.sin(t)],
        [corner_vecs[1][0], corner_vecs[2][0] + 2 * np.pi, lambda t: (-1.0 - obsx) / np.cos(t)],
        [corner_vecs[2][0], corner_vecs[3][0], lambda t: (-1.0 - obsy) / np.sin(t)],
        [corner_vecs[3][0], corner_vecs[0][0], lambda t: (1.0 - obsx) / np.cos(t)],
    
    
    Is = []
    for d in subdomain:
        I = scipy.integrate.dblquad(F, d[0], d[1], 0.0, d[2], epsabs=tol, epsrel=tol)
        Is.append(I)
    
    result = sum([I[0] for I in Is])
    err = sum([I[1] for I in Is])
    return result, err
```

```{code-cell} ipython3
%%time
est = compute_pair(-0.5, -0.5, lambda sx, sy: 1.0)
true = soln_coincident(-0.5, -0.5)
est, true, est[0] - true[0]
```

```{code-cell} ipython3
est = compute_pair(1, 1, lambda sx, sy: 1.0)
true = soln_coincident(1 - 1e-7, 1 - 1e-7)
est, true, est[0] - true[0]
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
%%time
import multiprocessing

def compute_grid(obs_scale, obs_offsetx, obs_offsety)
    inputs = []
    for obsi in range(N):
        for obsj in range(N):
            obsx = obs_scale * float(chebyshev_pts[obsi]) + obs_offsetx
            obsy = obs_scale * float(chebyshev_pts[obsj]) + obs_offsety
            for srci in range(N):
                for srcj in range(N):
                    inputs.append((obsx, obsy, srci, srcj))

    def mp_compute_pair(obsx, obsy, srci, srcj):
        basis_sxsy = basis_functions[srci].subs(x, sx) * basis_functions[srcj].subs(x, sy)
        basis = sp.lambdify((sx, sy), basis_sxsy, "numpy")
        return compute_pair(obsx, obsy, basis)

    p = multiprocessing.Pool()
    return p.starmap(mp_compute_pair, inputs)
```

```{code-cell} ipython3
integrals_and_err = np.array(integrals_and_err)
```

```{code-cell} ipython3
integrals = integrals_and_err[:, 0].reshape((N, N, N, N))
error = integrals_and_err[:, 1].reshape((N, N, N, N))
```

There are no estimated errors greated than `5e-15`:

```{code-cell} ipython3
np.array(inputs, dtype=object).reshape((5, 5, 5, 5, 4))[np.where(error > 5e-15)]
```

```{code-cell} ipython3
for i in range(1, N - 1):
    for j in range(1, N - 1):
        err = (
            soln_coincident(float(chebyshev_pts[i]), float(chebyshev_pts[j]))
            - integrals[i, j, :, :].sum()
        )
        print(err[0])
```

## Pre-computing near-field coefficients

```{code-cell} ipython3
np.save(
    "data/coincident_grid.npy",
    compute_grid(1, 1, 0, 0),
)
np.save(
    "data/adj1_grid.npy",
    compute_grid(1, 1, 2, 2),
)
np.save("data/adj2_grid.npy", compute_grid(1, 1, 0, 2))
np.save(
    "data/adj3_grid.npy",
    compute_grid(2, 2, 3, 3),
)
np.save(
    "data/adj4_grid.npy",
    compute_grid(2, 2, 1, 3),
)
```

```{code-cell} ipython3
grid_filenames = [
    "data/coincident_grid.npy",
    "data/adj1_grid.npy",
    "data/adj2_grid.npy",
    "data/adj3_grid.npy",
    "data/adj4_grid.npy",
]
raw_grids = np.array([np.load(g, allow_pickle=True) for g in grid_filenames])
```

### Test coincident

```{code-cell} ipython3
cest = (
    raw_grids[0, :, 0, 3]
    - raw_grids[0, :, 1, 3]
    - raw_grids[0, :, 5, 3]
    + raw_grids[0, :, 6, 3]
)
for i in range(1, n_chebyshev_terms - 1):
    for j in range(1, n_chebyshev_terms - 1):
        print(
            est[i * n_chebyshev_terms + j]
            - xy_soln_coincident(chebyshev_pts[i], chebyshev_pts[j])
        )
```

### Build nearfield

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
raw_grid_subset = raw_grids[:, :, np.array([[0, 1], [5, 6]]), 2]
coeffs = np.array([[1.0, -1.0], [-1.0, 1.0]])
```

```{code-cell} ipython3
obs_center = np.array([-2, 0.0])
src_center = np.array([0, 0.0])
```

```{code-cell} ipython3
nobs = 100
zoomx = np.array([obs_center[0] - 0.999, obs_center[0] + 0.999])
zoomy = np.array([obs_center[1] - 0.999, obs_center[1] + 0.999])
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obsx_flat = obsx.flatten()
obsy_flat = obsy.flatten()
```

```{code-cell} ipython3
# def nearfield_interact(src_center, obs_center, size_ratio):
# Currently, assume sizes are the same (size_ratio == 1)
# TODO: identify whether it's a type 1 or type 2 interaction
# TODO: reflect/rotate to get in position
A = obs_center - src_center
A /= np.linalg.norm(A)
if np.prod(A) == 0:
    # type 2
    pair_type = 2
    B = np.array([1.0, 0.0])
    B_center = np.array([2.0, 0.0])
else:
    # type 1
    pair_type = 1
    B = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    B_center = np.array([2.0, 2.0])
R0 = A[0] * B[0] + A[1] * B[1]
R1 = B[0] * A[1] - A[0] * B[1]
rot_mat = np.array([[R0, R1], [-R1, R0]])
```

```{code-cell} ipython3
A, B, rot_mat, rot_mat.dot(A) - B
```

```{code-cell} ipython3
rot_obsx_flat, rot_obsy_flat = rot_mat.dot(np.array([obsx_flat, obsy_flat]))
```

```{code-cell} ipython3
Ix, Iwts = cheblob(n_chebyshev_terms)
Ipts, Iwts2d = tensor_product(chebyshev_pts, Iwts)

rot_coeffs = rot_mat.dot(coeffs.T).T
rot_coeffs = np.array([[1, 1], [1, 1]])
F = np.sum(
    (rot_coeffs[None, :, :] * raw_grid_subset[pair_type]).reshape((-1, 4)), axis=1
)

F_interp = barycentric_tensor_product(
    rot_obsx_flat - B_center[0], rot_obsy_flat - B_center[1], Ix, Iwts, np.array([F])
)
F_interp2d = F_interp.reshape(obsx.shape)
```

```{code-cell} ipython3
xy_soln_coincident_np = sp.lambdify((ox, oy), coincident, "numpy")
xy_soln_nearfield_np = sp.lambdify((ox, oy), nearfield, "numpy")
F_correct = xy_soln_coincident_np(obsx_flat, obsy_flat).reshape(obsx.shape)
outside = (np.abs(obsx) > 1) | (np.abs(obsy) > 1)
F_correct[outside] = xy_soln_nearfield_np(obsx_flat, obsy_flat).reshape(obsx.shape)[
    outside
]
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
nI = n_chebyshev_terms
plt.figure(figsize=(8.5, 8.5))
plt.subplot(2, 2, 1)
levels = np.linspace(np.min(F_correct), np.max(F_correct), 7)
cntf = plt.contourf(
    (Ipts[:, 0] + center[0]).reshape((nI, nI)),
    (Ipts[:, 1] + center[1]).reshape((nI, nI)),
    F.reshape((nI, nI)),
    levels=levels,
    extend="both",
)
plt.contour(
    (Ipts[:, 0] + center[0]).reshape((nI, nI)),
    (Ipts[:, 1] + center[1]).reshape((nI, nI)),
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

plt.subplot(2, 2, 2)
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


plt.subplot(2, 2, 3)
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
plt.plot(Ipts[:, 0] + obs_center[0], Ipts[:, 1] + obs_center[1], "ro", markersize=4.5)
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
