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

### Lagrange basis functions

```{code-cell} ipython3
:tags: []

N = 5
chebyshev_pts = [sp.cos(sp.pi * i / (N - 1)) for i in range(N)][::-1]
chebyshev_pts_np = np.array([float(p) for p in chebyshev_pts])
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

### Precomputing coincident integrals with polar integration

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
    return [t, r]


def compute_coincident(obsx, obsy, basis):
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
        to_corner(obsx, obsy, 1, -1),
    ]
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
        [
            corner_vecs[1][0],
            corner_vecs[2][0] + 2 * np.pi,
            lambda t: (-1.0 - obsx) / np.cos(t),
        ],
        [corner_vecs[2][0], corner_vecs[3][0], lambda t: (-1.0 - obsy) / np.sin(t)],
        [corner_vecs[3][0], corner_vecs[0][0], lambda t: (1.0 - obsx) / np.cos(t)],
    ]

    Is = []
    for d in subdomain:
        I = scipy.integrate.dblquad(F, d[0], d[1], 0.0, d[2], epsabs=tol, epsrel=tol)
        Is.append(I)

    result = sum([I[0] for I in Is])
    err = sum([I[1] for I in Is])
    return result, err
```

```{code-cell} ipython3
import pickle

with open("data/constant_test_integral.pkl", "rb") as f:
    coincident, nearfield = pickle.load(f)

ox, oy = sp.symbols("ox, oy")
constant_soln_coincident = sp.lambdify((ox, oy), coincident, "numpy")
constant_soln_nearfield = sp.lambdify((ox, oy), nearfield, "numpy")
```

```{code-cell} ipython3
%%time
est = compute_coincident(-0.5, -0.5, lambda sx, sy: 1.0)
true = constant_soln_coincident(-0.5, -0.5)
est[0], true, est[0] - true
```

```{code-cell} ipython3
est = compute_coincident(1, 1, lambda sx, sy: 1.0)
true = constant_soln_coincident(1 - 1e-7, 1 - 1e-7)
est[0], true, est[0] - true
```

```{code-cell} ipython3
est = compute_coincident(0, 1, lambda sx, sy: 1.0)
true = constant_soln_coincident(0, -1 + 1e-7)
est[0], true, est[0] - true
```

```{code-cell} ipython3
:tags: []

import multiprocessing


def mp_compute_coincident(obsx, obsy, srci, srcj):
    basis_sxsy = basis_functions[srci].subs(x, sx) * basis_functions[srcj].subs(x, sy)
    basis = sp.lambdify((sx, sy), basis_sxsy, "numpy")
    return compute_coincident(obsx, obsy, basis)

def get_inputs(obs_scale, obs_offsetx, obs_offsety):
    inputs = []
    for obsi in range(N):
        for obsj in range(N):
            obsx = obs_scale * chebyshev_pts_np[obsi] + obs_offsetx
            obsy = obs_scale * chebyshev_pts_np[obsj] + obs_offsety
            for srci in range(N):
                for srcj in range(N):
                    inputs.append((obsx, obsy, srci, srcj))
    return inputs


def coincident_grid():
    inputs = get_inputs(1, 0, 0)
    p = multiprocessing.Pool()
    return inputs, np.array(p.starmap(mp_compute_coincident, inputs))
```

```{code-cell} ipython3
np.save("data/coincident_grid.npy", coincident_grid())
```

```{code-cell} ipython3
#integrals_and_err = compute_grid(1, 0, 0)
integrals_and_err = np.load('data/coincident_grid.npy')
integrals = integrals_and_err[:, 0].reshape((N, N, N, N))
error = integrals_and_err[:, 1].reshape((N, N, N, N))
```

There are no estimated errors greated than `5e-15`:

```{code-cell} ipython3
inputs_arr = np.array(inputs, dtype=object).reshape((5, 5, 5, 5, 4))
inputs_arr[np.where(error > 5e-15)]
```

```{code-cell} ipython3
for i in range(1, N - 1):
    for j in range(1, N - 1):
        err = (
            constant_soln_coincident(chebyshev_pts_np[i], chebyshev_pts_np[j])
            - integrals[i, j, :, :].sum()
        )
        print(err)
```

```{code-cell} ipython3
with open("data/xy_test_integral.pkl", "rb") as f:
    coincident, nearfield = pickle.load(f)
xy_soln_coincident = sp.lambdify((ox, oy), coincident, "numpy")
xy_soln_nearfield = sp.lambdify((ox, oy), nearfield, "numpy")
```

```{code-cell} ipython3
f = (1 - chebyshev_pts_np[:, None]) * (1 - chebyshev_pts_np[None, :] ** 2)

for i in range(1, N - 1):
    for j in range(1, N - 1):
        true = xy_soln_coincident(chebyshev_pts_np[i], chebyshev_pts_np[j])
        est = integrals[i, j, :, :].ravel().dot(f.ravel())
        err = true - est
        print(err)
```

## Pre-computing adjacent integrals

```{code-cell} ipython3
def is_on_source_edge(obsx, obsy):
    on_left_right_edges = np.abs(obsx) == 1 and np.abs(obsy) <= 1
    on_top_bottom_edges = np.abs(obsy) == 1 and np.abs(obsx) <= 1
    return (on_left_right_edges or on_top_bottom_edges)

def compute_nearfield(obsx, obsy, basis):
    if is_on_source_edge(obsx, obsy):
        return compute_coincident(obsx, obsy, basis)
    
    tol = 1e-16

    def F(srcy, srcx):
        return basis(srcx, srcy) * fundamental_solution(obsx, obsy, srcx, srcy)

    I = scipy.integrate.dblquad(F, -1, 1, -1, 1, epsabs=tol, epsrel=tol)
    return I
```

```{code-cell} ipython3
est = compute_nearfield(1.1, 1.1, lambda x, y: 1.0)
true = constant_soln_nearfield(1.1, 1.1)
est[0], true, est[0] - true
```

```{code-cell} ipython3
est = compute_nearfield(-1.1, -1.1, lambda x, y: (1 - x) * (1 - y ** 2))
true = xy_soln_nearfield(-1.1, -1.1)
est[0], true, est[0] - true
```

```{code-cell} ipython3
est = compute_nearfield(1.0, 1.0, lambda x, y: 1.0)
true = constant_soln_nearfield(1.0 + 1e-7, 1.0 + 1e-7)
est[0], true, est[0] - true
```

```{code-cell} ipython3
def mp_compute_nearfield(obsx, obsy, srci, srcj):
    basis_sxsy = basis_functions[srci].subs(x, sx) * basis_functions[srcj].subs(x, sy)
    basis = sp.lambdify((sx, sy), basis_sxsy, "numpy")
    return compute_nearfield(obsx, obsy, basis)


def compute_grid(obs_scale, obs_offsetx, obs_offsety):
    inputs = get_inputs(obs_scale, obs_offsetx, obs_offsety)
    p = multiprocessing.Pool()
    return np.array(p.starmap(mp_compute_nearfield, inputs))
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: []
---
import matplotlib.patches as patches
xrange = [-1.5,6]
yrange = [-1.5,6]

def size_and_aspect():
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.axis('off')
    #plt.axis('equal')

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((1, 1), 2, 2, linewidth=1, edgecolor='k', facecolor='none'))
plt.text(1.85, 1.7, "1", fontsize=30)
size_and_aspect()

plt.subplot(2,2,2)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((1, -1), 2, 2, linewidth=1, edgecolor='k', facecolor='none'))
plt.text(1.85, -0.3, "2", fontsize=30)
size_and_aspect()

plt.subplot(2,2,3)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((1, 1), 4, 4, linewidth=1, edgecolor='k', facecolor='none'))
plt.text(2.65, 2.65, "3", fontsize=30)
size_and_aspect()

plt.subplot(2,2,4)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((-1, 1), 4, 4, linewidth=1, edgecolor='k', facecolor='none'))
plt.text(0.65, 2.65, "4", fontsize=30)
size_and_aspect()

plt.show()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
%%time
np.save("data/adj1_grid.npy", compute_grid(1, 2, 2))
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
%%time
np.save("data/adj2_grid.npy", compute_grid(1, 0, 2))
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
%%time
np.save("data/adj3_grid.npy", compute_grid(2, 3, 3))
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
%%time
np.save("data/adj4_grid.npy", compute_grid(2, 1, 3))
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

The estimated error is extremely small for all the integrals!

```{code-cell} ipython3
np.where(raw_grids[:,:,1] > 5e-15)
```

## Rotations

+++

### Type 1

```{code-cell} ipython3
#(1 - chebyshev_pts_np[:, None]) * (1 - chebyshev_pts_np[None, :] ** 2)
correct = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        obsx = 2.0 + chebyshev_pts_np[i]
        obsy = 2.0 + chebyshev_pts_np[j]
        if np.abs(obsx) == 1 or np.abs(obsy) == 1:
            correct[i,j] = np.nan
        else:
            correct[i,j] = constant_soln_nearfield(obsx, obsy)
            
f = np.ones((N ** 2))
integrals = raw_grids[1, :, 0].reshape((N,N,N**2))
est = integrals.dot(f)
```

```{code-cell} ipython3
true - est
```

### Archive

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
