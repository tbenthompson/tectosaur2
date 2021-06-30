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
* build the symbolic solution for all x, y!

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

with open("data/test_integral.pkl", "rb") as f:
    coincident, nearfield = pickle.load(f)

ox, oy = sp.symbols("ox, oy")
xy_soln_coincident = sp.lambdify((ox, oy), coincident, "mpmath")
xy_soln_nearfield = sp.lambdify((ox, oy), nearfield, "mpmath")
```

```{code-cell} ipython3
def xy_laplacian_fnc(x, y):
    return (1 - x) * (1 - y)
```

```{code-cell} ipython3
import sympy as sp
```

### Lagrange basis functions

```{code-cell} ipython3
:tags: []

N = 5
chebyshev_pts = [sp.cos(sp.pi * i / (N - 1)) for i in range(N)][::-1]
x = sp.var('x')
```

```{code-cell} ipython3
basis_functions = []
for i in range(n_chebyshev_terms):
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

## Pre-computing near-field coefficients

```{code-cell} ipython3
mp_inv_pi = mp.mpf('1.0') / mp.pi
def volume_integral(obsx, obsy, I, J, srcx, srcy):
    interp_val = mp.mpf('1')
    dx = obsx - srcx
    dy = obsy - srcy
    r2 = (dx ** 2) + (dy ** 2)
    G = mp_C * mp.log(r2)
    out = G * interp_val
    return out
```

```{code-cell} ipython3
import scipy.integrate
A = scipy.integrate.dblquad(lambda x, y: volume_integral(-0.99, -0.99, 0, 0, x, y), -1, -0.99, -1, 1, epsabs=1e-14, epsrel=1e-14)
B = scipy.integrate.dblquad(lambda x, y: volume_integral(-0.99, -0.99, 0, 0, x, y), -0.99, 1, -1, 1, epsabs=1e-14, epsrel=1e-14)
```

```{code-cell} ipython3
A[0] + B[0]
```

```{code-cell} ipython3
:tags: []

n_chebyshev_terms = 5
chebyshev_pts = np.array([mp.cos(mp.pi * mp.mpf(i) / (n_chebyshev_terms - 1)) for i in range(n_chebyshev_terms)])
```

```{code-cell} ipython3
:tags: []

def compute_for_pt(obsx, obsy, srcI, srcJ, verbose=False):
    mp.dps = n_digits
    print('Computing integrals for point:', obsx, obsy)
    grid = []
    for n in range(n_chebyshev_terms):
        for m in range(n_chebyshev_terms):
            integral, error_estimate = mp.quadts(
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
from mpmath.calculus.quadrature import TanhSinh
```

```{code-cell} ipython3
obsx = mp.mpf('-0.99')
obsy = obsx
```

```{code-cell} ipython3
mp.dps = 30
```

```{code-cell} ipython3
TS = TanhSinh(mp)
```

```{code-cell} ipython3
degree = 3#TS.guess_degree(mp.prec)
nodes1 = np.array(TS.get_nodes(-1, obsx, degree, mp.prec - 15))
nodes2 = np.array(TS.get_nodes(obsx, 1, degree, mp.prec - 15))
nodesy = np.array(TS.get_nodes(-1, 1, degree, mp.prec - 15))
```

```{code-cell} ipython3

def clencurt(n1):
    """Computes the Clenshaw Curtis quadrature nodes and weights"""
    C = quadpy.c1.clenshaw_curtis(n1)
    return (C.points[::-1], C.weights[::-1])

# nodes1 = np.array(clencurt(20)).T
# nodes2 = np.array(clencurt(20)).T
#nodesy = np.array(clencurt(20)).T
```

```{code-cell} ipython3
%%time
S = 0
for j in range(nodesy.shape[0]):
    for i in range(nodes1.shape[0]):
        v1 = 1.0#volume_integral(obsx, obsy, 0, 0, nodes1[i,0], nodesy[j,0])
        v2 = 1.0#volume_integral(obsx, obsy, 0, 0, nodes2[i,0], nodesy[j,0])
        V1 = v1 * nodes1[i,1] * nodesy[j,1]
        V2 = v2 * nodes2[i,1] * nodesy[j,1]
        S += V1 + V2
```

```{code-cell} ipython3
S
```

```{code-cell} ipython3
quad_nodes = np.array(.calc_nodes(6, mp.prec - 9))
```

```{code-cell} ipython3
pts01 = (quad_nodes[:,0] + 1) / 2
wts01 = quad_nodes[:,1] / 2
```

```{code-cell} ipython3
pts01.shape, wts01[-4:], pts01[-4:]
```

```{code-cell} ipython3
def to_interval(a, b):
    return pts01 * (a - b) + a
```

```{code-cell} ipython3
def compute_for_pt2(obsx, obsy):
    
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
grid = compute_for_pt(2, mp.mpf('-0.99'), mp.mpf('-0.99'), verbose=True)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
p = multiprocessing.Pool()
def compute_grid(multx, multy, offsetx, offsety):
    X, Y = np.meshgrid(multx * chebyshev_pts + offsetx, multy * chebyshev_pts + offsety)
    grid_input = zip(
        [n_chebyshev_terms] * n_chebyshev_terms ** 2, 
        X.ravel(), 
        Y.ravel()
    )  
    return np.array(p.starmap(compute_for_pt, grid_input))
```

```{code-cell} ipython3
np.save("data/coincident_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('0.0'), mp.mpf('0.0')))
np.save("data/adj1_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('2.0'), mp.mpf('2.0')))
np.save("data/adj2_grid.npy", compute_grid(mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('0.0'), mp.mpf('2.0')))
np.save("data/adj3_grid.npy", compute_grid(mp.mpf('2.0'), mp.mpf('2.0'), mp.mpf('3.0'), mp.mpf('3.0')))
np.save("data/adj4_grid.npy", compute_grid(mp.mpf('2.0'), mp.mpf('2.0'), mp.mpf('1.0'), mp.mpf('3.0')))
```

```{code-cell} ipython3
grid_filenames = [
    "data/coincident_grid.npy",
    "data/adj1_grid.npy",
    "data/adj2_grid.npy",
    "data/adj3_grid.npy",
    "data/adj4_grid.npy"
]
raw_grids = np.array([np.load(g, allow_pickle=True) for g in grid_filenames])
```

### Test coincident

```{code-cell} ipython3
est = raw_grids[0,:,0,3] - raw_grids[0,:,1,3] - raw_grids[0,:,5,3] + raw_grids[0,:,6,3]
for i in range(1,n_chebyshev_terms-1):
    for j in range(1,n_chebyshev_terms-1):
        print(est[i * n_chebyshev_terms + j] - xy_soln_coincident(chebyshev_pts[i], chebyshev_pts[j]))
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
raw_grid_subset = raw_grids[:,:,np.array([[0,1],[5,6]]), 2]
coeffs = np.array([[1.0, -1.0], [-1.0, 1.0]])
```

```{code-cell} ipython3
obs_center = np.array([-2, 0.])
src_center = np.array([0,0.])
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
    B_center = np.array([2., 0.0])
else:
    # type 1
    pair_type = 1
    B = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)])
    B_center = np.array([2., 2.])
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
F = np.sum((rot_coeffs[None,:,:] * raw_grid_subset[pair_type]).reshape((-1, 4)), axis=1)

F_interp = barycentric_tensor_product(rot_obsx_flat - B_center[0], rot_obsy_flat - B_center[1], Ix, Iwts, np.array([F]))
F_interp2d = F_interp.reshape(obsx.shape)
```

```{code-cell} ipython3
xy_soln_coincident_np = sp.lambdify((ox, oy), coincident, "numpy")
xy_soln_nearfield_np = sp.lambdify((ox, oy), nearfield, "numpy")
F_correct = xy_soln_coincident_np(obsx_flat, obsy_flat).reshape(obsx.shape)
outside = (np.abs(obsx) > 1) | (np.abs(obsy) > 1)
F_correct[outside] = xy_soln_nearfield_np(obsx_flat, obsy_flat).reshape(obsx.shape)[outside]
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
    (Ipts[:, 0]+center[0]).reshape((nI, nI)),
    (Ipts[:, 1]+center[1]).reshape((nI, nI)),
    F.reshape((nI, nI)),
    levels=levels,
    extend="both",
)
plt.contour(
    (Ipts[:, 0]+center[0]).reshape((nI, nI)),
    (Ipts[:, 1]+center[1]).reshape((nI, nI)),
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
plt.plot(Ipts[:,0] + obs_center[0], Ipts[:,1] + obs_center[1], 'ro', markersize=4.5)
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
