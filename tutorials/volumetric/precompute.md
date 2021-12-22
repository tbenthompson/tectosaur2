---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# [DRAFT] Precomputing near-field volumetric integrals.

+++

## Setting up a test problem

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

%config InlineBackend.figure_format='retina'
```

### Lagrange basis functions

```{code-cell} ipython3
:tags: []

N = 7
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


multiprocessing.set_start_method('fork')
def coincident_grid():
    inputs = get_inputs(1, 0, 0)
    p = multiprocessing.Pool(5)
    return np.array(p.starmap(mp_compute_coincident, inputs))
```

```{code-cell} ipython3
%%time
np.save("data/coincident_grid.npy", coincident_grid())
```

```{code-cell} ipython3
# integrals_and_err = compute_grid(1, 0, 0)
integrals_and_err = np.load("data/coincident_grid.npy", allow_pickle=True)
integrals = integrals_and_err[:, 0].reshape((N, N, N, N))
error = integrals_and_err[:, 1].reshape((N, N, N, N))
```

There are no estimated errors greated than `5e-15`:

```{code-cell} ipython3
np.where(error > 5e-15)[0].shape[0]
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
xy_laplacian = lambda x, y: (1 - (1 - x) ** 3) * (1 - (y + 1) ** 2)
```

```{code-cell} ipython3
cheb2dX, cheb2dY = np.meshgrid(chebyshev_pts_np, chebyshev_pts_np)
cheb2d = np.array([cheb2dX, cheb2dY]).T.reshape((-1, 2)).copy()
```

```{code-cell} ipython3
f = xy_laplacian(cheb2d[:, 0], cheb2d[:, 1])

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
    return on_left_right_edges or on_top_bottom_edges


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
est = compute_nearfield(-1.1, -1.1, xy_laplacian)
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
:tags: []

import matplotlib.patches as patches

xrange = [-1.5, 6]
yrange = [-1.5, 6]


def size_and_aspect():
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    # plt.axis("off")
    ax = plt.gca()
    # ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    # plt.axis('equal')


plt.figure(figsize=(4, 8))

plt.subplot(3, 2, 1)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, 1), 2, 2, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(1.75, 1.65, "1", fontsize=30)
size_and_aspect()

plt.subplot(3, 2, 2)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, -1), 2, 2, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(1.75, -0.35, "2", fontsize=30)
size_and_aspect()

plt.subplot(3, 2, 3)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, 1), 4, 4, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(2.65, 2.65, "3", fontsize=30)
size_and_aspect()

plt.subplot(3, 2, 4)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, -1), 4, 4, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(2.65, 0.65, "4", fontsize=30)
size_and_aspect()


plt.subplot(3, 2, 5)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(1.25, 1.13, "5", fontsize=30)
size_and_aspect()

plt.subplot(3, 2, 6)
plt.gca().add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor="k"))
plt.gca().add_patch(
    patches.Rectangle((1, 0), 1, 1, linewidth=1, edgecolor="k", facecolor="none")
)
plt.text(1.25, 0.13, "6", fontsize=30)
size_and_aspect()

plt.show()
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj1_grid.npy", compute_grid(1, 2, 2))
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj2_grid.npy", compute_grid(1, 2, 0))
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj3_grid.npy", compute_grid(2, 3, 3))
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj4_grid.npy", compute_grid(2, 3, 1))
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj5_grid.npy", compute_grid(0.5, 1.5, 1.5))
```

```{code-cell} ipython3
:tags: []

%%time
np.save("data/adj6_grid.npy", compute_grid(0.5, 1.5, 0.5))
```

```{code-cell} ipython3
grid_filenames = [
    "data/coincident_grid.npy",
    "data/adj1_grid.npy",
    "data/adj2_grid.npy",
    "data/adj3_grid.npy",
    "data/adj4_grid.npy",
    "data/adj5_grid.npy",
    "data/adj6_grid.npy",
]
raw_grids = np.array([np.load(g, allow_pickle=True) for g in grid_filenames])
```

```{code-cell} ipython3
[raw_grids[i].shape for i in range(7)]
```

The estimated error is extremely small for all the integrals!

```{code-cell} ipython3
np.where(raw_grids[:, :, 1] > 5e-15)
```

```{code-cell} ipython3
all_integrals = raw_grids[:, :, 0].reshape((7, N, N, N ** 2))
```

## Rotations

```{code-cell} ipython3
def get_test_values(
    soln_fnc, obs_scale, obs_offsetx, obs_offsety, src_center=[0, 0], src_size=[2, 2]
):
    correct = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            obsx = obs_offsetx + obs_scale * chebyshev_pts_np[i]
            obsy = obs_offsety + obs_scale * chebyshev_pts_np[j]
            is_x_edge = np.abs(np.abs(obsx - src_center[0]) - (src_size[0] / 2)) < 1e-8
            is_y_edge = np.abs(np.abs(obsy - src_center[1]) - (src_size[1] / 2)) < 1e-8
            if is_x_edge or is_y_edge:
                correct[i, j] = np.nan
            else:
                correct[i, j] = soln_fnc(obsx, obsy)
    return correct
```

### Type 1

```{code-cell} ipython3
correct_upper_right = get_test_values(xy_soln_nearfield, 1.0, 2.0, 2.0)
correct_upper_left = get_test_values(xy_soln_nearfield, 1.0, -2.0, 2.0)
correct_lower_left = get_test_values(xy_soln_nearfield, 1.0, -2.0, -2.0)
correct_lower_right = get_test_values(xy_soln_nearfield, 1.0, 2.0, -2.0)
```

```{code-cell} ipython3
correct_upper_right
```

```{code-cell} ipython3
correct_upper_left
```

```{code-cell} ipython3
correct_lower_left
```

```{code-cell} ipython3
correct_lower_right
```

```{code-cell} ipython3
def nearfield_box(I, Fv, flipx, flipy, rotxy):
    Fv = Fv.reshape((N,N))

    n_rot = {
        (1, 1): 0,
        (1, -1): 1,
        (-1, -1): 2,
        (-1, 1): 3
    }[(flipx, flipy)]
    n_transpose = ((n_rot % 2) == 1) + rotxy

    # Rotate from input coordinates into position
    Fv = np.rot90(Fv, n_rot)
    if n_transpose % 2 == 1:
        Fv = Fv.T

    est = I.dot(Fv.ravel())
    # Reverse the transformation back to the original input space
    if n_transpose % 2 == 1:
        est = est.T
    est = np.rot90(est, -n_rot)
    return est


for C, flipx, flipy in [
    (correct_upper_right, 1, 1),
    (correct_upper_left, -1, 1),
    (correct_lower_left, -1, -1),
    (correct_lower_right, 1, -1),
]:
    Fv = xy_laplacian(cheb2d[:,0], cheb2d[:,1]).reshape((N,N))
    est = nearfield_box(all_integrals[1], Fv, flipx, flipy, 0)
    print(np.max(np.abs((C - est)[~np.isnan(C)])))
```

### Type 2

```{code-cell} ipython3
correct_middle_right = get_test_values(xy_soln_nearfield, 1.0, 2.0, 0.0)
correct_top_center = get_test_values(xy_soln_nearfield, 1.0, 0.0, 2.0)
correct_middle_left = get_test_values(xy_soln_nearfield, 1.0, -2.0, 0.0)
correct_bottom_center = get_test_values(xy_soln_nearfield, 1.0, 0.0, -2.0)
```

```{code-cell} ipython3
for C, flipx, flipy, rotxy in [
    (correct_middle_right, 1, 1, 0),
    (correct_top_center, 1, 1, 1),
    (correct_middle_left, -1, 1, 0),
    (correct_bottom_center, 1, -1, 1),
]:
    Fv = xy_laplacian(cheb2d[:,0], cheb2d[:,1])
    est = nearfield_box(all_integrals[2], Fv, flipx, flipy, rotxy)
    print(np.max(np.abs((C - est)[~np.isnan(C)])))
```

```{code-cell} ipython3
boxes = {
    # Type 0 (coincident)
    (1, 0, 0): (0, 1, 1, 0),
    # Type 1
    (1, 2, 2): (1, 1, 1, 0),
    (1, -2, 2): (1, -1, 1, 0),
    (1, -2, -2): (1, -1, -1, 0),
    (1, 2, -2): (1, 1, -1, 0),
    # Type 2
    (1, 2, 0): (2, 1, 1, 0),
    (1, 0, 2): (2, 1, 1, 1),
    (1, -2, 0): (2, -1, 1, 0),
    (1, 0, -2): (2, 1, -1, 1),
    # Type 3
    (2, 3, 3): (3, 1, 1, 0),
    (2, -3, 3): (3, -1, 1, 0),
    (2, -3, -3): (3, -1, -1, 0),
    (2, 3, -3): (3, 1, -1, 0),
    # Type 4
    (2, 1, 3): (4, 1, 1, 1),
    (2, -1, 3): (4, -1, 1, 1),
    (2, -3, 1): (4, -1, 1, 0),
    (2, -3, -1): (4, -1, -1, 0),
    (2, -1, -3): (4, -1, -1, 1),
    (2, 1, -3): (4, 1, -1, 1),
    (2, 3, -1): (4, 1, -1, 0),
    (2, 3, 1): (4, 1, 1, 0),
    # Type 5
    (0.5, 1.5, 1.5): (5, 1, 1, 0),
    (0.5, -1.5, 1.5): (5, -1, 1, 0),
    (0.5, -1.5, -1.5): (5, -1, -1, 0),
    (0.5, 1.5, -1.5): (5, 1, -1, 0),
    # Type 6
    (0.5, 0.5, 1.5): (6, 1, 1, 1),
    (0.5, -0.5, 1.5): (6, -1, 1, 1),
    (0.5, -1.5, 0.5): (6, -1, 1, 0),
    (0.5, -1.5, -0.5): (6, -1, -1, 0),
    (0.5, -0.5, -1.5): (6, -1, -1, 1),
    (0.5, 0.5, -1.5): (6, 1, -1, 1),
    (0.5, 1.5, -0.5): (6, 1, -1, 0),
    (0.5, 1.5, 0.5): (6, 1, 1, 0),
}
```

```{code-cell} ipython3
for box_loc, rot_params in boxes.items():
    soln_fnc = xy_soln_coincident if rot_params[0] == 0 else xy_soln_nearfield
    C = get_test_values(soln_fnc, *box_loc)
    Fv = xy_laplacian(cheb2d[:,0], cheb2d[:,1])
    est = nearfield_box(all_integrals[rot_params[0]], Fv, *rot_params[1:])
    print(np.max(np.abs((C - est)[~np.isnan(C)])))
```

## Scaling

+++

The remaining piece is to scale the source and observation boxes so that they fit into the rotation scheme above.

```{code-cell} ipython3
def get_test_values(
    soln_fnc, obs_scale, obs_offsetx, obs_offsety, src_center=[0, 0], src_size=2
):
    correct = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            obsx = obs_offsetx + obs_scale * chebyshev_pts_np[i]
            obsy = obs_offsety + obs_scale * chebyshev_pts_np[j]
            is_x_edge = np.abs(np.abs(obsx - src_center[0]) - (src_size / 2)) < 1e-8
            is_y_edge = np.abs(np.abs(obsy - src_center[1]) - (src_size / 2)) < 1e-8
            if is_x_edge or is_y_edge:
                correct[i, j] = np.nan
            else:
                correct[i, j] = soln_fnc(obsx, obsy)
    return correct
```

```{code-cell} ipython3
with open("data/constant_transformed_test_integral.pkl", "rb") as f:
    constant_soln_shifted = [sp.lambdify((ox, oy), I, "numpy") for I in pickle.load(f)]
with open("data/xy_transformed_test_integral.pkl", "rb") as f:
    xy_soln_shifted = [sp.lambdify((ox, oy), I, "numpy") for I in pickle.load(f)]
```

```{code-cell} ipython3
basis_integrals = np.empty((N, N))
for srci in range(N):
    for srcj in range(N):
        basis_sxsy = basis_functions[srci].subs(x, sx) * basis_functions[srcj].subs(x, sy)
        basis = sp.lambdify((sx, sy), basis_sxsy, "numpy")
        I = scipy.integrate.dblquad(basis, -1, 1, -1, 1, epsabs=1e-16, epsrel=1e-16)
        basis_integrals[srci, srcj] = I[0]
```

```{code-cell} ipython3
def scale_integral(I, basis_dot_F, src_s):
    scale_T = src_s / 2.0
    C = scale_T ** 2
    log_factor = C * (1 / (2 * np.pi)) * np.log(scale_T)
    return C * I + log_factor * basis_dot_F

F_xy = lambda x, y: (1 - x) * (1 - y ** 2)
for i, mult in enumerate([1, 2, 4, 8, 16]):
    src_c = np.array([0, 0])
    src_s = mult / 4.0
    obs_c = np.array([src_c[0] + 0.5 * src_s, src_c[1] - 1.5 * src_s])
    #obs_c = np.array([src_c[0] - 0.5 * src_s, src_c[1] - 1.5 * src_s])

    obs_s = src_s * 2
    transformed_obs_center = np.round(2 * (obs_c - src_c) / src_s, decimals=1)
    transformed_obs_size = np.round(obs_s / src_s, decimals=1)

    correct_xy = get_test_values(
        xy_soln_shifted[i],
        obs_s / 2.0,
        obs_c[0],
        obs_c[1],
        src_center=src_c,
        src_size=src_s,
    )

    src_box_pts = cheb2d * 0.5 * src_s + src_c[None,:]
    Fv_xy = F_xy(src_box_pts[:,0], src_box_pts[:,1])

    nearfield_info = boxes[(transformed_obs_size, *transformed_obs_center)]
    integral_type, flipx, flipy, rotxy = nearfield_info

    basis_I_xy = basis_integrals.ravel().dot(Fv_xy.ravel())

    I_xy = nearfield_box(all_integrals[integral_type], Fv_xy, flipx, flipy, rotxy)

    est_xy = scale_integral(I_xy, basis_I_xy, src_s)

    print(f'\nfor source size={src_s}')
    print("xy error: ", np.max(np.abs(correct_xy - est_xy)[~np.isnan(correct_xy)]))
```

### Computing for an arbitrary box pair.

+++

1. Scale the source box to have width and length 2.
2. Center the source box at `(0,0)`
3. Perform the same transformations on the observation box.
4. Because adjacent boxes are at most one level apart, they will fall into one of the 33 categories defined above.
5. Retrieve the integral type and rotation information and compute the integral!
6. Reverse the rotations.
7. Reverse the scaling via the equations above.

```{code-cell} ipython3
to_save = np.empty(2, object)
to_save[0] = basis_integrals
to_save[1] = all_integrals
np.save("data/nearfield_integrals.npy", to_save)
```
