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
:tags: [remove-cell]

from config import setup, import_and_display_fnc

setup()
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    qbx_matrix,
    build_interp_matrix,
    build_interpolator,
    qbx_setup,
    double_layer_matrix,
    hypersingular_matrix,
    PanelSurface,
    panelize_symbolic_surface,
    build_panel_interp_matrix,
    pts_grid,
)
```

This next section will construct the free surface panelized surface `free`. The `corner_resolution` specifies how large the panels will be near the fault-surface intersection. Away from that intersection, the panels will each be double the length of the prior panel, thus enabling the full surface to efficiently represent an effectively infinite free surface.

```{code-cell} ipython3
from common import Surface
import sympy as sp

corner_resolution = 5000
n_panels = 20

visco_depth = 20000
viscosity = 1e18
shear_modulus = 3e9

# It seems that we need several "small" panels right near the fault-surface intersection!
panels = [
    (-3 * corner_resolution, -2 * corner_resolution),
    (-2 * corner_resolution, -corner_resolution),
    (-corner_resolution, 0),
    (0, corner_resolution),
    (corner_resolution, 2 * corner_resolution),
    (2 * corner_resolution, 3 * corner_resolution),
]
for i in range(n_panels - len(panels)):
    panel_start = panels[-1][1]
    panel_end = panel_start + min(20000, corner_resolution * (1.3 ** i))
    panels.append((panel_start, panel_end))
    panels.insert(0, (-panel_end, -panel_start))
panels = np.array(panels)
scaled_panels = panels / panels[-1][1]

qx, qw = gauss_rule(16)
qinterp = gauss_rule(9 * 16)
t = sp.var("t")

free = panelize_symbolic_surface(t, -panels[-1][1] * t, 0 * t, scaled_panels, qx, qw)
free_interp = panelize_symbolic_surface(
    t, panels[-1][1] * t, 0 * t, scaled_panels, *qinterp
)
Im_free = build_panel_interp_matrix(free, free_interp)
```

```{code-cell} ipython3
panels[-1][1] - panels[-1][0], panels[-1][1]
```

```{code-cell} ipython3
plt.plot(np.log10(np.abs(free.pts[:, 0])))
```

```{code-cell} ipython3
fault_top = -0.0
fault_bottom = -15000.0

fault_panels = [
    (0, 0.5 * corner_resolution),
    (0.5 * corner_resolution, 1.0 * corner_resolution),
    (1.0 * corner_resolution, 1.5 * corner_resolution),
    (1.5 * corner_resolution, 2 * corner_resolution),
]
for i in range(100):
    panel_start = fault_panels[-1][1]
    panel_end = panel_start + 2 * corner_resolution * (2 ** i)
    if panel_end > fault_bottom * 0.75:
        panel_end = -fault_bottom
    fault_panels.append((panel_start, panel_end))
    panel_start = panel_end
    if panel_end == -fault_bottom:
        break
fault_panels = np.array(fault_panels)
scaled_fault_panels = 2 * ((fault_panels / fault_panels[-1][1]) - 0.5)
fault = panelize_symbolic_surface(
    t, 0 * t, -fault_panels[-1][1] * (t + 1) * 0.5, scaled_fault_panels, qx, qw
)
fault_interp = panelize_symbolic_surface(
    t, 0 * t, -fault_panels[-1][1] * (t + 1) * 0.5, scaled_fault_panels, *qinterp
)
```

Next, we need to carefully remove some of the QBX expansion centers. Because the expansion centers are offset towards the interior of the domain in the direction of the normal vector of the free surface, a few of them will be too close to the fault surface. We remove those. As a result, any evaluations in the corner will use the slightly farther away QBX expansion points. 

In the figure below, the QBX expansion centers are indicated in blue, while the expansion centers that we remove are indicated in red.

```{code-cell} ipython3
from common import QBXExpansions

r = (
    np.repeat((free.panel_bounds[:, 1] - free.panel_bounds[:, 0]), free.panel_sizes)
    * free.jacobians
    * 0.5
)
orig_expansions = qbx_setup(free, direction=1, r=r, p=10)
good = np.abs(orig_expansions.pts[:, 0]) > 0.30 * corner_resolution
expansions = QBXExpansions(
    orig_expansions.pts[good, :], orig_expansions.r[good], orig_expansions.p
)

plt.plot(expansions.pts[:, 0], expansions.pts[:, 1], "b.")
plt.plot(orig_expansions.pts[~good, 0], orig_expansions.pts[~good, 1], "r.")
plt.plot(free.pts[:, 0], free.pts[:, 1], "k-")
plt.plot(fault.pts[:, 0], fault.pts[:, 1], "k-")
plt.axis("equal")
# plt.xlim([-corner_resolution, corner_resolution])
# plt.ylim([-3 * corner_resolution, corner_resolution])
plt.xlim([100000, -100000])
plt.ylim([-30000, 0])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
```

Note that despite extending out to 1000 fault lengths away from the fault trace, we are only using 672 points to describe the free surface solution.

```{code-cell} ipython3
print("number of points in the free surface discretization:", free.n_pts)
print("       number of points in the fault discretization:", fault.n_pts)
```

```{code-cell} ipython3
A_raw = qbx_matrix(double_layer_matrix, free_interp, free.pts, expansions)[:, 0, :]
free_disp_to_free_disp = A_raw.dot(Im_free)
```

```{code-cell} ipython3
fault_slip_to_free_disp = -qbx_matrix(double_layer_matrix, fault, free.pts, expansions)[
    :, 0, :
]
```

```{code-cell} ipython3
free_disp_solve_mat = np.eye(free_disp_to_free_disp.shape[0]) + free_disp_to_free_disp
free_disp_solve_mat_inv = np.linalg.inv(free_disp_solve_mat)

slip = np.ones(fault.n_pts)
free_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip))

# Note that the analytical solution is slightly different than in the buried
# fault setting because we need to take the limit of an arctan as the
# denominator of the argument  goes to zero.
s = 1.0
analytical = (
    -s
    / (2 * np.pi)
    * (
        np.arctan(free.pts[:, 0] / (free.pts[:, 1] - fault_bottom))
        - np.arctan(free.pts[:, 0] / (free.pts[:, 1] + fault_bottom))
        - np.pi * np.sign(free.pts[:, 0])
    )
)
```

In the first row of graphs below, I show the solution extending to 10 fault lengths. In the second row, the solution extends to 1000 fault lengths. You can see that the solution matches to about 6 digits in the nearfield and 7-9 digits in the very farfield!

```{code-cell} ipython3
for XV in [100000.0, 10000000.0]:
    # XV = 5 * corner_resolution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(free.pts[:, 0], free_disp, "ko")
    plt.plot(free.pts[:, 0], analytical, "bo")
    plt.xlabel("$x$")
    plt.ylabel("$u_z$")
    plt.title("Displacement")
    plt.xlim([-XV, XV])
    plt.ylim([-0.6, 0.6])

    plt.subplot(1, 2, 2)
    plt.plot(free.pts[:, 0], np.log10(np.abs(free_disp - analytical)))
    plt.xlabel("$x$")
    plt.ylabel(r"$\log_{10}|u_{\textrm{BIE}} - u_{\textrm{analytic}}|$")
    plt.title("Error (number of digits of accuracy)")
    plt.tight_layout()
    plt.xlim([-XV, XV])
    plt.show()
```

```{code-cell} ipython3
import copy

VB = copy.deepcopy(free)
VB.pts[:, 1] -= visco_depth
stress_integral = np.zeros_like(VB.n_pts)
```

```{code-cell} ipython3
free_disp_to_VB_syz = (
    shear_modulus * qbx_matrix(hypersingular_matrix, free_interp, VB.pts, expansions)[:, 1, :]
).dot(Im_free)

fault_slip_to_VB_syz = shear_modulus * hypersingular_matrix(fault, VB.pts)[:, 1, :]

syz_free = free_disp_to_VB_syz.dot(free_disp)
syz_fault = fault_slip_to_VB_syz.dot(slip)
syz_full = syz_free + syz_fault
siay = 31556952
```

```{code-cell} ipython3
stress_integral = 20 * siay * (shear_modulus / viscosity) * syz_full
```

```{code-cell} ipython3
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_full)), 'k-')
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_free)), 'r-')
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_fault)), 'b-')
plt.show()
```

```{code-cell} ipython3
plt.plot(stress_integral)
plt.show()
```

```{code-cell} ipython3
import_and_display_fnc("common", "single_layer_matrix")
import_and_display_fnc("common", "adjoint_double_layer_matrix")
```

```{code-cell} ipython3
nobs = 100
zoomx = [-15000, 15000]
zoomy = [-31000, -1000]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
obsx = obs_pts[:, 0].reshape((nobs, nobs))
obsy = obs_pts[:, 1].reshape((nobs, nobs))

free_disp_to_volume_disp = double_layer_matrix(free, obs_pts)
fault_slip_to_volume_disp = double_layer_matrix(fault, obs_pts)
VB_S_to_volume_disp = (1.0 / shear_modulus) * single_layer_matrix(VB, obs_pts)

free_disp_to_volume_stress = shear_modulus * hypersingular_matrix(free, obs_pts)
fault_slip_to_volume_stress = shear_modulus * hypersingular_matrix(fault, obs_pts)
VB_S_to_volume_stress = adjoint_double_layer_matrix(VB, obs_pts)


def get_volumetric_disp(free_disp, slip, stress_integral):
    disp_free = free_disp_to_volume_disp.dot(free_disp)
    disp_fault = fault_slip_to_volume_disp.dot(slip)
    disp_VB = VB_S_to_volume_disp.dot(stress_integral)
    return (disp_free + disp_fault + disp_VB).reshape(obsx.shape)


def get_volumetric_stress(free_disp, slip, stress_integral):
    stress_free = free_disp_to_volume_stress.dot(free_disp)
    stress_fault = fault_slip_to_volume_stress.dot(slip)
    stress_VB = VB_S_to_volume_stress.dot(stress_integral)

    return (stress_free + stress_fault + stress_VB).reshape((*obsx.shape, 2))


def simple_plot(field, levels):
    n_dims = field.shape[2]
    plt.figure(figsize=(4 * n_dims, 4))
    for d in range(field.shape[2]):
        plt.subplot(1, n_dims, 1 + d)
        cntf = plt.contourf(
            obsx, obsy, field[:, :, d], levels=levels, extend="both", cmap="RdBu_r"
        )
        plt.contour(
            obsx,
            obsy,
            field[:, :, d],
            colors="k",
            linestyles="-",
            linewidths=0.5,
            levels=levels,
            extend="both",
        )
        plt.plot(free.pts[:, 0], free.pts[:, 1], "k-", linewidth=1.5)
        plt.plot(fault.pts[:, 0], fault.pts[:, 1], "k-", linewidth=1.5)
        plt.colorbar(cntf)
        plt.xlim(zoomx)
        plt.ylim(zoomy)
    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
for terms in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    disp = get_volumetric_disp(
        terms[0] * free_disp, terms[1] * slip, terms[2] * stress_integral
    )
    simple_plot(disp[:, :, None], None)
```

```{code-cell} ipython3
for terms in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    stress = get_volumetric_stress(
        terms[0] * free_disp, terms[1] * slip, terms[2] * stress_integral
    )
    levels = np.linspace(-2e4, 2e4, 21)
    simple_plot(stress, levels)
```

```{code-cell} ipython3
r = (
    np.repeat((VB.panel_bounds[:, 1] - VB.panel_bounds[:, 0]), VB.panel_sizes)
    * VB.jacobians
    * 0.5
)
VB_expansions = qbx_setup(VB, direction=-1, r=r, p=12)
VB_interp = copy.deepcopy(free_interp)
VB_interp.pts[:, 1] -= visco_depth

VB_S_to_free_disp2 = (1.0 / shear_modulus) * single_layer_matrix(VB_interp, free.pts)[:,0,:].dot(Im_free)
VB_S_to_free_disp = (1.0 / shear_modulus) * qbx_matrix(single_layer_matrix, VB_interp, free.pts, VB_expansions)[:,0,:].dot(Im_free)
```

```{code-cell} ipython3
# plt.plot(free.pts[:,0], free.pts[:,1], 'k-')
# plt.plot(expansions.pts[:,0], expansions.pts[:,1], 'k*')
# plt.plot(VB.pts[:,0], VB.pts[:,1], 'b-')
# plt.plot(VB_expansions.pts[:,0], VB_expansions.pts[:,1], 'b*')
# plt.plot(fault.pts[:,0], fault.pts[:,1], 'r-')
# plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.plot(VB_S_to_free_disp.dot(stress_integral), 'b-')
plt.plot(VB_S_to_free_disp2.dot(stress_integral), 'b-.')
plt.show()
```

```{code-cell} ipython3
VB_S_to_VB_syz_raw = qbx_matrix(
    adjoint_double_layer_matrix, VB_interp, VB.pts, VB_expansions
)[:, 1, :]
VB_S_to_VB_syz = VB_S_to_VB_syz_raw.dot(Im_free)
```

```{code-cell} ipython3
t_R_years = viscosity / shear_modulus / siay
```

```{code-cell} ipython3
from math import factorial


def Fn(n, x, D, H):
    return np.arctan(2 * x * D / (x ** 2 + (2 * n * H) ** 2 - D ** 2))


def analytic_to_surface(slip, D, H, x, t):
    t_R = viscosity / shear_modulus
    C = slip / np.pi
    T1 = np.arctan(D / x)
    T2 = 0
    for n in range(1, 50):
        m_factor = 0
        for m in range(1, n + 1):
            m_factor += ((t / t_R) ** (n - m)) / factorial(n - m)
        n_factor = 1 - np.exp(-t / t_R) * m_factor
        T2 += n_factor * Fn(n, x, D, H)
    return C * (T1 + T2)


def analytic(x, t):
    return analytic_to_surface(1.0, 15000, 20000, x, t)


for t in [0, 10.0 * siay, 20.0 * siay, 100.0 * siay]:
    plt.plot(free.pts[:, 0] / 1000.0, analytic(free.pts[:, 0], t), label=f"{t/siay:.0f} years")
plt.xlim([-1000, 1000])
plt.legend()
plt.show()
```

```{code-cell} ipython3
%%time
# The slip does not change so these two integral terms can remain
# outside the time stepping loop.
syz_fault = fault_slip_to_VB_syz.dot(slip)
rhs_slip = fault_slip_to_free_disp.dot(slip)

step_mult = 1
dt = 0.1 * siay / step_mult
stress_integral = np.zeros(VB.n_pts)
t = 0
disp_history = []
S_history = []
for i in range(1401 * step_mult):
    # Step 1) Solve for free surface displacement.
    rhs = rhs_slip + VB_S_to_free_disp.dot(stress_integral)
    free_disp = free_disp_solve_mat_inv.dot(rhs)
    if i % step_mult == 0:
        disp_history.append((t, free_disp))
        S_history.append((t, stress_integral.copy()))

    # Step 2): Calculate viscoelastic boundary stress yz component and then d[S]/dt
    syz_free = free_disp_to_VB_syz.dot(free_disp)
    syz_VB = -VB_S_to_VB_syz.dot(stress_integral)
    syz_full = syz_free + syz_fault + syz_VB
    dSdt = (shear_modulus / viscosity) * syz_full

    # Step 3): Update S, simple forward Euler time step.
    stress_integral += 2 * dSdt * dt
    t += dt
```

```{code-cell} ipython3
plt.figure(figsize=(14, 7))
X = free.pts[:, 0] / 1000
plt.subplot(1, 2, 1)
plt.plot(X, disp_history[0][1], "k-", linewidth=3, label="elastic")
plt.plot(X, analytic(free.pts[:, 0], disp_history[0][0]), "k-.", linewidth=3)
plt.plot(X, disp_history[100][1], "m-", label="10 yrs")
plt.plot(X, analytic(free.pts[:, 0], disp_history[100][0]), "m-.")
plt.plot(X, disp_history[200][1], "b-", label="20 yrs")
plt.plot(X, analytic(free.pts[:, 0], disp_history[200][0]), "b-.")
plt.plot(X, disp_history[300][1], "r-", label="30 yrs")
plt.plot(X, analytic(free.pts[:, 0], disp_history[300][0]), "r-.")
plt.plot([], [], " ", label="BIE = solid")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$u ~ \mathrm{(m)}$")
plt.legend()
plt.subplot(1,2,2)

plt.plot(X, np.log10(np.abs(analytic(free.pts[:, 0], disp_history[0][0]) - disp_history[0][1])), "k-.")
plt.plot(X, np.log10(np.abs(analytic(free.pts[:, 0], disp_history[100][0]) - disp_history[100][1])), "m-.")
plt.plot(X, np.log10(np.abs(analytic(free.pts[:, 0], disp_history[200][0]) - disp_history[200][1])), "b-.")
plt.plot(X, np.log10(np.abs(analytic(free.pts[:, 0], disp_history[300][0]) - disp_history[300][1])), "r-.")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$\log_{10}{|u_{\textrm{analytic}} - u|} ~ \mathrm{(m)}$")
plt.show()
```

```{code-cell} ipython3
for i in range(len(free.panel_bounds)):
    plt.plot([free.panel_bounds[i][0], free.panel_bounds[i][1]], [0, 0], 'k-*')
    plt.plot()
plt.show()
```

```{code-cell} ipython3
disp_history = disp_history[:1400]
```

```{code-cell} ipython3
plt.figure(figsize = (14,7))
plt.subplot(1,2,1)
for i in range(0, len(disp_history), 100):
    plt.plot(X, disp_history[i][1], "r-", linewidth = 0.5)
# for i in range(len(free.panel_bounds)):
#     ps = free.panel_start_idxs[i]
#     pe = ps + free.panel_sizes[i] - 1
#     xs = [X[ps], X[pe]]
#     ys = [disp_history[-1][1][ps], disp_history[-1][1][pe]]
#     plt.plot(xs, ys, 'k-*')
plt.subplot(1,2,2)
for i in range(0, len(S_history), 100):
    plt.plot(X, np.log10(np.abs(S_history[i][1])), "b-", linewidth = 0.5)
plt.show()
plt.figure()
for i in range(0, len(S_history), 100):
    plt.plot(X, S_history[i][1], "b-", linewidth = 0.5)
plt.show()
```

```{code-cell} ipython3
VB_S_to_fault_stress = adjoint_double_layer_matrix(VB, fault.pts)
free_disp_to_fault_stress = qbx_matrix(hypersingular_matrix, free_interp, fault.pts, fault_expansions)
fault_expansions = qbx_setup(VB, r=r, p=10)
fault_slip_to_fault_stress = qbx_matrix(hypersingular_matrix, fault_interp, fault.pts, fault_expansions)
```
