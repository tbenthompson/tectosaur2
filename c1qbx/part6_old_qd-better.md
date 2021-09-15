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

Plan

- Build a spherical surface, refined towards the fault-surface intersection. 
- Implement a rigid body motion constraint!
- Run a VE model on that. 
- Implement newton for solving the rate-state equations
- Solve for fault stress.
- Implement time stepping
- Run QD!

```{code-cell} ipython3
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    qbx_matrix,
    build_interp_matrix,
    build_interpolator,
    qbx_setup,
    single_layer_matrix,
    double_layer_matrix,
    adjoint_double_layer_matrix,
    hypersingular_matrix,
    PanelSurface,
    panelize_symbolic_surface,
    build_panel_interp_matrix,
    pts_grid,
)
```

```{code-cell} ipython3
corner_resolution = 250000
n_panels = 20

earth_radius = 6378 * 1000
visco_depth = 1000000
viscosity = 1e18
shear_modulus = 3e9
```

```{code-cell} ipython3
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
    panel_end = panel_start + min(corner_resolution * (2 ** i), 6.5e6)
    panels.append((panel_start, panel_end))
    panels.insert(0, (-panel_end, -panel_start))
panels = np.array(panels)
scaled_panels = panels / panels[-1][1]
```

```{code-cell} ipython3
40960000 / 1e6
```

```{code-cell} ipython3
qx, qw = gauss_rule(16)
qinterp = gauss_rule(96)
t = sp.var("t")
x = earth_radius * sp.cos(sp.pi/2 + sp.pi*t)
y = earth_radius * sp.sin(sp.pi/2 + sp.pi*t)
free = panelize_symbolic_surface(t, x, y, scaled_panels, qx, qw)
free_interp = panelize_symbolic_surface(t, x, y, scaled_panels, *qinterp)
Im_free = build_panel_interp_matrix(free, free_interp)
```

```{code-cell} ipython3
free.pts[free.n_pts // 2 - 1, 0] - free.pts[free.n_pts // 2, 0]
```

```{code-cell} ipython3
free.n_pts
```

```{code-cell} ipython3
fault_top = 0.0
fault_bottom = -corner_resolution * 3

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
fault_x = 0 * t
fault_y = earth_radius + fault_top + (fault_bottom - fault_top) * (t + 1) * 0.5
fault = panelize_symbolic_surface(t, fault_x, fault_y, scaled_fault_panels, qx, qw)
```

```{code-cell} ipython3
import copy
VB = copy.deepcopy(free)
VB.pts *= (earth_radius - visco_depth) / earth_radius
VB.jacobians *= (earth_radius - visco_depth) / earth_radius
```

```{code-cell} ipython3
VB_interp = copy.deepcopy(free_interp)
VB_interp.pts *= (earth_radius - visco_depth) / earth_radius
VB_interp.jacobians *= (earth_radius - visco_depth) / earth_radius
```

```{code-cell} ipython3
from common import QBXExpansions

free_r = (
    np.repeat((free.panel_bounds[:, 1] - free.panel_bounds[:, 0]), free.panel_sizes)
    * free.jacobians
    * 0.5
)
orig_expansions = qbx_setup(free, direction=1, r=free_r, p=10)
VB_r = (
    np.repeat((VB.panel_bounds[:, 1] - VB.panel_bounds[:, 0]), VB.panel_sizes)
    * VB.jacobians
    * 0.5
)
VB_expansions = qbx_setup(VB, direction = -1, r=VB_r, p=13)

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
plt.xlim(-50000, 50000)
plt.ylim(earth_radius - 90000, earth_radius + 10000)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
```

```{code-cell} ipython3
plt.plot(free.pts[:,0], free.pts[:,1], 'k-')
plt.plot(expansions.pts[:,0], expansions.pts[:,1], 'k*')
plt.plot(VB.pts[:,0], VB.pts[:,1], 'b-')
plt.plot(VB_expansions.pts[:,0], VB_expansions.pts[:,1], 'b*')
plt.plot(fault.pts[:,0], fault.pts[:,1], 'r-')
plt.show()
```

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
A = np.eye(free_disp_to_free_disp.shape[0]) + free_disp_to_free_disp
```

```{code-cell} ipython3
constraint_row = free.jacobians * free.quad_wts
free_disp_solve_mat = np.zeros((A.shape[0] + 1, A.shape[0] + 1))
free_disp_solve_mat[:-1, :-1] = A
free_disp_solve_mat[:-1, -1] = constraint_row
free_disp_solve_mat[-1, :-1] = constraint_row
```

```{code-cell} ipython3
free_disp_solve_mat_inv = np.linalg.inv(free_disp_solve_mat)

slip = np.ones(fault.n_pts)
rhs = np.zeros(A.shape[1] + 1)
rhs[:-1] = fault_slip_to_free_disp.dot(slip)
soln = free_disp_solve_mat_inv.dot(rhs)
free_disp = soln[:-1]

# Note that the analytical solution is slightly different than in the buried
# fault setting because we need to take the limit of an arctan as the
# denominator of the argument  goes to zero.
s = 1.0
analytical_y = free.pts[:, 1] - earth_radius
analytical = (
    -s
    / (2 * np.pi)
    * (
        np.arctan(free.pts[:, 0] / (analytical_y - fault_bottom))
        - np.arctan(free.pts[:, 0] / (analytical_y + fault_bottom))
        - np.pi * np.sign(free.pts[:, 0])
    )
)
```

```{code-cell} ipython3
XV = 4000000.0
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
nobs = 100
zoomx = [-600000, 600000]
zoomy = [earth_radius - 1240000, earth_radius - 10000]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
obsx = obs_pts[:, 0].reshape((nobs, nobs))
obsy = obs_pts[:, 1].reshape((nobs, nobs))

free_disp_to_volume_disp = double_layer_matrix(free, obs_pts)[:, 0, :]
fault_slip_to_volume_disp = double_layer_matrix(fault, obs_pts)[:, 0, :]
VB_S_to_volume_disp = (1.0 / shear_modulus) * single_layer_matrix(VB, obs_pts)[:, 0, :]
```

```{code-cell} ipython3
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
        plt.xlim(*zoomx)
        plt.ylim(*zoomy)
    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
stress_integral = np.zeros(VB.n_pts)
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
free_disp_to_VB_stress = (
    shear_modulus * qbx_matrix(hypersingular_matrix, free_interp, VB.pts, expansions)
).dot(Im_free)

fault_slip_to_VB_stress = shear_modulus * hypersingular_matrix(fault, VB.pts)
```

```{code-cell} ipython3
VB_S_to_free_disp = (1.0 / shear_modulus) * qbx_matrix(single_layer_matrix, VB_interp, free.pts, VB_expansions)[:,0,:].dot(Im_free)
```

```{code-cell} ipython3
VB_S_to_VB_stress_raw = qbx_matrix(
    adjoint_double_layer_matrix, VB_interp, VB.pts, VB_expansions
)
VB_S_to_VB_stress = VB_S_to_VB_stress_raw.dot(Im_free)
```

```{code-cell} ipython3
free_disp_to_VB_traction = np.sum(-VB.normals[:, :, None] * free_disp_to_VB_stress, axis=1)
fault_slip_to_VB_traction = np.sum(-VB.normals[:, :, None] * fault_slip_to_VB_stress, axis=1)
VB_S_to_VB_traction = np.sum(-VB.normals[:, :, None] * VB_S_to_VB_stress, axis=1)
```

```{code-cell} ipython3
%%time
# The slip does not change so these two integral terms can remain
# outside the time stepping loop.
VB_traction_fault = fault_slip_to_VB_traction.dot(slip)
rhs_slip = fault_slip_to_free_disp.dot(slip)

siay = 31556952
dt = 0.1 * siay
stress_integral = np.zeros(VB.n_pts)
t = 0
disp_history = []
rhs = np.zeros(free_disp_solve_mat_inv.shape[1])
for i in range(305):
    # Step 1) Solve for free surface displacement.
    rhs[:-1] = rhs_slip + VB_S_to_free_disp.dot(stress_integral)
    soln = free_disp_solve_mat_inv.dot(rhs)
    free_disp = soln[:-1]
    disp_history.append((t, free_disp))
    
    # Step 2): Calculate viscoelastic boundary stress yz component and then d[S]/dt
    VB_traction_free = free_disp_to_VB_traction.dot(free_disp)
    VB_traction_VB = -VB_S_to_VB_traction.dot(stress_integral)
    VB_traction_full = VB_traction_free + VB_traction_fault + VB_traction_VB
    dSdt = (shear_modulus / viscosity) * VB_traction_full

    # Step 3): Update S, simple forward Euler time step.
    stress_integral -= 2 * dSdt * dt
    t += dt
    
    if i % 25 == 0:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(stress_integral)
        plt.subplot(1,2,2)
        plt.plot(free_disp)
        plt.show()
```

```{code-cell} ipython3
# The constraint is working.
np.sum(free_disp * free.jacobians * free.quad_wts)
```

```{code-cell} ipython3
plt.figure(figsize=(7, 7))
X = free.pts[:, 0] / 1000
plt.plot(X, disp_history[0][1], "k-", linewidth=3, label="elastic")
# plt.plot(X, disp_history[1][1], "b-", linewidth=3, label="elastic")
# plt.plot(X, disp_history[2][1], "r-", linewidth=3, label="elastic")
# plt.plot(X, disp_history[3][1], "m-", linewidth=3, label="elastic")
plt.plot(X, disp_history[100][1], "m-", label="10 yrs")
plt.plot(X, disp_history[200][1], "b-", label="20 yrs")
plt.plot(X, disp_history[300][1], "r-", label="30 yrs")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$u ~ \mathrm{(m)}$")
plt.legend()
plt.show()
```
