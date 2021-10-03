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
import sympy as sp
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
    stage1_refine,
    qbx_panel_setup,
    stage2_refine,
)
```

This next section will construct the free surface panelized surface `free`. The `corner_resolution` specifies how large the panels will be near the fault-surface intersection. Away from that intersection, the panels will each be double the length of the prior panel, thus enabling the full surface to efficiently represent an effectively infinite free surface.

```{code-cell} ipython3
corner_resolution = 5000
surf_half_L = 1000000
fault_bottom = 15000
visco_depth = 20000

qx, qw = gauss_rule(6)
t = sp.var("t")

control_points = np.array([(0, 0, 2, corner_resolution)])
fault, free, VB = stage1_refine(
    [
        (t, t * 0, fault_bottom * (t + 1) * -0.5), # fault
        (t, -t * surf_half_L, 0 * t), # free surface
        (t, -t * surf_half_L, -visco_depth + 0 * t), # viscoelastic boundary
    ],
    (qx, qw),
    control_points=control_points,
)
fault = stage1_refine((t, t * 0, fault_bottom * (t + 1) * -0.5), (qx, qw))
free = stage1_refine(
    (t, -t * surf_half_L, 0 * t),
    (qx, qw),
    other_surfaces=[fault],
    control_points=control_points,
)
VB = stage1_refine(
    (t, -t * surf_half_L, -visco_depth + 0 * t), (qx, qw), other_surfaces=[fault, free]
)

free_expansions = qbx_panel_setup(free, other_surfaces=[fault], direction=1, p=10)
VB_expansions = qbx_panel_setup(free, other_surfaces=[fault, free], direction=1, p=10)

fault_stage2, fault_interp_mat = stage2_refine(fault, expansions, kappa=3)
free_stage2, free_interp_mat = stage2_refine(free, expansions)
VB_stage2, VB_interp_mat = stage2_refine(VB, expansions)
```

```{code-cell} ipython3
free.panel_length, VB.panel_length
```

```{code-cell} ipython3
print("number of points in the free surface discretization:", free.n_pts)
print("                        fault        discretization:", fault.n_pts)
print("                        free surface     quadrature:", free_stage2.n_pts)
print("                        fault            quadrature:", fault_stage2.n_pts)
```

```{code-cell} ipython3
%matplotlib inline
plt.figure()
plt.plot(fault.pts[:, 0], fault.pts[:, 1], "r-o")
plt.plot(fault_stage2.pts[:, 0], fault_stage2.pts[:, 1], "r*")
plt.plot(free_stage2.pts[:, 0], free_stage2.pts[:, 1], "k-o")
plt.plot(expansions.pts[:, 0], expansions.pts[:, 1], "bo")
for i in range(expansions.N):
    plt.gca().add_patch(
        plt.Circle(expansions.pts[i], expansions.r[i], color="b", fill=False)
    )
plt.xlim([-500, 500])
plt.ylim([-500, 1])
plt.show()
```

```{code-cell} ipython3
expansions2 = qbx_panel_setup(free, other_surfaces=[fault_stage2], direction=1, p=10)

expansions = expansions2
fault_stage2, fault_interp_mat = stage2_refine(fault, expansions, kappa=3)
free_stage2, free_interp_mat = stage2_refine(free, expansions)
```

```{code-cell} ipython3
print("number of points in the free surface discretization:", free.n_pts)
print("                        fault        discretization:", fault.n_pts)
print("                        free surface     quadrature:", free_stage2.n_pts)
print("                        fault            quadrature:", fault_stage2.n_pts)
```

```{code-cell} ipython3
%matplotlib inline
plt.figure()
plt.plot(fault.pts[:, 0], fault.pts[:, 1], "r-o")
plt.plot(fault_stage2.pts[:, 0], fault_stage2.pts[:, 1], "r*")
plt.plot(free_stage2.pts[:, 0], free_stage2.pts[:, 1], "k-o")
plt.plot(expansions.pts[:, 0], expansions.pts[:, 1], "bo")
for i in range(expansions.N):
    plt.gca().add_patch(
        plt.Circle(expansions.pts[i], expansions.r[i], color="b", fill=False)
    )
plt.xlim([-500, 500])
plt.ylim([-500, 1])
plt.show()
```

```{code-cell} ipython3
A_raw = qbx_matrix(double_layer_matrix, free_stage2, free.pts, expansions)[:, 0, :]
free_disp_to_free_disp = A_raw.dot(free_interp_mat.toarray())
```

```{code-cell} ipython3
fault_slip_to_free_disp = -qbx_matrix(
    double_layer_matrix, fault_stage2, free.pts, expansions
)[:, 0, :]
```

```{code-cell} ipython3
free_disp_solve_mat = np.eye(free_disp_to_free_disp.shape[0]) + free_disp_to_free_disp
free_disp_solve_mat_inv = np.linalg.inv(free_disp_solve_mat)

slip = np.ones(fault_stage2.n_pts)
free_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip))

# Note that the analytical solution is slightly different than in the buried
# fault setting because we need to take the limit of an arctan as the
# denominator of the argument  goes to zero.
s = 1.0
analytical_fnc = lambda x: -np.arctan(-fault_bottom / x) / np.pi
analytical = analytical_fnc(free.pts[:, 0])
```

```{code-cell} ipython3
for XV in [50000.0]:
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
    shear_modulus
    * qbx_matrix(hypersingular_matrix, free_interp, VB.pts, expansions)[:, 1, :]
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
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_full)), "k-")
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_free)), "r-")
plt.plot(VB.pts[:, 0], np.log10(np.abs(syz_fault)), "b-")
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
    levels = np.linspace(-1e5, 1e5, 21)
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

VB_S_to_free_disp2 = (1.0 / shear_modulus) * single_layer_matrix(VB_interp, free.pts)[
    :, 0, :
].dot(Im_free)
VB_S_to_free_disp = (1.0 / shear_modulus) * qbx_matrix(
    single_layer_matrix, VB_interp, free.pts, VB_expansions
)[:, 0, :].dot(Im_free)
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
plt.plot(VB_S_to_free_disp.dot(stress_integral), "b-")
plt.plot(VB_S_to_free_disp2.dot(stress_integral), "b-.")
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


def analytic_soln(x, t):
    return analytic_to_surface(1.0, 15000, 20000, x, t)


plt.figure(figsize=(6, 6))
for t in [0, 10.0 * siay, 20.0 * siay, 100.0 * siay]:
    plt.plot(
        free.pts[:, 0] / 1000.0,
        analytic(free.pts[:, 0], t),
        label=f"{t/siay:.0f} years",
    )
plt.xlim([-100, 100])
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
for color, i in [("m", 100), ("b", 200), ("r", 300)]:
    plt.plot(X, disp_history[i][1], color + "-", label=str(i) + " yrs")
    plt.plot(X, analytic_soln(free.pts[:, 0], disp_history[i][0]), color + "-.")
plt.plot([], [], " ", label="BIE = solid")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$u ~ \mathrm{(m)}$")
plt.legend()

plt.subplot(1, 2, 2)
for color, i in [("k", 0), ("m", 100), ("b", 200), ("r", 300)]:
    analytic = analytic_soln(free.pts[:, 0], disp_history[i][0])
    numerical = disp_history[i][1]
    diff = analytic - numerical
    plt.plot(X, np.log10(np.abs(diff)), color + "-.")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$\log_{10}{|u_{\textrm{analytic}} - u|} ~ \mathrm{(m)}$")
plt.show()
```

## MAJOR PROBLEMS REMAIN, SEE BELOW

```{code-cell} ipython3
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for i in range(0, len(disp_history), 100):
    plt.plot(X, disp_history[i][1], "r-", linewidth=0.5)
# for i in range(len(free.panel_bounds)):
#     ps = free.panel_start_idxs[i]
#     pe = ps + free.panel_sizes[i] - 1
#     xs = [X[ps], X[pe]]
#     ys = [disp_history[-1][1][ps], disp_history[-1][1][pe]]
#     plt.plot(xs, ys, 'k-*')
plt.subplot(1, 2, 2)
for i in range(0, len(S_history), 100):
    plt.plot(X, np.log10(np.abs(S_history[i][1])), "b-", linewidth=0.5)
plt.show()
plt.figure()
for i in range(0, len(S_history), 100):
    plt.plot(X, S_history[i][1], "b-", linewidth=0.5)
plt.show()
```

```{code-cell} ipython3
VB_S_to_fault_stress = adjoint_double_layer_matrix(VB, fault.pts)
free_disp_to_fault_stress = qbx_matrix(
    hypersingular_matrix, free_interp, fault.pts, fault_expansions
)
fault_expansions = qbx_setup(VB, r=r, p=10)
fault_slip_to_fault_stress = qbx_matrix(
    hypersingular_matrix, fault_interp, fault.pts, fault_expansions
)
```
