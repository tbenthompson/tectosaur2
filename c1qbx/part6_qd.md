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
    qbx_matrix2,
    single_layer_matrix,
    double_layer_matrix,
    adjoint_double_layer_matrix,
    hypersingular_matrix,
    stage1_refine,
    qbx_panel_setup,
    stage2_refine,
    pts_grid,
)
```

BUGBUGBUGBUG:
```
surf_half_L = 100000
corner_resolution = 5000
fault_bottom = 17000
shear_modulus = 3e10

qx, qw = gauss_rule(6)
t = sp.var("t")

control_points = [
    (0, 0, 0, corner_resolution),
    (0, -fault_bottom / 2, fault_bottom / 1.9, 500)
]
fault, free = stage1_refine(
    [
        (t, t * 0, fault_bottom * (t + 1) * -0.5),  # fault
        (t, -t * surf_half_L, 0 * t),  # free surface
    ],
    (qx, qw),
    control_points=control_points
)

fault_expansions, free_expansions = qbx_panel_setup(
    [fault, free], directions=[0, 1], p=10
)
```

```{code-cell} ipython3
surf_half_L = 100000
corner_resolution = 5000
fault_bottom = 16500
shear_modulus = 3e10

qx, qw = gauss_rule(6)
t = sp.var("t")

control_points = [
    (0, 0, 0, corner_resolution),
    (0, -fault_bottom / 2, fault_bottom / 1.9, 500)
]
fault, free = stage1_refine(
    [
        (t, t * 0, fault_bottom * (t + 1) * -0.5),  # fault
        (t, -t * surf_half_L, 0 * t),  # free surface
    ],
    (qx, qw),
    control_points=control_points
)

fault_expansions, free_expansions = qbx_panel_setup(
    [fault, free], directions=[0, 1], p=10
)
```

```{code-cell} ipython3
free_disp_to_free_disp = qbx_matrix2(
    double_layer_matrix, free, free.pts, free_expansions
)[:, 0, :]
fault_slip_to_free_disp = -qbx_matrix2(
    double_layer_matrix, fault, free.pts, free_expansions
)[:, 0, :]

free_disp_solve_mat = np.eye(free_disp_to_free_disp.shape[0]) + free_disp_to_free_disp
free_disp_solve_mat_inv = np.linalg.inv(free_disp_solve_mat)
```

```{code-cell} ipython3
slip = np.ones(fault.n_pts)

surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip))

# Note that the analytical solution is slightly different than in the buried
# fault setting because we need to take the limit of an arctan as the
# denominator of the argument  goes to zero.
s = 1.0
analytical_fnc = lambda x: -np.arctan(-fault_bottom / x) / np.pi
analytical = analytical_fnc(free.pts[:, 0])
```

In the first row of graphs below, I show the solution extending to 10 fault lengths. In the second row, the solution extends to 1000 fault lengths. You can see that the solution matches to about 6 digits in the nearfield and 7-9 digits in the very farfield!

```{code-cell} ipython3
%matplotlib inline
for XV in [50000]:
    # XV = 5 * corner_resolution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(free.pts[:, 0], surf_disp, "ko")
    plt.plot(free.pts[:, 0], analytical, "bo")
    plt.xlabel("$x$")
    plt.ylabel("$u_z$")
    plt.title("Displacement")
    plt.xlim([-XV, XV])
    plt.ylim([-0.6, 0.6])

    plt.subplot(1, 2, 2)
    plt.plot(free.pts[:, 0], np.log10(np.abs(surf_disp - analytical)))
    plt.xlabel("$x$")
    plt.ylabel(r"$\log_{10}|u_{\textrm{BIE}} - u_{\textrm{analytic}}|$")
    plt.title("Error (number of digits of accuracy)")
    plt.tight_layout()
    plt.xlim([-XV, XV])
    plt.show()
```

```{code-cell} ipython3
fault_slip_to_fault_stress = qbx_matrix2(
    hypersingular_matrix, fault, fault.pts, fault_expansions
)
free_disp_to_fault_stress = qbx_matrix2(
    hypersingular_matrix, free, fault.pts, fault_expansions
)
```

```{code-cell} ipython3
slip = np.cos(np.pi * fault.pts[:, 1] / 30000)
slip = np.cos(fault.pts[:,1] / 5000) ** 2 / (1 + np.exp(-(fault.pts[:, 1] + 12500) / 300.0))
#slip = np.log(1 + np.exp((13000+fault.pts[:,1])/100))/100
#slip = 1.0 / (1 + np.exp(-(15000+fault.pts[:,1]) / 100.0))
print('slip')
plt.plot(fault.pts[:, 1], slip)
plt.show()

surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip))
print('free surf disp')
plt.plot(free.pts[:, 0], surf_disp)
plt.show()

differential_stress = shear_modulus * (
    fault_slip_to_fault_stress.dot(slip) + free_disp_to_fault_stress.dot(surf_disp)
)

print('stress')
plt.plot(fault.pts[:, 1], differential_stress[:, 0], "b-", label='sxz')
plt.plot(fault.pts[:, 1], differential_stress[:, 1], "r-", label='syz')
plt.legend()
plt.show()
```

```{code-cell} ipython3
from dataclasses import dataclass
```

```{code-cell} ipython3
@dataclass
class FrictionParams:
    a: float
    b: float
    V0: float
    Dc: float
    f0: float
```

```{code-cell} ipython3
density = 2700             # rock density (kg/m^3)
cs = np.sqrt(shear_modulus / density) # Shear wave speed (m/s)
eta = shear_modulus / (2 * cs)        # The radiation damping coefficient (kg / (m^2 * s))
Vp = 1e-9                  # Rate of plate motion
sigma_n = 50e6   # Normal stress (Pa)

fp = FrictionParams(
    a = 0.015,        # direct velocity strengthening effect
    b = 0.02,         # state-based velocity weakening effect
    Dc = 0.2,         # state evolution length scale (m)
    f0 = 0.6,         # baseline coefficient of friction
    V0 = 1e-6,        # when V = V0, f = f0, V is (m/s)
)
```

```{code-cell} ipython3
def aging_law(fp, V, state):
    return (fp.b * fp.V0 / fp.Dc) * (
        np.exp((fp.f0 - state) / fp.b) - (V / fp.V0)
    )
```

```{code-cell} ipython3
def solve_state
```

```{code-cell} ipython3
from scipy.optimize import fsolve
presumed_init_velocity = Vp / 1000.0
init_state_scalar = fsolve(lambda S: aging_law(fp, presumed_init_velocity, S), 0.0)[0]
init_state = np.full(fault.n_pts, init_state_scalar)

init_slip = slip
init_conditions = np.concatenate((init_slip, init_state))
```

```{code-cell} ipython3
from scipy.integrate import RK23
tol = 1e-5
rk23 = RK23(
    derivs_fnc,
    0,
    init_conditions,
    1e50,
    atol = tol,
    rtol = tol
)
rk23.h_abs = siay / 10.0
```

```{code-cell} ipython3
done = False
while not done:
    dt = 
    slip += velocity * dt
    traction = elastic_solver(slip)
    velocity = rate_state_solver(traction)
```

problem: how do I impose a backslip forcing? ultimately, this is physically unrealistic but I need to do it anyway. 
- look at how the scec seas project does this?
- use some tapering function?
- have a basal panel that just has a linear imposed backslip.
