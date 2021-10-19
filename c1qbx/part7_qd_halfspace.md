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

```{code-cell} ipython3
surf_half_L = 100000
corner_resolution = 5000
fault_bottom = 16500
shear_modulus = 3e10

qx, qw = gauss_rule(6)
t = sp.var("t")

control_points = [
    (0, -fault_bottom / 2, fault_bottom / 2, 600),
]
fault, free = stage1_refine(
    [
        (t, t * 0, fault_bottom * (t + 1) * -0.5),  # fault
        (t, -t * surf_half_L, 0 * t),  # free surface
    ],
    (qx, qw),
    control_points=control_points,
)

fault_expansions, free_expansions = qbx_panel_setup(
    [fault, free], directions=[0, 1], p=10
)
```

```{code-cell} ipython3
fault.panel_length
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
fault_slip_to_fault_stress = shear_modulus * qbx_matrix2(
    hypersingular_matrix, fault, fault.pts, fault_expansions
)
free_disp_to_fault_stress = shear_modulus * qbx_matrix2(
    hypersingular_matrix, free, fault.pts, fault_expansions
)
```

```{code-cell} ipython3
slip = np.cos(np.pi * fault.pts[:, 1] / 30000)
slip = np.cos(fault.pts[:, 1] / 5000) ** 2 / (
    1 + np.exp(-(fault.pts[:, 1] + 12500) / 300.0)
)
# slip = np.log(1 + np.exp((13000+fault.pts[:,1])/100))/100
sigmoid = 1.0 / (1 + np.exp(-(15000 + fault.pts[:, 1]) / 100.0))
slip = sigmoid
print("slip")
plt.plot(fault.pts[:, 1], slip)
plt.show()

surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip))
print("free surf disp")
plt.plot(free.pts[:, 0], surf_disp)
plt.show()

differential_stress = fault_slip_to_fault_stress.dot(
    slip
) + free_disp_to_fault_stress.dot(surf_disp)

print("stress")
plt.plot(fault.pts[:, 1], differential_stress[:, 0], "b-", label="sxz")
plt.plot(fault.pts[:, 1], differential_stress[:, 1], "r-", label="syz")
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
siay = 31556952
density = 2700  # rock density (kg/m^3)
cs = np.sqrt(shear_modulus / density)  # Shear wave speed (m/s)
eta = shear_modulus / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
Vp = 1e-9  # Rate of plate motion
sigma_n = 50e6  # Normal stress (Pa)


fp = FrictionParams(
    a=0.010,  # direct velocity strengthening effect
    b=0.015,  # state-based velocity weakening effect
    Dc=0.2,  # state evolution length scale (m)
    f0=0.6,  # baseline coefficient of friction
    V0=1e-6,  # when V = V0, f = f0, V is (m/s)
)
```

```{code-cell} ipython3
fault.panel_length
```

```{code-cell} ipython3
mesh_L = np.max(fault.panel_length / (fault.n_pts / fault.n_panels))
Lb = shear_modulus * fp.Dc / (sigma_n * fp.b)
hstar = (
    (np.pi * shear_modulus * fp.Dc) /
    (sigma_n * (fp.b - fp.a))
)
mesh_L, Lb, hstar
```

```{code-cell} ipython3
def aging_law(fp, V, state):
    return (fp.b * fp.V0 / fp.Dc) * (np.exp((fp.f0 - state) / fp.b) - (V / fp.V0))
```

```{code-cell} ipython3
def F(fp, V, state):
    return fp.a * sigma_n * np.arcsinh(V / (2 * fp.V0) * np.exp(state / fp.a))


def dFdV(fp, V, state):
    expsa = np.exp(state / fp.a)
    Q = (V * expsa) / (2 * fp.V0)
    return fp.a * expsa * sigma_n / (2 * fp.V0 * np.sqrt(1 + Q * Q))


def qd_equation(fp, shear_stress, V, state):
    return shear_stress - eta * V - F(fp, V, state)


def qd_equation_dV(fp, V, state):
    return -eta - dFdV(fp, V, state)
```

```{code-cell} ipython3
import scipy.optimize
```

```{code-cell} ipython3
from scipy.optimize import fsolve

presumed_init_velocity = Vp
init_state_scalar = fsolve(lambda S: aging_law(fp, presumed_init_velocity, S), 0.7)[0]
init_state = np.full(fault.n_pts, init_state_scalar)

tau_i = F(fp, presumed_init_velocity, init_state_scalar) + eta * presumed_init_velocity
init_traction = np.full(fault.n_pts, tau_i)
```

$$
\int_{H} H^* u + \int_{F} H^* s = t
$$
$$
u + \int_{H} T^* u + \int_{F} T^* s = 0
$$

```{code-cell} ipython3
A = fault_slip_to_fault_stress[:,0,:]
B = free_disp_to_fault_stress[:,0,:]
C = -fault_slip_to_free_disp
Dinv = free_disp_solve_mat_inv
M = B.dot(Dinv.dot(C))
```

```{code-cell} ipython3
slip_deficit_fullspace = np.linalg.inv(fault_slip_to_fault_stress[:, 0, :]).dot(
    init_traction
)
slip_deficit_halfspace = np.linalg.inv(A - B.dot(Dinv.dot(C))).dot(
    init_traction
)
```

```{code-cell} ipython3
plt.plot(fault.pts[:, 1], slip_deficit_fullspace, "r-")
plt.plot(fault.pts[:, 1], slip_deficit_halfspace, "b-")
plt.show()
```

```{code-cell} ipython3
surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip_deficit_halfspace))
stress = free_disp_to_fault_stress.dot(surf_disp) + fault_slip_to_fault_stress.dot(
    slip_deficit_halfspace
)
plt.plot(fault.pts[:, 1], init_traction, "r-")
plt.plot(fault.pts[:, 1], stress[:, 0], "k-")
plt.show()
```

```{code-cell} ipython3
init_slip = -slip_deficit_halfspace
```

```{code-cell} ipython3
init_conditions = np.concatenate((init_slip, init_state))
```

```{code-cell} ipython3
plate_motion_backslip = np.full(
    fault.n_pts, Vp
)
#plate_motion_backslip = Vp * 1.0 / (1 + np.exp(-(15000 + fault.pts[:, 1]) / 100.0))
#plate_motion_backslip = -Vp * slip_deficit_halfspace / np.max(np.abs(slip_deficit_halfspace))
plt.plot(fault.pts[:,1], plate_motion_backslip)
plt.show()

SD = plate_motion_backslip
surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(SD))
stress = free_disp_to_fault_stress.dot(surf_disp) + fault_slip_to_fault_stress.dot(
    SD
)

plt.plot(fault.pts[:, 1], SD)
plt.show()
plt.plot(free.pts[:, 0], surf_disp)
plt.show()
plt.plot(fault.pts[:, 1], stress[:, 0], "r-")
plt.plot(fault.pts[:, 1], stress[:, 1], "b-")
plt.show()
```

```{code-cell} ipython3
# slip, state, slip_deficit, surf_disp, stress, V, dstatedt = calc_derivatives.state

# def qd(V):
#     return qd_equation(fp, stress[:,0], V, state)

# def qd_dV(V):
#     return qd_equation_dV(fp, V, state)

# V = scipy.optimize.newton(qd, calc_system_state.V_old, fprime=qd_dV)

# plt.plot(fault.pts[:,1], stress[:,0])
# plt.show()
# plt.plot(fault.pts[:,1], V)
# plt.show()
```

```{code-cell} ipython3
def calc_system_state(t, y, verbose=False):
    if verbose:
        print(t)
        print(t)
        print(t)
        print(t)
        print(t)
    
    slip = y[: init_slip.shape[0]]
    state = y[init_slip.shape[0] :]
    
    
    if np.any(state < 0) or np.any(state > 1.2):
        #print("BAD STATE VALUES")
        return False

    slip_deficit = t * plate_motion_backslip - slip
    surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip_deficit))
    stress = free_disp_to_fault_stress.dot(surf_disp) + fault_slip_to_fault_stress.dot(
        slip_deficit
    )
    shear = np.sum(stress * fault.normals, axis=1)

    # print(calc_derivatives.V_old)    
    def qd(V):
        return qd_equation(fp, shear, V, state)

    def qd_dV(V):
        return qd_equation_dV(fp, V, state)

    try:
        V = scipy.optimize.newton(qd, calc_system_state.V_old, fprime=qd_dV)
    except RuntimeError:
        return False
    V = Vp * (1 - sigmoid) + V * sigmoid
    
    #print(slip[0], state[0], shear[0], calc_derivatives.V_old[0], V[0])
    
    if not np.all(np.isfinite(V)):
        return False

    dstatedt = aging_law(fp, V, state)
    
    out = slip, state, slip_deficit, surf_disp, stress, V, dstatedt
    if verbose:
        plot_system_state(out)
    if np.any(np.abs(V) > 1):
        plot_system_state(out)
        import ipdb;ipdb.set_trace()
    calc_system_state.V_old = V

    return out
calc_system_state.V_old = np.full(fault.n_pts, presumed_init_velocity)

def plot_system_state(SS):
    slip, state, slip_deficit, surf_disp, stress, V, dstatedt = SS
    plt.title('slip')
    plt.plot(fault.pts[:,1], slip)
    plt.show()

    plt.title('state')
    plt.plot(fault.pts[:,1], state)
    plt.show()
    plt.title('shear')
    plt.plot(fault.pts[:,1], stress[:,0])
    plt.show()
    plt.title('velocity')
    plt.plot(fault.pts[:,1], V)
    plt.show()
    plt.title('dstatedt')
    plt.plot(fault.pts[:,1], dstatedt)
    plt.show()

def calc_derivatives(t, y):
    if not np.all(np.isfinite(y)):
        return np.inf * y
    state = calc_system_state(t, y)#, verbose=True)
    if not state:
        return np.inf * y
    calc_derivatives.state = state
    derivatives = np.concatenate((state[-2], state[-1]))
    return derivatives
```

```{code-cell} ipython3
_ = calc_system_state(siay, init_conditions)
```

```{code-cell} ipython3
t_history
```

```{code-cell} ipython3
from scipy.integrate import RK23

calc_system_state.V_old = np.full(fault.n_pts, presumed_init_velocity)

tol = 1e-5
rk23 = RK23(calc_derivatives, 0, init_conditions, 1e50, atol=tol, rtol=tol, max_step = 0.01 * siay)
rk23.h_abs = 1.0

n_steps = 5000
t_history = [0]
y_history = [init_conditions.copy()]
for i in range(n_steps):
    print(i)
    if rk23.step() != None:
        print(i)
        plot_system_state(calc_derivatives.state)
        break
#     if i > 7:
#         print(i)
#         print(i)
#         print(i)
#         print(i)
#         plot_system_state(calc_derivatives.state)
    if np.any(calc_derivatives.state[-2] > 1):
        plot_system_state(calc_derivatives.state)
        break#import ipdb;ipdb.set_trace()
    t_history.append(rk23.t)
    y_history.append(rk23.y.copy())
```

```{code-cell} ipython3
i = 500
t = t_history[i]
y = y_history[i]
slip = y[: init_slip.shape[0]]
state = y[init_slip.shape[0] :]

slip_deficit = t * plate_motion_backslip - slip
surf_disp = free_disp_solve_mat_inv.dot(fault_slip_to_free_disp.dot(slip_deficit))
stress = free_disp_to_fault_stress.dot(surf_disp) + fault_slip_to_fault_stress.dot(
    slip_deficit
)
shear = np.sum(stress * fault.normals, axis=1)

# print(calc_derivatives.V_old)    
def qd(V):
    return qd_equation(fp, shear, V, state)

def qd_dV(V):
    return qd_equation_dV(fp, V, state)

V = scipy.optimize.newton(qd, calc_system_state.V_old, fprime=qd_dV)
```

```{code-cell} ipython3
plt.title('slip')
plt.plot(fault.pts[:,1], slip)
plt.plot(fault.pts[:,1], slip_deficit)
plt.show()

plt.title('state')
plt.plot(fault.pts[:,1], state)
plt.show()
plt.title('shear')
plt.plot(fault.pts[:,1], stress[:,0])
plt.show()
plt.title('velocity')
plt.plot(fault.pts[:,1], V)
plt.show()
```

```{code-cell} ipython3
derivs_history = np.diff(y_history, axis=0) / np.diff(t_history)[:, None]

for i in range(len(y_history) - 1):
    y = y_history[i]
    yderivs = derivs_history[i]
    slip = y[: init_slip.shape[0]]
    state = y[init_slip.shape[0] :]
    vel = yderivs[: init_slip.shape[0]]
    statederiv = yderivs[init_slip.shape[0] :]
    print(t_history[i] / siay, np.max(np.abs(vel)))
#     plt.plot(np.log10(np.abs(vel)), "k-", linewidth=0.5)
#     plt.show()
```

problem: how do I impose a backslip forcing? ultimately, this is physically unrealistic but I need to do it anyway. 
- look at how the scec seas project does this?
- use some tapering function?
- have a basal panel that just has a linear imposed backslip.
