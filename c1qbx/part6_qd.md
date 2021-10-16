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

Based on: https://strike.scec.org/cvws/seas/download/SEAS_BP1_QD.pdf

Possible issues:

Fault tips: 
- Identify or specify singularities and then make sure that the QBX and quadrature account for the singularities. This would be helpful for avoiding the need to have the sigmoid transition.
- *Would it be useful to use an interpolation that includes the end points so that I can easily make sure that slip goes to zero at a fault tip?*

Initial conditions:
- Is the creep initial condition somehow wrong?
- Would a slip deficit formulation be easier to get right?
- What about just using far-field plate rate BCs?

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
fault_bottom = 40000
shear_modulus = 3.2e10

qx, qw = gauss_rule(6)
t = sp.var("t")

control_points = [
    (0, -fault_bottom / 2, fault_bottom / 2, 1200),
]
fault = stage1_refine(
    [
        (t, t * 0, fault_bottom * (t + 1) * -0.5),  # fault    
    ],
    (qx, qw),
    control_points=control_points,
)[0]

fault_expansions = qbx_panel_setup([fault], directions=[0], p=10)[0]
```

```{code-cell} ipython3
fault_slip_to_fault_stress = shear_modulus * qbx_matrix2(
    hypersingular_matrix, fault, fault.pts, fault_expansions
)
```

```{code-cell} ipython3
def sigmoid(x0, W):
    return 1.0 / (1 + np.exp((fault.pts[:, 1] - x0) / W))

central_pattern = sigmoid(-3000, 200) - sigmoid(-17000, 200)

slip = central_pattern
print("slip")
plt.plot(fault.pts[:, 1], slip)
plt.show()

differential_stress = fault_slip_to_fault_stress.dot(slip)

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
    a=0.020,  # direct velocity strengthening effect
    b=0.025,  # state-based velocity weakening effect
    Dc=0.05,  # state evolution length scale (m)
    f0=0.6,  # baseline coefficient of friction
    V0=1e-6,  # when V = V0, f = f0, V is (m/s)
)
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
slip_deficit_fullspace = np.linalg.inv(fault_slip_to_fault_stress[:, 0, :]).dot(
    init_traction
)
```

```{code-cell} ipython3
plt.plot(fault.pts[:, 1], slip_deficit_fullspace, "r-")
plt.show()
```

```{code-cell} ipython3
stress = fault_slip_to_fault_stress.dot(slip_deficit_fullspace)
plt.plot(fault.pts[:, 1], init_traction, "r-")
plt.plot(fault.pts[:, 1], stress[:, 0], "k-")
plt.show()
```

```{code-cell} ipython3
init_slip_deficit = -slip_deficit_fullspace
init_conditions = np.concatenate((init_slip_deficit, init_state))
```

```{code-cell} ipython3
def calc_system_state(t, y, verbose=False):
    if verbose:
        print(t)
        print(t)
        print(t)
        print(t)
        print(t)
    
    slip_deficit = y[: init_slip_deficit.shape[0]]
    state = y[init_slip_deficit.shape[0] :]
    
    
    if np.any(state < 0) or np.any(state > 1.2):
        return False

    stress = -fault_slip_to_fault_stress.dot(slip_deficit)
    shear = np.sum(stress * fault.normals, axis=1)

    def qd(V):
        return qd_equation(fp, shear, V, state)

    def qd_dV(V):
        return qd_equation_dV(fp, V, state)

    try:
        V = scipy.optimize.newton(qd, calc_system_state.V_old, fprime=qd_dV)
    except RuntimeError:
        return False
    calc_system_state.V_old = V
    
    if not np.all(np.isfinite(V)):
        return False

    dstatedt = aging_law(fp, V, state)
    
    slip_deficit_rate = Vp - V
    out = slip_deficit, state, stress, V, slip_deficit_rate, dstatedt
    if verbose:
        plot_system_state(out)
        
    return out
calc_system_state.V_old = np.full(fault.n_pts, presumed_init_velocity)

def plot_system_state(SS):
    slip_deficit, state, stress, V, slip_deficit_rate, dstatedt = SS
    
    plt.figure(figsize=(9,9))
    plt.subplot(2,3,1)
    plt.title('slip deficit')
    plt.plot(fault.pts[:,1], slip_deficit)

    plt.subplot(2,3,2)
    plt.title('state')
    plt.plot(fault.pts[:,1], state)
    
    plt.subplot(2,3,3)
    plt.title('shear')
    plt.plot(fault.pts[:,1], stress[:,0])
    
    plt.subplot(2,3,4)
    plt.title('slip rate')
    plt.plot(fault.pts[:,1], V)
    plt.tight_layout()
    
    
    plt.subplot(2,3,5)
    plt.title('slip deficit rate')
    plt.plot(fault.pts[:,1], slip_deficit_rate)
    plt.tight_layout()
    
    
    plt.subplot(2,3,6)
    plt.title('dstatedt')
    plt.plot(fault.pts[:,1], dstatedt)
    #plt.tight_layout()
    
    plt.show()

def calc_derivatives(t, y):
    print('trying', t / siay)
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
from scipy.integrate import RK23

calc_system_state.V_old = np.full(fault.n_pts, presumed_init_velocity)

tol = 1e-5
rk23 = RK23(calc_derivatives, 0, init_conditions, 1e50, atol=tol, rtol=tol, max_step = 0.01 * siay)
rk23.h_abs = 1.0

n_steps = 1000
t_history = [0]
y_history = [init_conditions.copy()]
for i in range(n_steps):
    print(i)
    if rk23.step() != None:
        print("TIME STEPPING FAILED")
        break
    
#     if rk23.t > 1.4 * siay:
#         plot_system_state(calc_derivatives.state)
    
    t_history.append(rk23.t)
    y_history.append(rk23.y.copy())
```

```{code-cell} ipython3
derivs_history = np.diff(y_history, axis=0) / np.diff(t_history)[:, None]

for i in range(len(y_history) - 1):
    y = y_history[i]
    yderivs = derivs_history[i]
    slip = y[: init_slip_deficit.shape[0]]
    state = y[init_slip_deficit.shape[0] :]
    vel = yderivs[: init_slip_deficit.shape[0]]
    statederiv = yderivs[init_slip_deficit.shape[0] :]
    print(t_history[i] / siay, np.max(np.abs(vel)))
#     plt.plot(np.log10(np.abs(vel)), "k-", linewidth=0.5)
#     plt.show()
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

problem: how do I impose a backslip forcing? ultimately, this is physically unrealistic but I need to do it anyway. 
- look at how the scec seas project does this?
- use some tapering function?
- have a basal panel that just has a linear imposed backslip.
