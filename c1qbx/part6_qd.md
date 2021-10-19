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

Goals:
- move this to include the free surface
- make another version that also includes a viscoelastic layer!
- fix up the newton solver and figure out why that was causing trouble
- ideally, understand why swapping Dc for a smaller value helped prevent problems
- optimize rate_state_solver
- make nicer plots
- compare with SEAS results. 


Possible issues:

Fault tips: 
- Identify or specify singularities and then make sure that the QBX and quadrature account for the singularities. This would be helpful for avoiding the need to have the sigmoid transition.
- *Would it be useful to use an interpolation that includes the end points so that I can easily make sure that slip goes to zero at a fault tip?*

Initial conditions:
- Is the creep initial condition somehow wrong?
- Would a slip deficit formulation be easier to get right?
- What about just using far-field plate rate BCs?
- *The pre-stress formulation from the SEAS document!*

Newton solver:
- Is this just broken? It's a pretty shit solver.

problem: how do I impose a backslip forcing? ultimately, this is physically unrealistic but I need to do it anyway. 
- look at how the scec seas project does this?
- use some tapering function?
- have a basal panel that just has a linear imposed backslip.

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
from dataclasses import dataclass
```

```{code-cell} ipython3
@dataclass
class FrictionParams:
    a: np.ndarray
    b: float
    V0: float
    Dc: float
    f0: float
```

```{code-cell} ipython3
siay = 31556952
density = 2670  # rock density (kg/m^3)
cs = np.sqrt(shear_modulus / density)  # Shear wave speed (m/s)
eta = shear_modulus / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
Vp = 1e-9  # Rate of plate motion
sigma_n = 50e6  # Normal stress (Pa)

a0 = 0.01
amax = 0.025
fy = fault.pts[:,1]
H = 30000 #TODO:
H2 = 10000
h = 3000
a = np.where(
    fy > -H,
    np.where(
        fy > -H2,
        np.where(
            fy > -H2 + h,
            amax,
            amax + (a0 - amax) * (fy + (H2 - h)) / -h,
        ),
        a0
    ),
    np.where(
        fy > -(H + h),
        a0 + (amax - a0) * (fy + H) / -h,
        amax
    )
)

fp = FrictionParams(
    a=a,  # direct velocity strengthening effect
    b=0.015,  # state-based velocity weakening effect
    Dc=0.008,  # state evolution length scale (m)
    f0=0.6,  # baseline coefficient of friction
    V0=1e-6,  # when V = V0, f = f0, V is (m/s)
)
```

```{code-cell} ipython3
plt.figure(figsize=(3,5))
plt.plot(fp.a, fy)
plt.plot(np.full(fy.shape[0], fp.b), fy)
plt.xlim([0, 0.03])
plt.show()
```

```{code-cell} ipython3
mesh_L = np.max(fault.panel_length / (fault.n_pts / fault.n_panels))
Lb = shear_modulus * fp.Dc / (sigma_n * fp.b)
hstar = (
    (np.pi * shear_modulus * fp.Dc) /
    (sigma_n * (fp.b - fp.a))
)
mesh_L, Lb, np.min(hstar[hstar > 0])
```

```{code-cell} ipython3
def aging_law(fp, V, state):
    return (fp.b * fp.V0 / fp.Dc) * (np.exp((fp.f0 - state) / fp.b) - (V / fp.V0))
```

```{code-cell} ipython3
def F(fp, V, state):
    return sigma_n * fp.a * np.arcsinh(V / (2 * fp.V0) * np.exp(state / fp.a))


def dFdV(fp, V, state):
    expsa = np.exp(state / fp.a)
    Q = (V * expsa) / (2 * fp.V0)
    return fp.a * expsa * sigma_n / (2 * fp.V0 * np.sqrt(1 + Q * Q))
```

```{code-cell} ipython3
import scipy.optimize
```

```{code-cell} ipython3
from scipy.optimize import fsolve
import copy

fp_amax = copy.deepcopy(fp)
fp_amax.a = amax

init_state_scalar = fsolve(lambda S: aging_law(fp, Vp, S), 0.7)[0]
init_state = np.full(fault.n_pts, init_state_scalar)
tau_amax = F(fp_amax, Vp, init_state_scalar) + eta * Vp
init_traction = np.full(fault.n_pts, tau_amax)
```

```{code-cell} ipython3
def qd_equation(fp, shear_stress, V, state):
    return init_traction + shear_stress - eta * V - F(fp, V, state)

def qd_equation_dV(fp, V, state):
    return -eta - dFdV(fp, V, state)

def rate_state_solve(fp, shear, V_old, state):
    
    def qd(V):
        return qd_equation(fp, shear, V, state)

    def qd_dV(V):
        return qd_equation_dV(fp, V, state)
    
    # TODO: This is really not a good newton solver.
    V = V_old
    max_iter = 150
    for i in range(max_iter):
        f = qd_equation(fp, shear, V, state)
        dfdv = qd_equation_dV(fp, V, state)
        rate = (0.1 if i < 50 else 1.0)
        Vn = V - rate  * (f / dfdv)
        if np.max(np.abs(V - Vn) / Vn) < 1e-10:
            #print('solved after ', i)
            break
        V = Vn
        if i == max_iter - 1:
            raise Exception("Failed to converge.")
    return Vn
```

```{code-cell} ipython3
init_slip = np.zeros(fault.n_pts)
init_conditions = np.concatenate((init_slip, init_state))
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
        return False

    slip_deficit = (t * Vp - slip)# * central_pattern
    stress = fault_slip_to_fault_stress.dot(slip_deficit)
    shear = -np.sum(stress * fault.normals, axis=1)

    try:
        V = rate_state_solve(fp, shear, calc_system_state.V_old, state)
    except RuntimeError:
        return False
    if not np.all(np.isfinite(V)):
        return False
    calc_system_state.V_old = V

    dstatedt = aging_law(fp, V, state)
    
    out = slip, slip_deficit, state, stress, V, dstatedt
    if verbose:
        plot_system_state(out)
        
    return out
calc_system_state.V_old = np.full(fault.n_pts, Vp)

def plot_system_state(SS):
    slip, slip_deficit, state, stress, V, dstatedt = SS
    
    plt.figure(figsize=(15,9))
    plt.subplot(2,3,1)
    plt.title('slip')
    plt.plot(fault.pts[:,1], slip)

    plt.subplot(2,3,2)
    plt.title('state')
    plt.plot(fault.pts[:,1], state)
    
    plt.subplot(2,3,3)
    plt.title('shear')
    plt.plot(fault.pts[:,1], stress[:,0])
    
    plt.subplot(2,3,4)
    plt.title('slip rate')
    plt.plot(fault.pts[:,1], V)
    
    plt.subplot(2,3,6)
    plt.title('dstatedt')
    plt.plot(fault.pts[:,1], dstatedt)
    plt.tight_layout()
    
    plt.show()

def calc_derivatives(t, y):
    #print('trying', t / siay)
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
%load_ext line_profiler
```

```{code-cell} ipython3
def integrate():
    from scipy.integrate import RK23

    calc_system_state.V_old = np.full(fault.n_pts, Vp)

    tol = 1e-5
    rk23 = RK23(calc_derivatives, 0, init_conditions, 1e50, atol=tol, rtol=tol)#, max_step = 0.01 * siay)
    rk23.h_abs = 1.0

    n_steps = 500
    t_history = [0]
    y_history = [init_conditions.copy()]
    for i in range(n_steps):
        #print(i)
        if rk23.step() != None:
            print("TIME STEPPING FAILED")
            break

        if i % 500 == 0:# or i % 10 == 0:
            print(i, rk23.t / siay)
    #     if rk23.t > 1.4 * siay:
            plot_system_state(calc_derivatives.state)
            #break

        t_history.append(rk23.t)
        y_history.append(rk23.y.copy())
```

```{code-cell} ipython3
%lprun -f calc_system_state -f integrate integrate()
```

```{code-cell} ipython3
from scipy.integrate import RK23

calc_system_state.V_old = np.full(fault.n_pts, Vp)

tol = 1e-5
rk23 = RK23(calc_derivatives, 0, init_conditions, 1e50, atol=tol, rtol=tol)#, max_step = 0.01 * siay)
rk23.h_abs = 1.0

n_steps = 2500
t_history = [0]
y_history = [init_conditions.copy()]
for i in range(n_steps):
    #print(i)
    if rk23.step() != None:
        print("TIME STEPPING FAILED")
        break
    
    if i % 500 == 0:# or i % 10 == 0:
        print(i, rk23.t / siay)
#     if rk23.t > 1.4 * siay:
        plot_system_state(calc_derivatives.state)
        #break
    
    t_history.append(rk23.t)
    y_history.append(rk23.y.copy())
```

```{code-cell} ipython3
derivs_history = np.diff(y_history, axis=0) / np.diff(t_history)[:, None]
max_vel = np.max(np.abs(derivs_history), axis=1)
plt.plot(np.array(t_history[1:]) / siay, np.log10(max_vel))
plt.show()
```

```{code-cell} ipython3
last_plt_t = -1000
last_plt_slip = init_slip
for i in range(len(y_history) - 1):
    y = y_history[i]
    t = t_history[i]
    slip = y[: init_slip.shape[0]]
    should_plot = False
    if max_vel[i] > 0.01 and t - last_plt_t > 1:#np.max(np.abs(slip - last_plt_slip)) > 0.25:
        should_plot = True
        color = 'r'
    if t - last_plt_t > 5 * siay:
        should_plot = True
        color = 'b'
    if should_plot:
        plt.plot(slip, fy / 1000.0, color + '-', linewidth = 0.5)
        last_plt_t = t
        last_plt_slip = slip
plt.xlim([0, np.max(last_plt_slip)])
plt.ylim([-35, -5])
plt.ylabel(r'$\textrm{z (km)}$')
plt.xlabel(r'$\textrm{slip (m)}$')
plt.show()
```

```{code-cell} ipython3

```
