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

```{code-cell} ipython3
:tags: [remove-cell]

from tectosaur2.nb_config import setup

setup()
```

```{code-cell} ipython3
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tectosaur2 import gauss_rule, refine_surfaces, integrate_term, tensor_dot
from tectosaur2.elastic2d import elastic_t, elastic_h
from tectosaur2.mesh import panelize_symbolic_surface, concat_meshes
from tectosaur2.rate_state import MaterialProps, qd_equation, solve_friction, aging_law
```

```{code-cell} ipython3
surf_half_L = 50000
fault_length = 40000
max_panel_length = 800
n_decollement = 200
mu = shear_modulus = 3.2e10
nu = 0.25

qx, qw = gauss_rule(4)
sp_t = sp.var("t")

edges = np.linspace(-1, 1, n_decollement+1)
panel_bounds = np.stack((edges[:-1],edges[1:]), axis=1)
angle_rad = sp.pi / 10
sp_x = (sp_t + 1) / 2 * sp.cos(angle_rad) * fault_length
sp_y = -(sp_t + 1) / 2 * sp.sin(angle_rad) * fault_length
# decollement = panelize_symbolic_surface(
#     sp_t, sp_x, sp_y,
#     panel_bounds,
#     qx, qw
# )
```

```{code-cell} ipython3
splay_fault_length = 20000
splay_offset = 10000
splay_angle_rad = sp.pi / 3
splay_sp_t = sp.var("t_s")
splay_sp_x = splay_offset + (splay_sp_t + 1) / 2 * sp.cos(splay_angle_rad)
splay_sp_y = -(splay_sp_t + 1) / 2 * sp.sin(splay_angle_rad)
```

```{code-cell} ipython3
intersection = sp.solve([splay_sp_x - sp_x,splay_sp_y - sp_y], sp_t, splay_sp_t)
splay_length = 0.5 * float(intersection[splay_sp_t])
splay_sp_x = splay_offset + (splay_sp_t + 1) / 2 * sp.cos(splay_angle_rad) * splay_length
splay_sp_y = -(splay_sp_t + 1) / 2 * sp.sin(splay_angle_rad) * splay_length
n_splay = int(splay_length / (fault_length / n_decollement))
```

```{code-cell} ipython3
splay_edges = np.linspace(-1, 1, n_splay+1)
splay_panel_bounds = np.stack((splay_edges[:-1],splay_edges[1:]), axis=1)
# splay = panelize_symbolic_surface(
#     splay_sp_t, splay_sp_x, splay_sp_y,
#     splay_panel_bounds,
#     qx, qw
# )
# faults = concat_meshes((decollement, splay))
```

```{code-cell} ipython3
def sp_line(start, end):
    t01 = (sp_t + 1) * 0.5
    xv = start[0] + t01 * (end[0] - start[0])
    yv = start[1] + t01 * (end[1] - start[1])
    return sp_t, xv, yv

b_depth = float(sp_y.subs(sp_t, 1))
b_cut = float(sp_x.subs(sp_t, 1))
splay_intersection_x = float(splay_sp_x.subs(splay_sp_t, 1))
splay_intersection_y = float(splay_sp_y.subs(splay_sp_t, 1))

decollement1, decollement2, splay, bottom1, bottom2, left, right, top1, top2, top3 = refine_surfaces(
    [
        sp_line([0, 0], [splay_intersection_x, splay_intersection_y]),
        sp_line([splay_intersection_x, splay_intersection_y], [b_cut, b_depth]),
        (splay_sp_t, splay_sp_x, splay_sp_y),

        sp_line([-surf_half_L, b_depth], [b_cut, b_depth]),
        sp_line([b_cut, b_depth], [surf_half_L, b_depth]),
        sp_line([-surf_half_L, 0], [-surf_half_L, b_depth]),
        sp_line([surf_half_L, b_depth], [surf_half_L, 0]),

        sp_line([surf_half_L, 0], [splay_offset, 0]),
        sp_line([splay_offset, 0], [0, 0]),
        sp_line([0, 0], [-surf_half_L, 0]),
    ],
    (qx, qw),
    control_points = [
        (0, 0, 0.5 * fault_length, fault_length/n_decollement),
        (b_cut, b_depth, 0.5 * fault_length, fault_length/n_decollement),
        (splay_offset, 0, 0.5 * splay_fault_length, splay_fault_length/n_splay),
        (splay_intersection_x, splay_intersection_y, splay_fault_length, splay_fault_length/n_splay),

        (0, 0, surf_half_L, 50000),
        (surf_half_L, b_depth, 50000, 5000),
        (-surf_half_L, b_depth, 50000, 5000),
        (surf_half_L, 0, 50000, 5000),
        (-surf_half_L, 0, 50000, 5000),
    ]
)
decollement = concat_meshes((decollement1, decollement2))
top = concat_meshes((top1, top2, top3))
bottom = concat_meshes((bottom1, bottom2))
free = concat_meshes((top, bottom))
free = concat_meshes((top, bottom))
fixed = concat_meshes((left, right))
faults = concat_meshes((decollement, splay))
```

```{code-cell} ipython3
print(
    f"The free surface mesh has {free.n_panels} panels with a total of {free.n_pts} points."
)
print(
    f"The farfield surface mesh has {fixed.n_panels} panels with a total of {fixed.n_pts} points."
)
print(
    f"The splay mesh has {splay.n_panels} panels with a total of {splay.n_pts} points."
)
print(
    f"The decollement mesh has {decollement.n_panels} panels with a total of {decollement.n_pts} points."
)
```

```{code-cell} ipython3
plt.plot(free.pts[:,0]/1000, free.pts[:,1]/1000, 'k-')
plt.plot(fixed.pts[:,0]/1000, fixed.pts[:,1]/1000, 'k-')
plt.plot(decollement.pts[:,0]/1000, decollement.pts[:,1]/1000, 'r-')
plt.plot(splay.pts[:,0]/1000, splay.pts[:,1]/1000, 'r-')
plt.xlabel(r'$x ~ \mathrm{(km)}$')
plt.ylabel(r'$y ~ \mathrm{(km)}$')
plt.axis('scaled')
plt.xlim([-60, 60])
plt.ylim([-30, 5])
plt.show()
```

And, to start off the integration, we'll construct the operators necessary for solving for free surface displacement from fault slip.

```{code-cell} ipython3
singularities = np.array(
    [
        [-surf_half_L, 0],
        [surf_half_L, 0],
        [-surf_half_L, b_depth],
        [surf_half_L, b_depth],
        [float(sp_x.subs(sp_t,-1)), float(sp_y.subs(sp_t,-1))],
        [float(sp_x.subs(sp_t,1)), float(sp_y.subs(sp_t,1))],
        [float(splay_sp_x.subs(splay_sp_t,-1)), float(splay_sp_y.subs(splay_sp_t,-1))],
        [float(splay_sp_x.subs(splay_sp_t,1)), float(splay_sp_y.subs(splay_sp_t,1))],
    ]
)
```

```{code-cell} ipython3
def to_traction(obs_surf, mat):
    nx = obs_surf.normals[:, 0]
    ny = obs_surf.normals[:, 1]
    normal_mult = np.transpose(np.array([[nx, 0 * nx, ny], [0 * nx, ny, nx]]), (2, 0, 1))
    return np.sum(
        mat.reshape((obs_surf.n_pts, 3, -1, 2))[:, None, :, :, :]
        * normal_mult[:, :, :, None, None],
        axis=2,
    ).reshape((obs_surf.n_pts, 2, -1, 2))

def flatten_tensor_mat(mat):
    return mat.reshape((mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3]))
```

```{code-cell} ipython3
b_cut, b_depth
```

```{code-cell} ipython3
(free_disp_to_fixed_disp, fixed_disp_to_fixed_disp, fault_slip_to_fixed_disp), report = integrate_term(
    elastic_t(nu), fixed.pts, free, fixed, faults, singularities=singularities, safety_mode=True, return_report=True
)
```

```{code-cell} ipython3
(free_disp_to_free_stress, fixed_disp_to_free_stress, fault_slip_to_free_stress), report = integrate_term(
    elastic_h(nu), free.pts, free, fixed, faults, singularities=singularities, safety_mode=True, return_report=True
)
```

```{code-cell} ipython3
from tectosaur2.debug import plot_centers
plot_centers(report, [-50, 50], [-10, 2])
plot_centers(report, [b_cut-50, b_cut+50], [b_depth-2, b_depth+50])
```

```{code-cell} ipython3
free_disp_to_free_disp, fixed_disp_to_free_disp, fault_slip_to_free_disp = integrate_term(
    elastic_t(nu), top.pts, free, fixed, faults, singularities=singularities, safety_mode=True
)
```

```{code-cell} ipython3
(free_disp_to_fault_stress, fixed_disp_to_fault_stress, fault_slip_to_fault_stress), report = integrate_term(
    elastic_h(nu), faults.pts, free, fixed, faults, singularities=singularities, safety_mode=True, return_report=True
)
```

```{code-cell} ipython3
from tectosaur2.debug import plot_centers
plot_centers(report, [-50, 50], [-10, 2])
plt.show()
```

```{code-cell} ipython3
free_disp_to_free_traction = to_traction(free, free_disp_to_free_stress)
fixed_disp_to_free_traction = to_traction(free, fixed_disp_to_free_stress)
fault_slip_to_free_traction = to_traction(free, fault_slip_to_free_stress)
```

```{code-cell} ipython3
def to_matrix(components):
    return np.concatenate([np.concatenate(c, axis = 1) for c in components], axis = 0)
```

```{code-cell} ipython3
fault_nx = faults.normals[:, 0]
fault_ny = faults.normals[:, 1]
slip_field = np.stack((-fault_ny, fault_nx), axis=1)
rhs_mat = to_matrix([[fault_slip_to_free_traction], [fault_slip_to_fixed_disp]]).reshape((-1, faults.n_pts * 2))
rhs_slip = rhs_mat.dot(slip_field.ravel())

left_u = np.empty((left.n_pts, 2))
left_u[:, 0] = np.full(left.n_pts, 0)
left_u[:, 1] = np.full(left.n_pts, 0)
right_u = np.empty((right.n_pts, 2))
right_u[:, 0] = np.full(right.n_pts, -1.0)
right_u[:, 1] = np.full(right.n_pts, 0)
rhs_plate = np.concatenate((np.zeros((free.n_pts, 2)), left_u, right_u))
```

```{code-cell} ipython3
Ifixed = np.eye(fixed.n_pts * 2)
Ifree = np.eye(free.n_pts * 2)
lhs = to_matrix(
    [
        [
            -flatten_tensor_mat(free_disp_to_free_traction),
            -flatten_tensor_mat(fixed_disp_to_free_traction),
        ],
        [
            -flatten_tensor_mat(free_disp_to_fixed_disp),
            -flatten_tensor_mat(fixed_disp_to_fixed_disp),
        ],
    ]
)
```

```{code-cell} ipython3
lhs_inv = np.linalg.inv(lhs)
```

```{code-cell} ipython3
density_slip = np.linalg.solve(lhs, rhs_slip.ravel())
density_plate = np.linalg.solve(lhs, rhs_plate.ravel())
plt.plot(density_slip[::2])
plt.plot(density_slip[1::2])
plt.show()
plt.plot(density_plate[::2])
plt.plot(density_plate[1::2])
plt.show()
```

```{code-cell} ipython3
density_to_free_disp = np.concatenate((flatten_tensor_mat(free_disp_to_free_disp), flatten_tensor_mat(fixed_disp_to_free_disp)), axis=1)
surf_disp_slip = (-density_to_free_disp.dot(density_slip) - tensor_dot(fault_slip_to_free_disp, slip_field).ravel()).reshape((-1,2))
surf_disp_plate = (-density_to_free_disp.dot(density_plate)).reshape((-1,2))
surf_disp_combined = surf_disp_plate + surf_disp_slip
plt.figure(figsize=(7,7))
plt.plot(top.pts[:, 0], surf_disp_slip[:,0], 'k-', label='fault slip x')
plt.plot(top.pts[:, 0], surf_disp_slip[:,1], 'b-', label='fault slip y')
plt.plot(top.pts[:, 0], surf_disp_plate[:,0], 'k-.', label='plate motion x')
plt.plot(top.pts[:, 0], surf_disp_plate[:,1], 'b-.', label='plate motion y')
plt.legend()
plt.show()
plt.plot(top.pts[:, 0], surf_disp_combined[:,0], 'k-')
plt.plot(top.pts[:, 0], surf_disp_combined[:,1], 'b-')
plt.show()
```

```{code-cell} ipython3
density_to_fault_stress = np.concatenate((flatten_tensor_mat(free_disp_to_fault_stress), flatten_tensor_mat(fixed_disp_to_fault_stress)), axis=1)
total_fault_slip_to_fault_stress = shear_modulus*(-density_to_fault_stress.dot(lhs_inv.dot(rhs_mat)) - fault_slip_to_fault_stress.reshape((-1, faults.n_pts * 2)))
fault_stress_slip = total_fault_slip_to_fault_stress.dot(slip_field.ravel()).reshape((-1,3))
fault_stress_plate = shear_modulus*(-density_to_fault_stress.dot(lhs_inv.dot(rhs_plate.ravel()))).reshape((-1,3))
fault_stress_combined = fault_stress_slip + fault_stress_plate
```

```{code-cell} ipython3
# plt.plot(faults.pts[:, 0], fault_stress_slip[:,0], 'k-')
# plt.plot(faults.pts[:, 0], fault_stress_slip[:,1], 'b-')
# plt.plot(faults.pts[:, 0], fault_stress_slip[:,2], 'r-')
plt.plot(faults.pts[:, 0], fault_stress_plate[:,0], 'k-.')
plt.plot(faults.pts[:, 0], fault_stress_plate[:,1], 'b-.')
plt.plot(faults.pts[:, 0], fault_stress_plate[:,2], 'r-.')
plt.show()
plt.plot(decollement.pts[:, 0], fault_stress_combined[:decollement.n_pts,0], 'k-')
plt.plot(decollement.pts[:, 0], fault_stress_combined[:decollement.n_pts,1], 'b-')
plt.plot(decollement.pts[:, 0], fault_stress_combined[:decollement.n_pts,2], 'r-')
plt.ylim([-10e6, 10e6])
plt.show()
plt.plot(splay.pts[:, 1], fault_stress_combined[decollement.n_pts:,0], 'k-')
plt.plot(splay.pts[:, 1], fault_stress_combined[decollement.n_pts:,1], 'b-')
plt.plot(splay.pts[:, 1], fault_stress_combined[decollement.n_pts:,2], 'r-')
plt.ylim([-10e6, 10e6])
plt.show()
```

```{code-cell} ipython3
nx = faults.normals[:, 0]
ny = faults.normals[:, 1]
normal_mult = np.transpose(np.array([[nx, 0 * nx, ny], [0 * nx, ny, nx]]), (2, 0, 1))
total_fault_slip_to_fault_traction = np.sum(
    total_fault_slip_to_fault_stress.reshape((-1, 3, faults.n_pts, 2))[:, None, :, :, :]
    * normal_mult[:, :, :, None, None],
    axis=2,
).reshape((-1, 2 * faults.n_pts))
```

```{code-cell} ipython3
fault_traction_plate = np.sum(
    fault_stress_plate[:, None, :] * normal_mult, axis=2
).ravel()
```

```{code-cell} ipython3
stress2slip = np.linalg.inv(total_fault_slip_to_fault_traction)
```

```{code-cell} ipython3
plate_slip = stress2slip.dot(fault_traction_plate).reshape((-1, 2))
print(plate_slip.shape)
plt.plot(decollement.pts[:, 0], plate_slip[:decollement.n_pts,0], 'k-')
plt.plot(decollement.pts[:, 0], plate_slip[:decollement.n_pts,1], 'b-')
# plt.ylim([-10e6, 10e6])
plt.show()
plt.plot(splay.pts[:, 1], plate_slip[decollement.n_pts:,0], 'k-')
plt.plot(splay.pts[:, 1], plate_slip[decollement.n_pts:,1], 'b-')
# plt.ylim([-10e6, 10e6])
plt.show()
```

## Rate and state friction

+++

Okay, now that we've constructed the necessary boundary integral operators, we get to move on to describing the frictional behavior on the fault.

#### TODO: Explain!!

```{code-cell} ipython3
siay = 31556952  # seconds in a year
density = 2670  # rock density (kg/m^3)
cs = np.sqrt(shear_modulus / density)  # Shear wave speed (m/s)
Vp = 1e-9  # Rate of plate motion
sigma_n0 = 50e6  # Normal stress (Pa)

# parameters describing "a", the coefficient of the direct velocity strengthening effect
a0 = 0.01
amax = 0.025
H = 15000
h = 3000
fx = faults.pts[:, 0]
fy = faults.pts[:, 1]
fd = -np.sqrt(fx ** 2 + fy ** 2)
# a = np.where(
#     fd > -H, a0, np.where(fd > -(H + h), a0 + (amax - a0) * (fd + H) / -h, amax)
# )
a = a0 + (amax-a0) / (1 + np.exp(-(-fd - 18000) / 1000))

mp = MaterialProps(a=a, b=0.015, Dc=0.008, f0=0.6, V0=1e-6, eta=shear_modulus / (2 * cs))
```

```{code-cell} ipython3
plt.figure(figsize=(3, 5))
plt.plot(mp.a, fd/1000, 'ko', label='a')
plt.plot(np.full(fy.shape[0], mp.b), fd/1000, label='b')
plt.xlim([0, 0.03])
plt.ylabel('depth')
plt.legend()
plt.show()
```

```{code-cell} ipython3
mesh_L = np.max(np.abs(np.diff(fd[:decollement.n_pts])))
Lb = shear_modulus * mp.Dc / (sigma_n0 * mp.b)
hstar = (np.pi * shear_modulus * mp.Dc) / (sigma_n0 * (mp.b - mp.a))
mesh_L, Lb, np.min(hstar[hstar > 0])
```

```{code-cell} ipython3
mesh_L = np.max(np.abs(np.diff(fd[decollement.n_pts:])))
Lb = shear_modulus * mp.Dc / (sigma_n0 * mp.b)
hstar = (np.pi * shear_modulus * mp.Dc) / (sigma_n0 * (mp.b - mp.a))
mesh_L, Lb, np.min(hstar[hstar > 0])
```

## Quasidynamic earthquake cycle derivatives

```{code-cell} ipython3
from scipy.optimize import fsolve
import copy

init_state_scalar = fsolve(lambda S: aging_law(mp, Vp, S), 0.7)[0]

init_tau = np.full(faults.n_pts, 0.5 * sigma_n0)
init_sigma = np.full(faults.n_pts, sigma_n0)
init_state = np.full(faults.n_pts, init_state_scalar)
init_slip = np.zeros(faults.n_pts)
init_conditions = np.concatenate((init_slip, init_state))
```

```{code-cell} ipython3
class SystemState:

    V_old = np.full(faults.n_pts, Vp)
    state = None

    def calc(self, t, y, verbose=False):
        # Separate the slip_deficit and state sub components of the
        # time integration state.
        slip = y[: init_slip.shape[0]]
        state = y[init_slip.shape[0] :]

        # If the state values are bad, then the adaptive integrator probably
        # took a bad step.
        if np.any((state < 0) | (state > 2.0)):
            print("bad state")
            return False

        # The big three lines solving for quasistatic shear stress, slip rate
        # and state evolution
        traction = -(fault_traction_plate * Vp * t).reshape((-1,2))
        slip_vector = -np.stack((slip * -ny, slip * nx), axis=1).ravel()
        traction += total_fault_slip_to_fault_traction.dot(slip_vector).reshape((-1, 2))
        delta_sigma_qs = np.sum(traction * np.stack((nx, ny), axis=1), axis=1)
        delta_tau_qs = -np.sum(traction * np.stack((-ny, nx), axis=1), axis=1)
        tau_qs = init_tau + delta_tau_qs
        sigma_qs = init_sigma + delta_sigma_qs

        # Need to properly handle the
        # tau_sign = np.sign(tau_qs)
        # tau_abs = np.abs(tau_qs)
        V = solve_friction(mp, sigma_qs, tau_qs, self.V_old, state)
        if not V[2]:
            print("convergence failed")
            return False

        V = V[0]
        if not np.all(np.isfinite(V)):
            print("infinite V")
            return False
        dstatedt = aging_law(mp, V, state)
        self.V_old = V

        out = (
            slip,
            state,
            delta_sigma_qs,
            sigma_qs,
            delta_tau_qs,
            tau_qs,
            V,
            dstatedt
        )
        self.data = out
        return self.data
```

```{code-cell} ipython3
def plot_system_state(t, SS, xlim=None):
    """This is just a helper function that creates some rough plots of the
    current state to help with debugging"""
    (
        slip,
        state,
        delta_sigma_qs,
        sigma_qs,
        delta_tau_qs,
        tau_qs,
        V,
        dstatedt,
    ) = SS


    plt.figure(figsize=(20, 8))
    plt.suptitle(f"t={t/siay}")
    for j in range(2):
        idx_start = 0 if j == 0 else decollement.n_pts
        idx_end = decollement.n_pts if j == 0 else faults.n_pts
        origin = np.array([[0,0]]) if j == 0 else np.array([[splay_offset,0]])
        fd = -np.linalg.norm(faults.pts[idx_start:idx_end] - origin , axis=1)
        plt.subplot(2, 5, 5*j+1)
        plt.title("slip")
        plt.plot(fd, slip[idx_start:idx_end])
        plt.xlim(xlim)

        plt.subplot(2, 5, 5*j+2)
        plt.title("V")
        plt.plot(fd, V[idx_start:idx_end])
        plt.xlim(xlim)

        plt.subplot(2, 5, 5*j+3)
        plt.title(r"$\sigma_{qs}$")
        plt.plot(fd, sigma_qs[idx_start:idx_end])
        plt.xlim(xlim)

        plt.subplot(2, 5, 5*j+4)
        plt.title(r"$\tau_{qs}$")
        plt.plot(fd, tau_qs[idx_start:idx_end], 'k-o')
        plt.xlim(xlim)

        plt.subplot(2, 5, 5*j+5)
        plt.title("state")
        plt.plot(fd, state[idx_start:idx_end])
        plt.xlim(xlim)
    plt.show()
```

```{code-cell} ipython3
def calc_derivatives(state, t, y):
    """
    This helper function calculates the system state and then extracts the
    relevant derivatives that the integrator needs. It also intentionally
    returns infinite derivatives when the `y` vector provided by the integrator
    is invalid.
    """
    if not np.all(np.isfinite(y)):
        return np.inf * y
    state_vecs = state.calc(t, y)
    if not state_vecs:
        return np.inf * y
    derivatives = np.concatenate((state_vecs[-2], state_vecs[-1]))
    return derivatives
```

## Integrating through time

```{code-cell} ipython3
%%time
from scipy.integrate import RK23, RK45

# We use a 5th order adaptive Runge Kutta method and pass the derivative function to it
# the relative tolerance will be 1e-11 to make sure that even
state = SystemState()
derivs = lambda t, y: calc_derivatives(state, t, y)
integrator = RK45
atol = Vp * 1e-6
rtol = 1e-11
rk = integrator(derivs, 0, init_conditions, 1e50, atol=atol, rtol=rtol)

# Set the initial time step to one day.
rk.h_abs = 60 * 60 * 24

# Integrate for 1000 years.
max_T = 3000 * siay

n_steps = 50000
t_history = [0]
y_history = [init_conditions.copy()]
for i in range(n_steps):
    # Take a time step and store the result
    if rk.step() != None:
        raise Exception("TIME STEPPING FAILED")
    t_history.append(rk.t)
    y_history.append(rk.y.copy())

    # Print the time every 5000 steps
    if i % 1000 == 0:
        print(f"step={i}, time={rk.t / siay} yrs, step={(rk.t - t_history[-2]) / siay}")
    if i % 1000 == 0:
        print(f"step={i}, time={rk.t / siay} yrs, step={(rk.t - t_history[-2]) / siay}")
        plot_system_state(rk.t, state.calc(rk.t, rk.y))#, xlim=[-21000, -14000])

    if rk.t > max_T:
        break

y_history = np.array(y_history)
t_history = np.array(t_history)
```

## Plotting the results

```{code-cell} ipython3
derivs_history = np.diff(y_history, axis=0) / np.diff(t_history)[:, None]
max_vel = np.max(np.abs(derivs_history), axis=1)
plt.plot(t_history[1:] / siay, np.log10(max_vel))
plt.xlabel('$t ~~ \mathrm{(yrs)}$')
plt.ylabel('$\log_{10}(V)$')
plt.show()
```

```{code-cell} ipython3
plt.figure(figsize=(10, 4))
last_plt_t = -1000
last_plt_slip = init_slip_deficit
event_times = []
for i in range(len(y_history) - 1):
    y = y_history[i]
    t = t_history[i]
    slip_deficit = y[: init_slip_deficit.shape[0]]
    should_plot = False

    # Plot a red line every three second if the slip rate is over 0.1 mm/s.
    if (
        max_vel[i] >= 0.0001 and t - last_plt_t > 3
    ):
        if len(event_times) == 0 or t - event_times[-1] > siay:
            event_times.append(t)
        should_plot = True
        color = "r"

    # Plot a blue line every fifteen years during the interseismic period
    if t - last_plt_t > 15 * siay:
        should_plot = True
        color = "b"

    if should_plot:
        # Convert from slip deficit to slip:
        slip = -slip_deficit + Vp * t
        plt.plot(slip, fd / 1000.0, color + "-", linewidth=0.5)
        last_plt_t = t
        last_plt_slip = slip
plt.xlim([0, np.max(last_plt_slip)])
plt.ylim([-40, 0])
plt.ylabel(r"$\textrm{z (km)}$")
plt.xlabel(r"$\textrm{slip (m)}$")
plt.tight_layout()
plt.savefig("halfspace.png", dpi=300)
plt.show()
```
