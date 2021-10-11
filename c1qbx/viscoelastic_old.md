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
%load_ext autoreload
%autoreload 2
```

# Viscoelasticity

+++

**This is really cool. There's a simple boundary integral equation for layered linear viscoelasticity. I started working through the math thinking that I'd need to handle a volumetric term, but then I realized that the volumetric term reduces to just a boundary integral for the special case of layered viscoelasticity.**

## A layered viscoelastic PDE
Let's start from the constitutive equation for linear elasticity and the constitutive equation for a Maxwell rheology, both in antiplane strain:

\begin{align}
\textrm{Elastic:  }~~ \vec{\sigma} &= 2\mu\vec{\epsilon}\\
\textrm{Maxwell:  }~~ \dot{\vec{\sigma}} &= 2\mu\dot{\vec{\epsilon}} - \frac{\mu}{\eta}\vec{\sigma}
\end{align}

where $\vec{\sigma} = (\sigma_{xz}, \sigma_{yz})$ is the vector stress in the antiplane setting, $\mu$ is the shear modulus and $\eta$ is the viscosity.

We'll add the definition of strain, again in the antiplane setting:
\begin{equation}
\vec{\epsilon} = \frac{1}{2}\nabla u
\end{equation}

And Newton's law with a body force:
\begin{equation}
\nabla \cdot \vec{\sigma} = 0
\end{equation}

So, that for an elastic rheology, the result simplifies to the Laplace equation (or Poisson equation with zero right hand side). I'll carry the $2\mu$ through despite it dropping out because it will help later to understand the implementation.
\begin{align}
\nabla \cdot (2\mu \vec{\epsilon}) &= 0 \\
2\mu \nabla^2 u &= 0
\end{align}

The result is a bit more complex for a Maxwell rheology. Ultimately though, we can still re-arrange the terms to make the result look like a Poisson equation with a funny looking right-hand-side. Inserting the Maxwell rheology equation into the time derivative of Newton's law:
\begin{align}
\nabla \cdot (2\mu\dot{\vec{\epsilon}} - \frac{\mu}{\eta}\vec{\sigma}) = 0\\
2\mu \nabla^2 \dot{u} = \nabla \cdot (\frac{\mu}{\eta}\vec{\sigma})
\end{align}

Let's explore that right hand side a bit more because the stress divergence component is going to drop out because the divergence of stress is zero.
\begin{align}
\nabla \cdot (\frac{\mu}{\eta}\vec{\sigma}) &= \nabla (\frac{\mu}{\eta}) \cdot \vec{\sigma} + \frac{\mu}{\eta} (\nabla \cdot \vec{\sigma})\\
&= \nabla (\frac{\mu}{\eta}) \cdot \vec{\sigma}
\end{align}

Now consider the figure below, a classic layered Maxwell viscoelastic half space where $\mu$ is constant and $\eta$ varies only as a step function across the elastic/viscoelastic boundary. As a result, the term that includes a viscosity divergence will simplify to:
\begin{equation}
\nabla (\frac{\mu}{\eta}) = (0, -\delta(y=D) \frac{\mu}{\eta_V})
\end{equation}

The negative is because inverse viscosity decreases in the positive y direction.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: []
---
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
%config InlineBackend.figure_format='retina'

x = np.linspace(-1, 1, 100)
plt.plot(3 * x, 0 * x, "k-")
plt.plot(3 * x, -2 + 0 * x, "k-")
plt.plot(0 * x, -0.75 + 0.75 * x, "k-")
plt.text(-2.5, -1, "$\\eta = \\infty$", fontsize=24)
plt.text(-2.5, -3, "$\\eta = \\eta_V$", fontsize=24)
plt.text(1.5, 0.1, "$t = 0$", fontsize=24)
plt.text(1.5, -1.8, "$y = D$", fontsize=24)


plt.gca().add_patch(plt.Circle((0.2, -0.75), 0.10, color="k", fill=False))
plt.gca().add_patch(plt.Circle((0.2, -0.75), 0.05, color="k", fill=True))
plt.gca().add_patch(plt.Circle((-0.2, -0.75), 0.10, color="k", fill=False))
plt.plot([-0.25, -0.15], [-0.8, -0.7], "k-", linewidth=1)
plt.plot([-0.15, -0.25], [-0.8, -0.7], "k-", linewidth=1)

plt.xlim([-3, 3])
plt.ylim([-4, 0.5])
plt.axis("off")
plt.savefig("layered_ve.pdf")
plt.show()
```

So, the final equation we'd like to solve is:
\begin{equation}
2\mu \nabla^2 \frac{\partial u}{\partial t} = -\delta(y=D) \frac{\mu}{\eta_V} \sigma_y 
\end{equation}

In some situations, it can be more convenient to consider the equation in terms of the displacement rather than the velocity:
\begin{equation}
2\mu \nabla^2 u = -\delta(y=D) \frac{\mu}{\eta_V} \int_{0}^{T} \sigma_y dt
\end{equation}

+++

## Viscoelastic boundary integral equation

Now, we'll transform the above equation into integral form. First, remember from the previous section that, when solving for the displacement resulting from a antiplane strike slip fault within a half space, the integral form was:

\begin{equation}
u(\mathbf{p}) = -\int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q} 
\end{equation}
where $\mathbf{p}$ is the observation point, $u$ is the displacement, $s$ is the slip on the fault, $H$ is the free surface, $F$ is the fault surface and $\frac{\partial G}{\partial n_q}$ is the kernel of the double layer potential. 

To extend this to a setting with a body force term, we can add a volume integral term:
\begin{equation}
u(\mathbf{p}) = -\int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q} + \int_{V} G(\mathbf{p},\mathbf{q}) f(\mathbf{q}) d\mathbf{q}
\end{equation}

Now, substituting in the body force from above for $f(\mathbf{q})$, the fascinating thing is that the delta function means that the volume integral will reduce to a surface integral over the surface $B$ that defines the boundary between the viscoelastic region and the elastic region. In our example, $B$ is defined by $y=D$.
\begin{equation}
\int_{V} G(\mathbf{p},\mathbf{q}) f(\mathbf{q}) d\mathbf{q} = -\frac{\mu}{\eta_V} \int_{B} G(\mathbf{p}, \mathbf{q}) \bigg[ \int_0^T \sigma_y(\mathbf{q}) dt \bigg] d\mathbf{q}
\end{equation}

So, the result is a purely boundary integral equation for the behavior of a fault in a viscoelastic and elastic layered space. 

\begin{equation}
u(\mathbf{p}) = -\int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q} - \int_{B} G(\mathbf{p}, \mathbf{q}) \bigg[\frac{\mu}{\eta_V} \int_0^T \sigma_y(\mathbf{q}) dt \bigg] d\mathbf{q}
\end{equation}

Assign $S$ as the stress integral:

\begin{equation}
S(\mathbf{q}) = \frac{\mu}{\eta_V} \int_0^T \sigma_y(\mathbf{q}) dt
\end{equation}

This integral equation results in a fairly simple time stepping algorithm where, given $S^n$.
1. Solve the BIE for $u$. 
2. Compute $\frac{\partial S(\mathbf{q})}{\partial t} = \sigma_y(\mathbf{q})$ for all $\mathbf{q} \in B$.
3. Compute $S^{n+1}$ according to the time integral.
4. Repeat for the next time step.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    double_layer_matrix,
    hypersingular_matrix,
    qbx_setup,
    qbx_matrix
)

%config InlineBackend.figure_format='retina'

fault_half_width = 5000
fault_depth = 5000
surf_L = 250000
visco_depth = 20000
viscosity = 1e18
shear_modulus = 3e9


def fault_fnc(q):
    return [
        0 * q,
        (q - 1) * fault_half_width - fault_depth,
        -np.ones_like(q),
        0 * q,
        np.full_like(q, fault_half_width),
    ]


def free_fnc(q):
    return [surf_L * q, 0 * q, 0 * q, np.ones_like(q), np.full_like(q, surf_L)]
```

```{code-cell} ipython3
qr_free = gauss_rule(2000)
free = free_fnc(qr_free[0])

qbx_p = 5
# Following the previous examples.
# 1) Choose the expansion centers off the boundary.
expansions = qbx_setup(free, qr_free, direction=1)
# 2) Build a matrix that takes an input displacement
qbx_expand_free = qbx_expand_matrix(
    double_layer_matrix, free, qr_free, qbx_center_x, qbx_center_y, qbx_r, qbx_p=qbx_p
)
# 3) Evaluate the QBX expansions for observation points on the boundary.
# The first two arguments here are the x and y coordinates on the boundary.
qbx_eval_free = qbx_eval_matrix(
    free[0][None, :], free[1][None, :], qbx_center_x, qbx_center_y, qbx_p=qbx_p
)[0]
# 4) Multiply the expansion and evaluation matrices to get the full boundary integral matrix.
free_disp_to_free_disp = np.real(
    np.sum(qbx_eval_free[:, None, :, None] * qbx_expand_free, axis=2)
)[:, 0, :]
```

```{code-cell} ipython3
qr_fault = gauss_rule(25)
fault = fault_fnc(qr_fault[0])

fault_slip_to_free_disp = double_layer_matrix(fault, qr_fault, free[0], free[1])[
    :, 0, :
]
slip = np.ones_like(qr_fault[0])
v = fault_slip_to_free_disp.dot(slip)
```

```{code-cell} ipython3
free_disp_solve_mat = np.eye(free_disp_to_free_disp.shape[0]) + free_disp_to_free_disp
free_disp_solve_mat_inv = np.linalg.inv(free_disp_solve_mat)
free_disp = free_disp_solve_mat_inv.dot(v)
```

```{code-cell} ipython3
plt.plot(free[0] / 1000.0, free_disp)
plt.xlabel(r"$x \textrm{(km)}$")
plt.ylabel(r"$u_z \textrm{(m)}$")
plt.title("Displacement")
```

```{code-cell} ipython3
qr_VB = gauss_rule(500)
VB = free_fnc(qr_VB[0])
VB[1] -= visco_depth
stress_integral = np.zeros_like(VB[0])
```

```{code-cell} ipython3
plt.plot(free[0], free[1], "k-")
plt.plot(fault[0], fault[1], "r-")
plt.plot(VB[0], VB[1], "b-")
```

```{code-cell} ipython3
free_disp_to_VB_syz = (
    shear_modulus
    * hypersingular_matrix(surface=free, obsx=VB[0], obsy=VB[1], quad_rule=qr_free)[
        :, 1, :
    ]
)
fault_slip_to_VB_syz = (
    shear_modulus
    * hypersingular_matrix(surface=fault, obsx=VB[0], obsy=VB[1], quad_rule=qr_fault)[
        :, 1, :
    ]
)

syz_free = free_disp_to_VB_syz.dot(free_disp)
syz_fault = fault_slip_to_VB_syz.dot(slip)
syz_full = syz_free + syz_fault
siay = 31556952
```

```{code-cell} ipython3
stress_integral = 20 * siay * (shear_modulus / viscosity) * syz_full
```

```{code-cell} ipython3
plt.plot(VB[0], syz_full)
plt.show()
```

```{code-cell} ipython3
plt.plot(stress_integral)
plt.show()
```

```{code-cell} ipython3
def single_layer_matrix(surface, quad_rule, obsx, obsy):
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    dx = obsx[:, None] - srcx[None, :]
    dy = obsy[:, None] - srcy[None, :]
    r2 = (dx ** 2) + (dy ** 2)
    G = (1.0 / (4 * np.pi)) * np.log(r2)

    return (G * curve_jacobian * quad_rule[1][None, :])[:, None, :]


def adjoint_double_layer_matrix(surface, quad_rule, obsx, obsy):
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    dx = obsx[:, None] - srcx[None, :]
    dy = obsy[:, None] - srcy[None, :]
    r2 = dx ** 2 + dy ** 2

    out = np.empty((obsx.shape[0], 2, surface[0].shape[0]))
    out[:, 0, :] = dx
    out[:, 1, :] = dy

    C = -1.0 / (2 * np.pi * r2)

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (curve_jacobian * quad_rule[1][None, :]))[:, None, :]
```

Solving this problem will be a bit different from past sections in that there's a time component now. I'm going to assume at least some familiarity with standard differential equation time stepping methods. If you haven't run into a Runge-Kutta method before, I'd suggest [finding a good reference](https://faculty.washington.edu/rjl/fdmbook/).

```{code-cell} ipython3
nobs = 100
zoomx = [-15000, 15000]
zoomy = [-31000, -1000]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)

free_disp_to_volume_disp = double_layer_matrix(
    surface=free, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_free
)
fault_slip_to_volume_disp = double_layer_matrix(
    surface=fault, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_fault
)
VB_S_to_volume_disp = (1.0 / shear_modulus) * single_layer_matrix(
    surface=VB, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_VB
)

free_disp_to_volume_stress = shear_modulus * hypersingular_matrix(
    surface=free, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_free
)
fault_slip_to_volume_stress = shear_modulus * hypersingular_matrix(
    surface=fault, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_fault
)
VB_S_to_volume_stress = adjoint_double_layer_matrix(
    surface=VB, obsx=obsx.ravel(), obsy=obsy.ravel(), quad_rule=qr_VB
)


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
        plt.plot(free[0], free[1], "k-", linewidth=1.5)
        plt.plot(fault[0], fault[1], "k-", linewidth=1.5)
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
VB_S_to_free_disp = (1.0 / shear_modulus) * single_layer_matrix(
    surface=VB, obsx=free[0], obsy=free[1], quad_rule=qr_VB
)[:, 0, :]
```

```{code-cell} ipython3
qbx_p = 5
# Following the previous examples.
# 1) Choose the expansion centers off the boundary.
qbx_center_x, qbx_center_y, qbx_r = qbx_choose_centers(VB, qr_VB, direction=1)
# 2) Build a matrix that takes an input displacement
qbx_expand_VB = qbx_expand_matrix(
    adjoint_double_layer_matrix,
    VB,
    qr_VB,
    qbx_center_x,
    qbx_center_y,
    qbx_r,
    qbx_p=qbx_p,
)[:, 1, :, :]
# 3) Evaluate the QBX expansions for observation points on the boundary.
# The first two arguments here are the x and y coordinates on the boundary.
qbx_eval_VB = qbx_eval_matrix(
    VB[0][None, :], VB[1][None, :], qbx_center_x, qbx_center_y, qbx_p=qbx_p
)[0]
# 4) Multiply the expansion and evaluation matrices to get the full boundary integral matrix.
VB_S_to_VB_syz = np.real(np.sum(qbx_eval_VB[:, :, None] * qbx_expand_VB, axis=1))
```

Relaxation time scale!

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


def analytic(t):
    return analytic_to_surface(1.0, 15000, 20000, free[0], t) - analytic_to_surface(
        1.0, 5000, 20000, free[0], t
    )


# for t in [0, 10.0*siay, 20.0*siay,100.0*siay]:
#     plt.plot(free[0] / 1000.0, analytic(t), label=f'{t/siay:.0f} years')
# plt.xlim([-100,100])
# plt.legend()
# plt.show()
```

```{code-cell} ipython3
# The slip does not change so these two integral terms can remain
# outside the time stepping loop.
syz_fault = fault_slip_to_VB_syz.dot(slip)
rhs_slip = fault_slip_to_free_disp.dot(slip)

dt = 0.1 * siay
stress_integral = np.zeros_like(VB[0])
t = 0
disp_history = []
for i in range(301):
    # Step 1) Solve for free surface displacement.
    rhs = rhs_slip + VB_S_to_free_disp.dot(stress_integral)
    free_disp = free_disp_solve_mat_inv.dot(rhs)
    disp_history.append((t, free_disp))

    # Step 2): Calculate viscoelastic boundary stress yz component and then d[S]/dt
    syz_free = free_disp_to_VB_syz.dot(free_disp)
    syz_VB = VB_S_to_VB_syz.dot(stress_integral)
    syz_full = syz_free + syz_fault + syz_VB
    dSdt = (shear_modulus / viscosity) * syz_full

    # Step 3): Update S, simple forward Euler time step.
    stress_integral += dSdt * dt
    t += dt
```

```{code-cell} ipython3
plt.figure(figsize=(7, 7))
X = free[0] / 1000
plt.plot(X, disp_history[0][1], "k-", linewidth=3, label="elastic")
plt.plot(X, analytic(disp_history[0][0]), "k-.", linewidth=3)
plt.plot(X, disp_history[100][1], "m-", label="10 yrs")
plt.plot(X, analytic(disp_history[100][0]), "m-.")
plt.plot(X, disp_history[200][1], "b-", label="20 yrs")
plt.plot(X, analytic(disp_history[200][0]), "b-.")
plt.plot(X, disp_history[300][1], "r-", label="30 yrs")
plt.plot(X, analytic(disp_history[300][0]), "r-.")
plt.plot([], [], " ", label="BIE = solid")
plt.xlim([-100, 100])
plt.xlabel(r"$x ~ \mathrm{(km)}$")
plt.ylabel(r"$u_z ~ \mathrm{(m)}$")
plt.legend()
plt.show()
```

```{code-cell} ipython3

```
