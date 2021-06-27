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

# [DRAFT] Viscoelasticity

## TODO-list

* Figure out how to handle the divergence of stress term. To calculate that directly would require a higher order boundary integral operator. That should be feasible... 

## Summary

A starting point:

* Only works when the viscoelastic region is a decent distance away from the free surface. 
* For a problem like this where we are solving for surface displacement, it should be okay for the viscoelastic region to overlap the fault. However, if we solve for fault stress like in a QD model, then some more work will be necessary to handle when the observation point is inside the body force region.

+++

## Derivation

Let's start from the constitutive equation for elasticity and the constitutive equation for a Maxwell rheology. We'll do this in full 3D complexity and then reduce to plane-strain later.

$$\textrm{Elastic:  }~~ \sigma_{ij} = 2\mu\epsilon_{ij} + \lambda\epsilon_{kk}\delta_{ij}$$
$$\textrm{Maxwell:  }~~ \dot{\sigma}_{ij} + \frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij}) = 2\mu\dot{\epsilon}_{ij} + \lambda\dot{\epsilon}_{kk}\delta_{ij}$$

And Newton's law:

$$ \sum_j \frac{\partial \sigma_{ij}}{\partial x_j} = f_i $$

So, for an elastic rheology:

$$ \sum_j \frac{\partial [2\mu\epsilon_{ij} + \lambda\epsilon_{kk}\delta_{ij}]}{\partial x_j} = f_i $$

For a viscoelastic rheology, remembering the time derivative: $\dot{\sigma}_{ij}$:

$$ \sum_j \frac{\partial \big[2\mu\dot{\epsilon}_{ij} + \lambda\dot{\epsilon}_{kk}\delta_{ij} - [\frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij})]\big]}{\partial x_j} = \dot{f}_i $$

Rearranging the viscoelastic equation: 

$$ \sum_j \frac{\partial \big[2\mu\dot{\epsilon}_{ij} + \lambda\dot{\epsilon}_{kk}\delta_{ij}\big]}{\partial x_j} = \dot{f}_i  + \sum_j \frac{\partial[\frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij})]}{\partial x_j} $$

Rewriting the right hand side as $F_i$, we see that this is an elastic problem with a funny body force:

$$ \textrm{"Elastic-like": }~~ \sum_j \frac{\partial \big[2\mu\dot{\epsilon}_{ij} + \lambda\dot{\epsilon}_{kk}\delta_{ij}\big]}{\partial x_j} = F_i$$
$$ \textrm{"Viscoelastic body force": }~~ F_i = \dot{f}_i + \sum_j \frac{\partial[\frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij})]}{\partial x_j} $$

It's important to notice that in the main "Elastic-like" equation, we are operating in terms of velocity/strain-rate/stressing-rate because the whole equation has had a time derivative applied.

+++

## Integrating in time
So, how do we solve this? I see two broad ways depending on where we would prefer to do the time integration. 

1. Compute velocities and stressing rates at each time step and then integrate those to obtain displacement and stress. 
2. Integrate the "Elastic-like" equation analytically (trivial, since it just involves removing some dots), so that we can solve directly for displacement and stress then do a time integration to compute the current viscoelastic body force. 

I think method #2 is likely to be more accurate and stable simply because it involves lower order derivatives. Instead of integrating stressing rate in time to get stress, we are integrating stress in time to get the total viscoelastic body force. To do this, let's define:

$$ V_i = \int_{0}^{t} \sum_j \frac{\partial[\frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij})]}{\partial x_j} dt $$

and rearrange this to involve a time step:

$$ V^{n+1}_i = V^{n}_i + \int_{t^n}^{t^{n+1}} \sum_j \frac{\partial[\frac{\mu}{\eta}(\sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij})]}{\partial x_j} dt$$

Returning to the main law of motion:

$$\sum_j \frac{\partial \sigma_{ij}^{n}}{\partial x_j} = f^{n}_i + V^{n}_i$$

So, what's the final solution procedure?

1. Given a viscoelastic "body force", $V_i^n$, use standard techniques to solve for stress, $\sigma_{ij}^n$, and displacement $u_i^n$.
2. Now, integrate the time step equation above to get $V_i^{n+1}$.
3. Repeat.

+++

## Simplify for antiplane strain

For antiplane shear, the stress is dramatically simplified, such that we really only care about a vector stress like: 
\begin{equation}
\sigma_z = (\sigma_{xz}, \sigma_{yz})
\end{equation}

and the viscoelastic body force term simplfies to:
 
\begin{equation}
V_z = \int_{0}^{t} \frac{\mu}{\eta}(\frac{\partial \sigma_{xy}}{\partial y} + \frac{\partial \sigma_{xz}}{\partial z}) dt
\end{equation}

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import gauss_rule, double_layer_matrix, qbx_choose_centers, qbx_expand_matrix, qbx_eval_matrix
```

```{code-cell} ipython3
import sympy as sp
```

```{code-cell} ipython3
fault_depth = 0.5
def fault_fnc(q):
    return 0*q, q - 1 - fault_depth, -np.ones_like(q), 0*q, 1.0
```

```{code-cell} ipython3
surf_L = 10
def flat_fnc(q):
    return surf_L*q, 0*q, 0*q, np.ones_like(q), surf_L
```

```{code-cell} ipython3
def slip_fnc(xhat):
    # This must be zero at the endpoints!
    return np.where(
        xhat < -0.9, 
        (1.0 + xhat) * 10,
        np.where(xhat < 0.9, 
                 1.0,
                 (1.0 - xhat) * 10
                )
    )
```

```{code-cell} ipython3
plt.plot(slip_fnc(np.linspace(-1, 1, 100)))
```

```{code-cell} ipython3
qr_fault = gauss_rule(50)
fault = fault_fnc(qr_fault[0])
```

```{code-cell} ipython3
qr_flat = gauss_rule(2500)
flat = flat_fnc(qr_flat[0])
```

```{code-cell} ipython3
qbx_p = 5
qbx_center_x, qbx_center_y, qbx_r = qbx_choose_centers(flat, qr_flat)
qbx_expand_flat = qbx_expand_matrix(double_layer_matrix, flat, qr_flat, qbx_center_x, qbx_center_y, qbx_r, qbx_p=qbx_p)
qbx_eval_flat = qbx_eval_matrix(flat[0][None,:], flat[1][None,:], qbx_center_x, qbx_center_y, qbx_p=qbx_p)[0]
A = np.real(np.sum(qbx_eval_flat[:,:,None] * qbx_expand_flat, axis=1))
```

```{code-cell} ipython3
B = double_layer_matrix(fault, qr_fault, flat[0], flat[1])
slip = slip_fnc(qr_fault[0])
v = B.dot(slip)
```

```{code-cell} ipython3
surf_disp = np.linalg.solve(A - 0.5 * np.eye(A.shape[0]), v)
```

```{code-cell} ipython3
plt.plot(surf_disp)
plt.show()
```

```{code-cell} ipython3
nobs = 100
zoomx = [-2.5, 2.5]
zoomy = [-5.1, -0.1]
# zoomx = [-25, 25]
# zoomy = [-45, 5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)

disp_flat = double_layer_matrix(
    surface   = flat,
    obsx      = obsx.flatten(), 
    obsy      = obsy.flatten(),
    quad_rule = qr_flat
).dot(surf_disp).reshape(obsx.shape)
disp_fault = double_layer_matrix(
    surface   = fault,
    obsx      = obsx.flatten(), 
    obsy      = obsy.flatten(),
    quad_rule = qr_fault
).dot(slip).reshape(obsx.shape)
disp_full = disp_flat + disp_fault

levels = np.linspace(-0.5,0.5,21)
cntf = plt.contourf(obsx, obsy, disp_full, levels = levels, extend="both")
plt.contour(obsx, obsy, disp_full, colors='k', linestyles='-', linewidths=0.5, levels = levels, extend="both")
plt.plot(flat[0], flat[1], 'k-', linewidth=1.5)
plt.plot(fault[0], fault[1], 'k-', linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.show()
```

```{code-cell} ipython3

```
