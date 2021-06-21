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

# Solving a strike slip underneath topography.

In the previous sections, we've assumed full knowledge of a source field and calculated interior potential (displacement for elasticity) and potential gradient (stress for elasticity). However, in many boundary integral applications, we don't have full knowledge of the relevant fields on the boundaries. As a result, we need to solve for those fields. Here, we'll do exactly that. Continuing with the antiplane shear setting, we will model a strike slip fault beneath the surface of the Earth. We'll start with a flat surface and then move on to building a model with some surface topography. 

(TODO, THIS SECTION FEELS WEAK!! MAYBE CUT?)The issue of solving for unknown boundary fields raises all sorts of existence and uniqueness issues in the theory of PDEs. I'm going to gloss over all that and just say that for "simple" PDEs like Poisson's equation or the Navier equations of linear elasticity, as a rule of thumb, if you know half of the relevant fields, you should be able to solve for the other half. For example, in a Poisson problem, if we know potential, we can solve for potential gradient. Or, in elasticity, if we know traction, we can solve for displacement. 

Link to the section in the sa_tdes post.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import gauss_rule, double_layer_matrix, qbx_choose_centers, qbx_expand_matrix, qbx_eval_matrix

%config InlineBackend.figure_format='retina'
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
surf_L = 100
def flat_fnc(q):
    return surf_L*q, 0*q, 0*q, np.ones_like(q), surf_L
```

```{code-cell} ipython3
def slip_fnc(xhat):
    # This must be zero at the endpoints!
    return np.where(
        xhat < -0.99, 
        (1.0 + xhat) * 10,
        np.where(xhat < 0.99, 
                 1.0,
                 (1.0 - xhat) * 10
                )
    )
```

```{code-cell} ipython3
plt.plot(slip_fnc(np.linspace(-1, 1, 100)))
```

```{code-cell} ipython3
qr_fault = gauss_rule(500)
fault = fault_fnc(qr_fault[0])
```

```{code-cell} ipython3
qr_flat = gauss_rule(2000)
flat = flat_fnc(qr_flat[0])
```

```{code-cell} ipython3
qbx_p = 5
qbx_center_x, qbx_center_y, qbx_r = qbx_choose_centers(flat, qr_flat, direction=1)
qbx_expand_flat = qbx_expand_matrix(double_layer_matrix, flat, qr_flat, qbx_center_x, qbx_center_y, qbx_r, qbx_p=qbx_p)
```

```{code-cell} ipython3
qbx_expand_flat.shape
```

```{code-cell} ipython3
qbx_eval_flat = qbx_eval_matrix(flat[0][None,:], flat[1][None,:], qbx_center_x, qbx_center_y, qbx_p=qbx_p)[0]
```

```{code-cell} ipython3
qbx_eval_flat.shape
```

```{code-cell} ipython3
A = np.real(np.sum(qbx_eval_flat[:,None,:,None] * qbx_expand_flat, axis=2))[:,0,:]
```

```{code-cell} ipython3
B = double_layer_matrix(fault, qr_fault, flat[0], flat[1])[:,0,:]
slip = slip_fnc(qr_fault[0])
v = B.dot(slip)
```

```{code-cell} ipython3
s = -0.5
analytical = -s / (2 * np.pi) * (
    np.arctan(flat[0] / (flat[1] + 2.5)) - 
    np.arctan(flat[0] / (flat[1] - 2.5)) - 
    np.arctan(flat[0] / (flat[1] + 0.5)) +
    np.arctan(flat[0] / (flat[1] - 0.5))
)
```

```{code-cell} ipython3
surf_disp = np.linalg.solve(A - 0.5 * np.eye(A.shape[0]), v)
```

```{code-cell} ipython3
plt.figure(figsize=(9,9))
plt.plot(surf_disp)
plt.plot(analytical)
plt.show()
plt.figure(figsize=(9,9))
plt.plot(surf_disp - analytical)
#plt.plot(analytical)
plt.show()
```

```{code-cell} ipython3
nobs = 100
zoomx = [-2.5, 2.5]
zoomy = [-4.5, 0.5]
# zoomx = [-25, 25]
# zoomy = [-45, 5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
levels = np.linspace(-1.0,1.0,11)
cntf = plt.contourf(obsx, obsy, disp_full)#, levels = levels, extend="both")
plt.contour(obsx, obsy, disp_full, colors='k', linestyles='-', linewidths=0.5)#, levels = levels, extend="both")
plt.plot(flat[0], flat[1], 'k-', linewidth=1.5)
plt.plot(fault[0], fault[1], 'k-', linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.show()
```

```{code-cell} ipython3
def symbolic_surface(t, x, y):
    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    ddt_norm = sp.simplify(sp.sqrt(dxdt ** 2 + dydt ** 2))
    dxdt /= ddt_norm
    dydt /= ddt_norm
    
    return x, y, dydt, -dxdt, ddt_norm

def symbolic_eval(t, tvals, exprs):
    out = []
    for e in exprs:
        out.append(sp.lambdify(t, e, "numpy")(tvals))
    return out

sym_t = sp.symbols('t')
sym_x = surf_L*sym_t
sym_y = sp.exp(-sym_t**2 * 50) * sp.Rational(1.5) - sp.Rational(1.5)
sym_topo = symbolic_surface(sym_t, sym_x, sym_y)

qr_topo = gauss_rule(800)
topo = symbolic_eval(sym_t, qr_topo[0], sym_topo)
```

```{code-cell} ipython3
sp.Eq(sp.var('\\vec{n}'), sp.Tuple(sym_topo[2], sym_topo[3]))
```

```{code-cell} ipython3
plt.plot(topo[0], topo[1])
plt.quiver(topo[0], topo[1], topo[2], topo[3], scale = 20)
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
```

## TODO: Introduce interaction_matrix function.
## TODO: Add interior_eval function to part 2.
## Compare with analytical solution

```{code-cell} ipython3
A = self_interaction_matrix(double_layer_matrix, topo, qr_topo)
B = double_layer_matrix(fault, qr_fault, topo[0], topo[1])
slip = slip_fnc(qr_fault[0])
v = B.dot(slip)
surf_disp = np.linalg.solve(A - 0.5 * np.eye(A.shape[0]), v)
```

```{code-cell} ipython3
plt.plot(surf_disp)
plt.show()
```

```{code-cell} ipython3
nobs = 400
zoomx = [-2.5, 2.5]
zoomy = [-4.5, 1.5]
# zoomx = [-25, 25]
# zoomy = [-45, 5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
```

```{code-cell} ipython3
disp_topo = double_layer_matrix(
    surface   = topo,
    obsx      = obsx.flatten(), 
    obsy      = obsy.flatten(),
    quad_rule = qr_topo
).dot(surf_disp).reshape(obsx.shape)
disp_fault = double_layer_matrix(
    surface   = fault,
    obsx      = obsx.flatten(), 
    obsy      = obsy.flatten(),
    quad_rule = qr_fault
).dot(slip).reshape(obsx.shape)
```

```{code-cell} ipython3
plt.figure(figsize = (16,6))
plt.subplot(1,3,1)
levels = np.linspace(-0.1,0.1,21)
cntf = plt.contourf(obsx, obsy, disp_topo, levels = levels, extend="both")
plt.contour(obsx, obsy, disp_topo, colors='k', linestyles='-', linewidths=0.5, levels = levels, extend="both")
plt.plot(topo[0], topo[1], 'k-', linewidth=1.5)
plt.plot(fault[0], fault[1], 'k-', linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(1,3,2)
levels = np.linspace(-0.5,0.5,21)
cntf = plt.contourf(obsx, obsy, disp_fault, levels = levels, extend="both")
plt.contour(obsx, obsy, disp_fault, colors='k', linestyles='-', linewidths=0.5, levels = levels, extend="both")
plt.plot(topo[0], topo[1], 'k-', linewidth=1.5)
plt.plot(fault[0], fault[1], 'k-', linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.subplot(1,3,3)
levels = np.linspace(-0.5,0.5,21)
cntf = plt.contourf(obsx, obsy, disp_topo + disp_fault, levels = levels, extend="both")
plt.contour(obsx, obsy, disp_topo + disp_fault, colors='k', linestyles='-', linewidths=0.5, levels = levels, extend="both")
plt.plot(topo[0], topo[1], 'k-', linewidth=1.5)
plt.plot(fault[0], fault[1], 'k-', linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)

plt.show()
```
