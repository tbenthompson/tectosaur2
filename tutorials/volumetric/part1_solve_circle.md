---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# [DRAFT] Dirichlet problem on a circle.

+++

## TODO:

* Based on mms_circle
* Deal with the homogeneous solution.
* Add back in the body force.
* The remaining fundamental issue is the singularity in the volume integral. Set up a way of testing the accuracy of this integral.

+++

## Setup

+++

## Direct to surface eval

+++

For the Poisson equation with Dirichlet boundary conditions:
\begin{split}
\nabla u &= f  ~~ \textrm{in} ~~ \Omega\\
u &= g ~~ \textrm{on} ~~ \partial \Omega
\end{split}
`u_body_force` is the integral:

\begin{equation}
v(x) = \int_{\Omega} G(x,y) f(y) dy
\end{equation}

which satisfies equation 1 but not 2.

Then, compute homogeneous solution with appropriate boundary conditions:

\begin{split}
\nabla u^H &= 0 ~~ \textrm{in} ~~ \Omega \\
u^H &= g - v|_{\partial \Omega}  ~~ \textrm{on} ~~ \partial \Omega
\end{split}

So, first, I need to compute $g - v|_{\partial \Omega}$

```{code-cell} ipython3
:tags: [remove-cell]

from tectosaur2.nb_config import setup

setup()
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from tectosaur2 import integrate_term, refine_surfaces, gauss_rule
from tectosaur2.laplace2d import double_layer
```

```{code-cell} ipython3
nq = 10
qx, qw = gauss_rule(nq)

t = sp.symbols("t")
theta = sp.pi * (t + 1)
circle = refine_surfaces([(t, sp.cos(theta), sp.sin(theta))], (qx,qw))
circle.n_pts, circle.n_panels
```

```{code-cell} ipython3
## This is g
def soln_fnc(x, y):
    return 2 + x + y

bcs = soln_fnc(circle.pts[:, 0], circle.pts[:, 1])
A,report = integrate_term(double_layer, circle.pts, circle, return_report=True)
surf_density = np.linalg.solve(-A[:,0,:,0], bcs)
```

```{code-cell} ipython3
plt.plot(circle.quad_pts, bcs, 'r-')
plt.plot(circle.quad_pts, surf_density, 'b-')
plt.show()
```

```{code-cell} ipython3
nobs = 200
offset = -0.1
zoomx = [-1.0 + offset, 1.0 - offset]
zoomy = [-1.0 + offset, 1.0 - offset]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obsx, obsy = np.meshgrid(xs, ys)
obs2d = np.array([obsx.flatten(), obsy.flatten()]).T.copy()
obs2d_mask = np.sqrt(obs2d[:, 0] ** 2 + obs2d[:, 1] ** 2) <= 1
obs2d_mask_sq = obs2d_mask.reshape(obsx.shape)
obs2d_mask_away = np.sqrt(obs2d[:, 0] ** 2 + obs2d[:, 1] ** 2) <= 0.9
obs2d_mask_away_sq = obs2d_mask_away.reshape(obsx.shape)
correct = soln_fnc(obsx, obsy)
```

```{code-cell} ipython3
%%time
interior_disp_mat = integrate_term(double_layer, obs2d, circle, tol=1e-13)
u_soln = -interior_disp_mat[:, 0, :, 0].dot(surf_density).reshape(obsx.shape)
```

```{code-cell} ipython3
:tags: []

plt.figure(figsize=(12, 4))
for i, to_plot in enumerate([u_soln, u_soln, correct]):
    plt.subplot(1, 3, 1 + i)
    if i == 0:
        levels = np.linspace(np.min(to_plot), np.max(to_plot), 21)
    else:
        cmin = np.min(correct)
        cmax = np.max(correct)
        if cmin == cmax:
            cmin = 0.9 * cmin
            cmax = 1.1 * cmax
        levels = np.linspace(cmin, cmax, 21)

    cntf = plt.contourf(obsx, obsy, to_plot, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        to_plot,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.plot(circle.pts[0], circle.pts[1], "k-", linewidth=1.5)
    plt.colorbar(cntf)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.title(["u\_soln", "u\_soln", "correct"][i])
    plt.axis('scaled')
plt.tight_layout()

plt.figure()
to_plot = np.log10(np.abs(correct - u_soln))
levels = np.linspace(-16, 0, 21)
cntf = plt.contourf(obsx, obsy, to_plot, levels=levels, extend="both")
plt.contour(
    obsx,
    obsy,
    to_plot,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(circle.pts[0], circle.pts[1], "k-", linewidth=1.5)
plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.title("err(u\_soln)")
plt.axis('scaled')
plt.show()
```
