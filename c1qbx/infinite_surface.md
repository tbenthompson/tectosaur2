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

# A strike slip fault underneath topography.

## Mathematical background
Note that much of the discussion here parallels the discussion in [the section on solving topographic problems with TDEs](../tdes/sa_tdes.md).

Last time we solved for the displacement and stress given slip on an infinitely long strike slip fault. This time, we'll make the problem more interesting (and a bit harder!) by adding in a free surface. First, we'll replicate classical solutions for the surface displacement on a half-space given slip on a fault. Then, we'll add in a topographic free surface and have some fun!

Note that mathematicians may be uncomfortable with the precision of some of the statements here. If you feel that way, I'd recommend taking a look at an [integral equations textbook like Kress](https://www.springer.com/us/book/9781461495925).

Let's start with the integral equation for displacement resulting from slip on a crack/fault and displacement and traction on another arbitrary surface:

\begin{equation}
u(\mathbf{p}) = \int_{H} G(\mathbf{p}, \mathbf{q}) t(\mathbf{q}) d\mathbf{q} - \int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q}
\end{equation}

where $H$ is the surface, $F$ is the fault, $t$ is traction, $u$ is displacement and $s$ is slip. Immediately, we can see that, for a free surface, the first term is zero because the traction is zero, so we are left with:

\begin{equation}
u(\mathbf{p}) = -\int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q}
\end{equation}

So, this equation says that we can calculate displacement anywhere if we know displacement on $H$ and slip on $F$. But, we don't know the displacement on $H$! So, step one is going to be solving for the unknown displacement on the surface $H$. 

To solve for surface displacement, intuitively, we need to reduce from two unknowns to one unknown in this equation. Currently we have both the unknown displacement on $H$ and the unknown displacement at an arbitrary point in the volume, $u(\mathbf{p})$. To remedy this situation, we will simply choose to only enforce the integral equation at boundary points $\mathbf{p} \in H$. This can be proven sufficient for the equation to hold everywhere and is the fundamental basis of boundary value problems in potential theory and elastostatics. Because $u$ is now only evaluated on the boundary, this equation is solvable for $u(\mathbf{p})$ given $s$. 

```{note}
**Uniqueness**: In informal terms, the uniqueness theorems state that for a linear elastic boundary value problem, if we know displacement and traction on the boundary of the domain, then we can calculate displacement and traction everywhere in the volume. 
```

Re-arranging to put the unknown displacement on the left hand side: 
\begin{equation}
u(\mathbf{p}) + \int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} = -\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q}  ~~~~~~ \forall \mathbf{p} \in H
\end{equation}

Note that the integrals are now being evaluated in their singular limits. Previously, we have only evaluated nearly singular integrals that arise when the observation point is very close to the source surface. Fortunately, QBX works just as well in the limit to the boundary and as a result, nothing about our integration procedures will need to change. 

Once discretized using our QBX-based quadrature tools from the previous sections, this will look like:
\begin{equation}
\mathbf{Iu} + \mathbf{Au} = \mathbf{Bs}
\end{equation}

where $\mathbf{A}$ and $\mathbf{B}$ are matrices representing the action of the integral terms on surface displacement and fault slip respectively. $\mathbf{I}$ is the identity matrix.

+++

## Solving for surface displacement

+++

For the first section, we're going to reproduce analytical solutions for the displacement due to slip on a buried fault under a half space. That solution {cite:p}`segallEarthquakeVolcanoDeformation2010` is:

\begin{equation}
u_z = \frac{-s}{2\pi}\bigg[\tan^{-1}(\frac{x}{y + d_1}) - \tan^{-1}(\frac{x}{y - d_1}) - \tan^{-1}(\frac{x}{y + d_2}) + \tan^{-1}(\frac{x}{y - d_2}) \bigg]
\end{equation}

To start with the numerical implementation, we'll define two surfaces. The `fault_fnc` function will define a fault extending from 0.5 units depth to 1.5 units depth and the `flat_fnc` function will define a free surface extending from -25 to 25 units along the x axis. 

Note that while we're trying to approximate a free surface, we won't attempt to actually have an infinite surface here. It will turn out that 25 units in both directions is enough to get a very good match with the analytical solution. If we truly wanted to model an infinite surface numerically, I could imagine achieving that via an adaptive quadrature method combined with something like [Gauss-Laguerre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature). I would enjoy trying that out some day! However, the goal here isn't to actually model an infinite domain. The Earth is not infinite! Instead, the goal is simply to demonstrate that our numerical methods work on an analytically tractable problem before moving on to problems that are not analytically intractable.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import (
    discretize_symbolic_surface,
    gauss_rule,
    double_layer_matrix,
    qbx_setup,
    interior_matrix,
    build_interpolator,
    interp_surface,
    line,
    pts_grid
)

fault_top = -0.0
fault_bottom = -2.5
def fault_fnc(n):
    return line(n, (0, fault_top), (0, fault_bottom))


surf_L = 10
def flat_fnc(n):
    return line(n, (surf_L, 0), (-surf_L, 0))
```

Conceptually, the $\mathbf{A}$ matrix is the interaction of the flat free surface with itself. Let's build that matrix first using the QBX tools from the last two sections. As mentioned, we won't need to do anything fundamentally different for evaluating these integrals in the limit of the observation point being on the boundary. In fact, because the observation surface and the source surface are the same, we can avoid a lot of the complexity in the `qbx_interior_eval_matrix` and `interior_matrix` functions. We'll create a new `self_interaction_matrix` function:

```{code-cell} ipython3
import_and_display_fnc('common', 'self_interaction_matrix')
import_and_display_fnc('common', 'interp_matrix')
```

```{code-cell} ipython3
from common import Surface

n_segments = 8
segment_start = 0.0
segments = []
for i in range(n_segments - len(segments)):
    segment_end = segment_start + 0.01 * (2.0 ** i)
    segments.append((segment_start, segment_end))
    segment_start = segment_end
    if segment_start > 2.0:
        break

starting_length = segments[-1][1] - segments[-1][0]
for i in range(n_segments - len(segments)):
    segment_end = segment_start + starting_length * (1.3 ** i)
    segments.append((segment_start, segment_end))
    segment_start = segment_end

    
raw_quad_pts = []
raw_quad_wts = []
qx, qw = gauss_rule(16)
for i in range(len(segments)):
    width = segments[i][1] - segments[i][0]
    qx_transformed = segments[i][0] + ((qx + 1) * 0.5) * width
    qw_transformed = qw / 2.0 * width
    raw_quad_pts.append(qx_transformed)
    raw_quad_wts.append(qw_transformed)

half_quad_pts = np.concatenate(raw_quad_pts) / segments[-1][1]
half_quad_wts = np.concatenate(raw_quad_wts) / segments[-1][1]

quad_pts = np.concatenate((-half_quad_pts[::-1], half_quad_pts))
quad_wts = np.concatenate((half_quad_wts[::-1], half_quad_wts))

plt.plot(quad_pts * segments[-1][1])
plt.show()
```

```{code-cell} ipython3
segments
```

```{code-cell} ipython3
import sympy as sp
t = sp.var('t')
widths = []
surfs = []
for i in range(n_segments):
    width = (segments[i][1] - segments[i][0])
    widths.insert(0, width)
    widths.append(width)
    
    x = segments[i][1] - (t + 1) * 0.5 * width
    surfs.insert(0, discretize_symbolic_surface(qx, qw, t, x, 0 * t))
    
    x = -segments[i][0] - (t + 1) * 0.5 * width
    surfs.append(discretize_symbolic_surface(qx, qw, t, x, 0 * t))

interp_surfs = []
interp_mats = []
for s in surfs:
    kappa = 6
    si = interp_surface(s, *gauss_rule(kappa * s.n_pts))
    Im = interp_matrix(build_interpolator(s.quad_pts), si.quad_pts)
    interp_surfs.append(si)
    interp_mats.append(Im)

flat = Surface(
    np.concatenate([s.pts[:,0] for s in surfs]),
    np.concatenate([s.quad_wts * w * 0.5 for w, s in zip(widths, surfs)]) / segments[-1][1],
    np.concatenate([s.pts for s in surfs]),
    np.concatenate([s.normals for s in surfs]),
    np.concatenate([s.jacobians * 2 / w for w,s in zip(widths, surfs)]) * segments[-1][1]
)

flat_interp = Surface(
    np.concatenate([s.pts[:,0] for s in interp_surfs]),
    np.concatenate([s.quad_wts * w * 0.5 for w, s in zip(widths, interp_surfs)]) / segments[-1][1],
    np.concatenate([s.pts for s in interp_surfs]),
    np.concatenate([s.normals for s in interp_surfs]),
    np.concatenate([s.jacobians * 2 / w for w,s in zip(widths, interp_surfs)]) * segments[-1][1]
)

np.sum(flat.quad_wts), np.sum(flat_interp.quad_wts)

plt.plot(flat.pts[:,0])
plt.show()

Im_flat = np.zeros((flat_interp.n_pts, flat.n_pts))

for i in range(len(surfs)):
    Im_flat[i * interp_surfs[0].n_pts:(i+1)*interp_surfs[0].n_pts,i*surfs[0].n_pts:(i+1)*surfs[0].n_pts] = interp_mats[i]
```

```{code-cell} ipython3
from common import QBXExpansions
fault_kappa = 6
fault = fault_fnc(400)

r = np.repeat(np.array(widths), flat.n_pts // len(widths)) / 2

flat_expansions = qbx_setup(flat, direction=1, r=r)
# corner = np.concatenate((np.abs(fault_expansions.pts[:,1]) < 0.1, np.abs(flat_expansions.pts[:,0]) < 0.01))
# corner = np.full(exp_pts.shape[0], False)
exp_r = np.concatenate((fault_expansions.r, flat_expansions.r))
expansions = QBXExpansions(
    exp_pts[~corner],
    exp_r[~corner],
    5
)
print(expansions.N)


plt.plot(exp_pts[corner,0], exp_pts[corner,1], '.')
plt.plot(flat.pts[:,0], flat.pts[:,1], 'k-')
plt.plot(fault.pts[:,0], fault.pts[:,1], 'k-')
plt.axis('equal')
plt.xlim([-0.15,0.15])
plt.ylim([-0.2,0.1])
plt.show()
```

```{code-cell} ipython3
A_raw = interior_matrix(double_layer_matrix, flat_interp, flat.pts, expansions)[:,0,:]
A = A_raw.dot(Im_flat)
```

```{code-cell} ipython3
B = -interior_matrix(
    double_layer_matrix,
    fault_interp,
    flat.pts,
    expansions
)[:,0,:].dot(Im_fault)
```

```{code-cell} ipython3
lhs = np.eye(A.shape[0]) + A

slip = np.ones(fault.n_pts)
rhs = B.dot(slip)

surf_disp = np.linalg.solve(lhs, rhs)

s = 1.0
if fault_top == 0.0:
    analytical = (
        -s
        / (2 * np.pi)
        * (
            np.arctan(flat.pts[:,0] / (flat.pts[:,1] - fault_bottom))
            - np.arctan(flat.pts[:,0] / (flat.pts[:,1] + fault_bottom))
            - np.pi * np.sign(flat.pts[:,0])
        )
    )
else:
    analytical = (
        -s
        / (2 * np.pi)
        * (
            np.arctan(flat.pts[:,0] / (flat.pts[:,1] - fault_bottom))
            - np.arctan(flat.pts[:,0] / (flat.pts[:,1] + fault_bottom))
            - np.arctan(flat.pts[:,0] / (flat.pts[:,1] - fault_top))
            + np.arctan(flat.pts[:,0] / (flat.pts[:,1] + fault_top))
        )
    )

XV = 1.0
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(flat.pts[:,0], surf_disp, 'k-o')
plt.plot(flat.pts[:,0], analytical, 'b-')
plt.xlabel("$x$")
plt.ylabel("$u_z$")
plt.title("Displacement")
plt.xlim([-XV,XV])
plt.ylim([-0.6, 0.6])

plt.subplot(1, 2, 2)
plt.plot(flat.pts[:,0], np.log10(np.abs(surf_disp - analytical)))
plt.xlabel("$x$")
plt.ylabel("$\log_{10}|u_{\textrm{BIE}} - u_{\textrm{analytic}}|$")
plt.title("Error")
plt.tight_layout()
plt.xlim([-XV, XV])
plt.show()
```

## Evaluating interior displacement

```{code-cell} ipython3
:tags: []

nobs = 100
zoomx = [-2.5, 2.5]
zoomy = [-4.5, 0.5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
```

Now that we have the surface displacement, we can return to the integral form of the interior displacement: 

\begin{equation}
u(\mathbf{p}) = \int_{H} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) u(\mathbf{q}) d\mathbf{q} +\int_{F} \frac{\partial G}{\partial n_q}(\mathbf{p}, \mathbf{q}) s(\mathbf{q}) d\mathbf{q}
\end{equation}

and, using QBX via the `interior_eval` function from the last section, directly calculate the two integrals on the right hand side. The integral over $H$ will be `disp_flat` and the integral over $F$ will be `disp_fault`.

```{code-cell} ipython3
:tags: []

fault_expansions = qbx_setup(fault, mult=5.0)

disp_flat = interior_matrix(
    double_layer_matrix,
    flat_interp,
    obs_pts,
    flat_expansions
).dot(Im).dot(surf_disp).reshape((nobs, nobs))

disp_fault = interior_matrix(
    double_layer_matrix,
    fault,
    obs_pts,
    fault_expansions
).dot(slip).reshape((nobs, nobs))

disp_full = disp_flat + disp_fault
```

```{code-cell} ipython3
:tags: []

levels = np.linspace(-0.5, 0.5, 11)
cntf = plt.contourf(obsx, obsy, disp_full, levels=levels, extend="both", cmap="bwr")
plt.contour(
    obsx,
    obsy,
    disp_full,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(flat.pts[:,0], flat.pts[:,1], "k-", linewidth=1.5)
plt.plot(fault.pts[:,0], fault.pts[:,1], "k-", linewidth=1.5)
plt.fill(np.array([np.min(zoomx), np.max(zoomx), np.max(zoomx), np.min(zoomx)]), np.array([0, 0, np.max(zoomy), np.max(zoomy)]), "w", zorder=100)

plt.colorbar(cntf)
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.xlabel("$x \; \mathrm{(m)}$")
plt.ylabel("$y \; \mathrm{(m)}$")
plt.show()
```

## Adding topography

+++

In the rest of this section, we'll replicate the calculation above except for a free surface with topography. The calculations will be identical except for the construction of the surface itself. 

First, I'll construct a mesh with a Gaussian shaped hill above the fault.

```{code-cell} ipython3
:tags: []



sp.Eq(sp.var("x,y"), sp.Tuple(sym_x, sym_y))
```

I'll write a generic function here that can accept a parameterization of a curve and return the tuple of symbolic `(x, y, normal_x, normal_y, norm)`.

```{code-cell} ipython3
:tags: []

def symbolic_surface(t, x, y):
    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    ddt_norm = sp.simplify(sp.sqrt(dxdt ** 2 + dydt ** 2))
    dxdt /= ddt_norm
    dydt /= ddt_norm
    return x, y, -dydt, dxdt, ddt_norm


sym_topo = symbolic_surface(sym_t, sym_x, sym_y)
```

Here are the symbolic normal vector and norm of the transformation.

+++

I'll plot a quick diagram of the mesh and the surface normals.

```{code-cell} ipython3
:tags: []

import sympy as sp

t = sp.symbols("t")
x = -surf_L * sym_t
y = sp.exp(-(sym_t ** 2) * 200) * sp.Rational(2.0) - sp.Rational(2.0)

qr_topo = gauss_rule(800)
topo = discretize_symbolic_surface(*qr_topo, t, x, y)
```

```{code-cell} ipython3
:tags: []

plt.plot(topo.pts[:,0], topo.pts[:,1])
plt.quiver(topo.pts[:,0], topo.pts[:,1], topo.normals[:,0], topo.normals[:,1], scale=20)
plt.plot(fault.pts[:,0], fault.pts[:,1], 'r-')
plt.xlim([-2.5, 2.5])
plt.ylim([-3.5, 1.5])
```

And let's follow the same procedure as before, we'll solve for the surface displacement. The shape doesn't look that different from before, but the peak surface displacements near the fault are a bit higher.

```{code-cell} ipython3
:tags: []

topo_interp = interp_surface(topo, *gauss_rule(2 * topo.n_pts))
topo_expansions = qbx_setup(topo, mult=5.0, direction=1, p=6)

A_raw = self_interaction_matrix(double_layer_matrix, topo, topo_interp, topo_expansions)[0][:,0,:]
Im = interp_matrix(build_interpolator(topo.quad_pts), topo_interp.quad_pts)
A = A_raw.dot(Im)

B = -double_layer_matrix(fault, topo.pts)[:, 0, :]
rhs = B.dot(slip)
lhs = np.eye(A.shape[0]) + A
surf_disp_topo = np.linalg.solve(lhs, rhs)

plt.plot(topo.pts[:,0], surf_disp_topo, "k-", label="Topography")
plt.plot(flat.pts[:,0], surf_disp, "b-", label="Flat")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u_z$")
plt.show()
```

## Interior displacement under topography

```{code-cell} ipython3
:tags: []

nobs = 200
zoomx = [-2.5, 2.5]
zoomy = [-3.5, 1.5]
xs = np.linspace(*zoomx, nobs)
ys = np.linspace(*zoomy, nobs)
obs_pts = pts_grid(xs, ys)
```

And finally, we'll plot the displacement in the volume underneath the topography. Just like the in the flat case. To show the effect of both integral terms, I've separated out the topography integral term and the fault integral term in the plots.

```{code-cell} ipython3
:tags: []

disp_topo = interior_matrix(
    double_layer_matrix,
    topo_interp,
    obs_pts,
    topo_expansions
).dot(Im).dot(surf_disp_topo).reshape((nobs, nobs))

disp_fault = interior_matrix(
    double_layer_matrix,
    fault,
    obs_pts,
    fault_expansions
).dot(slip).reshape((nobs, nobs))

disp_full = disp_topo + disp_fault
```

```{code-cell} ipython3
:tags: []

obsx = obs_pts[:,0].reshape((nobs, nobs))
obsy = obs_pts[:,1].reshape((nobs, nobs))

levels = np.linspace(-0.5, 0.5, 21)
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
cntf = plt.contourf(obsx, obsy, disp_topo, levels=levels, extend="both", cmap="bwr")
plt.contour(
    obsx,
    obsy,
    disp_topo,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(topo.pts[:,0], topo.pts[:,1], "k-", linewidth=2.5)
plt.plot(fault.pts[:,0], fault.pts[:,1], "k-", linewidth=2.5)
fill_poly = np.append(topo.pts[:,0], np.array([np.min(topo.pts[:,0]), np.max(topo.pts[:,0])])), np.append(topo.pts[:,1], np.array([zoomy[1], zoomy[1]]))
plt.fill(*fill_poly, "w", zorder=100)
plt.colorbar(cntf, label="$u \; (m)$")
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.xticks(np.array([zoomx[0], 0, zoomx[1]]))
plt.yticks(np.array([zoomy[0], -1, zoomy[1]]))
plt.xlabel("$x ~ \mathrm{(m)}$")
plt.ylabel("$y ~ \mathrm{(m)}$")
plt.title("Surface displacement integral term")

plt.subplot(1, 3, 2)
cntf = plt.contourf(obsx, obsy, disp_fault, levels=levels, extend="both", cmap="bwr")
plt.contour(
    obsx,
    obsy,
    disp_fault,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(topo.pts[:,0], topo.pts[:,1], "k-", linewidth=2.5)
plt.plot(fault.pts[:,0], fault.pts[:,1], "k-", linewidth=2.5)
plt.fill(*fill_poly, "w", zorder=100)
plt.colorbar(cntf, label="$u \; (m)$")
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.xticks(np.array([zoomx[0], 0, zoomx[1]]))
plt.yticks(np.array([zoomy[0], -1, zoomy[1]]))
plt.xlabel("$x ~ \mathrm{(m)}$")
plt.ylabel("$y ~ \mathrm{(m)}$")
plt.title("Fault slip integral term")

plt.subplot(1, 3, 3)
cntf = plt.contourf(obsx, obsy, disp_topo + disp_fault, levels=levels, extend="both", cmap="bwr")
plt.contour(
    obsx,
    obsy,
    disp_topo + disp_fault,
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=levels,
    extend="both",
)
plt.plot(topo.pts[:,0], topo.pts[:,1], "k-", linewidth=2.5)
plt.plot(fault.pts[:,0], fault.pts[:,1], "k-", linewidth=2.5)
plt.fill(*fill_poly, "w", zorder=100)
plt.colorbar(cntf, label="$u \; (m)$")
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.xticks(np.array([zoomx[0], 0, zoomx[1]]))
plt.yticks(np.array([zoomy[0], -1, zoomy[1]]))
plt.xlabel("$x ~ \mathrm{(m)}$")
plt.ylabel("$y ~ \mathrm{(m)}$")
plt.title("Full displacement solution")

plt.tight_layout()
plt.show()
```
