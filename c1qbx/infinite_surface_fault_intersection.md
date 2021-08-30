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

# Fault-surface intersection

In the last section, we constructed a method for solving for surface displacement on a free surface given antiplane slip on a fault beneath the free surface. However, the fault was not allowed to intersect the surface of the Earth. In fact, as we will demonstrate here, if the fault had intersected the surface, the surface displacement solution would have been very wrong! In this section, we will fix this problem!

In addition, the last section compared against an analytical solution that assumes an infinite free surface. Unfortunately, we weren't able to match the analytical solution exactly because it's hard to approximate an infinite free surface. It would be easy to stop there and make a compelling argument that the numerical method is working just fine since the error was quite low away from the tips of the free surface. But that didn't leave me satisfied. I want to fit the arctan solution as exactly as possible!

So, our goals in this section are to:
1. Model a fault that intersects the surface of the Earth.
2. Model an infinite free surface to the best of our ability.

Both of these goals will lead to more general methods that are useful for a wide range of problems. In particular, modeling an infinite free surface will force us to implement some **adaptive meshing** tools that will be very useful for other problems where the spatial scale of interest varies widely through the domain. 

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
    pts_grid,
    self_interaction_matrix,
    interp_matrix
)

fault_top = -0.0
fault_bottom = -1.0
def fault_fnc(n):
    return line(n, (0, fault_top), (0, fault_bottom))
```

```{code-cell} ipython3
from dataclasses import dataclass

# Hierarchy: Boundary -> Segment -> Panel -> Point
# A boundary consists of several segments.
# A segments consists of a single parametrized curve that might be composed of several panels.
# A panel consists of a quadrature rule defined over a subset of a parametrized curve.
@dataclass()
class PanelSurface:
    quad_pts: np.ndarray
    quad_wts: np.ndarray
    pts: np.ndarray
    normals: np.ndarray
    jacobians: np.ndarray
    panel_starts: np.ndarray
    panel_sizes: np.ndarray
        
    @property
    def n_pts(
        self,
    ):
        return self.pts.shape[0]
    
def panelize_symbolic_surface(t, x, y, panel_bounds, qx, qw):
    
```

```{code-cell} ipython3
from common import Surface
import sympy as sp

CR = 0.5
n_segments = 30

# It seems that we need several "small" segments right near the fault-surface intersection!
segments = [(0, CR),(CR, 2*CR),(2*CR, 3*CR)]
for i in range(n_segments - len(segments)):
    segment_start = segments[-1][1]
    segment_end = segment_start + CR * (2 ** i)
    segments.insert(0, (-segment_end, -segment_start))
    segments.append((segment_start, segment_end))
```

```{code-cell} ipython3
qx, qw = gauss_rule(16)
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
    
segments
```

```{code-cell} ipython3
def build_fault():
    segment_start = 0.0
    segments = []
    for i in range(100):
        segment_end = segment_start + 2 * CR * (2 ** i)
        if segment_end > fault_bottom * 0.75:
            segment_end = -fault_bottom
        segments.append((segment_start, segment_end))
        segment_start = segment_end
        if segment_end == -fault_bottom:
            break

    widths = []
    surfs = []
    for i in range(len(segments)):
        qx, qw = gauss_rule(max(128 // (2 ** i), 16))
        width = (segments[i][1] - segments[i][0])
        widths.insert(0, width)
        widths.append(width)

        y = -(segments[i][0] + (t + 1) * 0.5 * width)
        surfs.append(discretize_symbolic_surface(qx, qw, t, 0 * t, y))

    return Surface(
        np.concatenate([s.pts[:,0] for s in surfs]),
        np.concatenate([s.quad_wts * w * 0.5 for w, s in zip(widths, surfs)]) / segments[-1][1],
        np.concatenate([s.pts for s in surfs]),
        np.concatenate([s.normals for s in surfs]),
        np.concatenate([s.jacobians * 2 / w for w,s in zip(widths, surfs)]) * segments[-1][1]
    )
fault = build_fault()
plt.plot(fault.pts[:,0], fault.pts[:,1], '-o')
plt.show()
```

```{code-cell} ipython3
flat.n_pts, fault.n_pts
```

```{code-cell} ipython3
from common import QBXExpansions

r = np.repeat(np.array(widths), flat.n_pts // len(widths)) / 2
orig_expansions = qbx_setup(flat, direction=1, r=r, p=10)
good = np.abs(orig_expansions.pts[:,0]) > 0.30 * CR
expansions = QBXExpansions(
    orig_expansions.pts[good,:],
    orig_expansions.r[good],
    orig_expansions.p
)

plt.plot(expansions.pts[:,0], expansions.pts[:,1], '.')
plt.plot(flat.pts[:,0], flat.pts[:,1], 'k-o', linewidth = 0.5)
plt.plot(fault.pts[:,0], fault.pts[:,1], 'k-o')
plt.axis('equal')
plt.xlim([-CR,CR])
plt.ylim([-3 * CR, CR])
plt.show()
```

```{code-cell} ipython3
A_raw = interior_matrix(double_layer_matrix, flat_interp, flat.pts, expansions)[:,0,:]
A = A_raw.dot(Im_flat)
```

```{code-cell} ipython3
B = -interior_matrix(
    double_layer_matrix,
    fault,
    flat.pts,
    expansions
)[:,0,:]
```

```{code-cell} ipython3
surf_disp = np.linalg.solve(np.eye(A.shape[0]) + A, B.dot(np.ones(fault.n_pts)))

s = 1.0
analytical = (
    -s
    / (2 * np.pi)
    * (
        np.arctan(flat.pts[:,0] / (flat.pts[:,1] - fault_bottom))
        - np.arctan(flat.pts[:,0] / (flat.pts[:,1] + fault_bottom))
        - np.pi * np.sign(flat.pts[:,0])
    )
)

XV = 10.0
#XV = 5 * CR
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(flat.pts[:,0], surf_disp, 'ko')
plt.plot(flat.pts[:,0], analytical, 'bo')
plt.xlabel("$x$")
plt.ylabel("$u_z$")
plt.title("Displacement")
plt.xlim([-XV,XV])
plt.ylim([-0.6, 0.6])

plt.subplot(1, 2, 2)
plt.plot(flat.pts[:,0], np.log10(np.abs(surf_disp - analytical) / np.abs(analytical)))
plt.xlabel("$x$")
plt.ylabel(r"$\log_{10}|u_{\textrm{BIE}} - u_{\textrm{analytic}}|$")
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
