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

# Fault-surface intersection and infinite free surfaces.

**This is a very rough draft**

In the last section, we constructed a method for solving for surface displacement on a free surface given antiplane slip on a fault beneath the free surface. However, the fault was not allowed to intersect the surface of the Earth. In fact, as we will demonstrate here, if the fault had intersected the surface, the surface displacement solution would have been very wrong! In this section, we will fix this problem!

In addition, the last section compared against an analytical solution that assumes an infinite free surface. Unfortunately, we weren't able to match the analytical solution exactly because it's hard to approximate an infinite free surface. It would be easy to stop there and make a compelling argument that the numerical method is working just fine since the error was quite low away from the tips of the free surface. But that didn't leave me satisfied. I want to fit the arctan solution as exactly as possible!

So, our goals in this section are to:
1. Model a fault that intersects the surface of the Earth.
2. Model an infinite free surface to the best of our ability.

Both of these goals will lead to more general methods that are useful for a wide range of problems. In particular, modeling an infinite free surface will force us to implement some **adaptive meshing** tools that will be very useful for other problems where the spatial scale of interest varies widely through the domain.

```{code-cell} ipython3
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    qbx_matrix,
    symbolic_eval,
    build_interp_matrix,
    build_interpolator,
    qbx_setup,
    double_layer_matrix,
)
```

## Panels

The key to solving both these problem will be to separate the surface into many smaller components that we will call "panels". Up until now, every surface we've dealt with has been parameterized with a single curve and then discretized using a single quadrature rule, either Gaussian or trapezoidal. Why does separating this single surface into subcomponents help?
1. A Gaussian quadrature rule implicitly assumes that the function being integrated is smooth. However, at a fault-surface intersection, the surface displacement will actually be discontinuous! 
2. The resolution needed near the fault will be high. The resolution needed 100 fault lengths away from the fault trace is very low. So, we will need to have higher point density in some places than others. We will achieve this by refining some panels to be much smaller than others. 

[draft note] Explain the code below more.

+++

What is necessary for refinement?
1. The surfaces!
2. The source functions.
3. Ideally, the solution function. 

The refinement necessary for solution and quadrature is distinct. 
It would be nice to simply use separate meshes for the two.
The solution mesh would be:
- coarser than the quadrature mesh.
- required to not vary by more than a factor of two from panel to panel
 
The quadrature grid would:
- have values calculated from an interpolation operation.
- have no restrictions on the variation in panel length.
- allow for either h or p adaptivity. does h adaptivity ever make sense since we're just going to be interpolating the existing solution? Yes, because there may be fine-scale features!

So, the overall plan is:
1. take input surfaces and refine them into a proto-solution mesh.
   1. refine based on radius of curvature.
   2. refine based on locally specified length scale.
   3. refine based on other nearby surfaces.
2. interpolate boundary condition functions on to the proto-solution mesh producing both a final solution mesh and the coefficients of the interpolated function. Often this step will be unnecessary because the boundary data is known only at certain nodes or is simple and constant. 
3. form expansion centers and then construct the quadrature mesh via further refinement of solution meshes accounting for the necessary quadrature order for expansion center. 
4. compute a solution and, if desired, refine again based on the solution and then re-solve. 

In a fault-surface intersection problem, there are two operators that we want to compute:
1. DLP, source = fault, target = free surf
2. DLP, source = free surf, target = free surf

In the Wala Stage 1 and Stage 2

```{code-cell} ipython3
import_and_display_fnc('common', 'PanelSurface')
import_and_display_fnc('common', 'panelize_symbolic_surface')
import_and_display_fnc('common', 'refine_panels')
import_and_display_fnc('common', 'stage1_refine')
import_and_display_fnc('common', 'qbx_panel_setup')
import_and_display_fnc('common', 'stage2_refine')
```

```{code-cell} ipython3
corner_resolution = 0.5
surf_half_L = 2000

qx, qw = gauss_rule(6)
t = sp.var("t")


control_points = np.array([(0, 0, 2, corner_resolution)])
fault = stage1_refine((t, t * 0, (t + 1) * -0.5), (qx, qw))
flat = stage1_refine(
    (t, -t * surf_half_L, 0 * t), (qx, qw), other_surfaces=[fault], control_points=control_points
)
expansions = qbx_panel_setup(flat, other_surfaces=[fault], direction=1, p=10)
fault_stage2, fault_interp_mat = stage2_refine(fault, expansions)
flat_stage2, flat_interp_mat = stage2_refine(flat, expansions)
```

```{code-cell} ipython3
%matplotlib inline
plt.figure()
plt.plot(fault.pts[:,0], fault.pts[:,1], 'r-o')
plt.plot(fault_stage2.pts[:,0], fault_stage2.pts[:,1], 'r*')
plt.plot(flat_stage2.pts[:,0], flat_stage2.pts[:,1], 'k-o')
plt.plot(expansions.pts[:,0], expansions.pts[:,1], 'bo')
for i in range(expansions.N):
    plt.gca().add_patch(plt.Circle(expansions.pts[i], expansions.r[i], color='b', fill=False))
plt.xlim([-0.5,0.5])
plt.ylim([-1, 0.1])
plt.show()
```

In the figure below, I plot $log_{10}(x)$ against the point index. You can see that the spacing of points is much finer near the fault surface intersection and rapidly increases away from the fault surface intersection.

```{code-cell} ipython3
plt.plot(np.log10(np.abs(flat.pts[:,0])))
plt.xlabel(r'$i$')
plt.ylabel(r'$\log_{10}(|x|)$')

plt.show()
```

```{code-cell} ipython3
print('number of points in the free surface discretization:', flat.n_pts)
print('                        fault        discretization:', fault.n_pts)
print('                        free surface     quadrature:', flat_stage2.n_pts)
print('                        fault            quadrature:', fault_stage2.n_pts)
```

```{code-cell} ipython3
%%time
A_raw = qbx_matrix(double_layer_matrix, flat_stage2, flat.pts, expansions)[:, 0, :]
```

```{code-cell} ipython3
%%time
A = A_raw.dot(flat_interp_mat.toarray())
```

```{code-cell} ipython3
B = -qbx_matrix(double_layer_matrix, fault_stage2, flat.pts, expansions)[:, 0, :]
```

```{code-cell} ipython3
lhs = np.eye(A.shape[0]) + A
rhs = B.dot(np.ones(fault_stage2.n_pts))
surf_disp = np.linalg.solve(lhs, rhs)

# Note that the analytical solution is slightly different than in the buried 
# fault setting because we need to take the limit of an arctan as the 
# denominator of the argument  goes to zero.
s = 1.0
analytical_fnc = lambda x: -np.arctan(-1 / x) / np.pi
analytical = analytical_fnc(flat.pts[:,0])
```

In the first row of graphs below, I show the solution extending to 10 fault lengths. In the second row, the solution extends to 1000 fault lengths. You can see that the solution matches to about 6 digits in the nearfield and 7-9 digits in the very farfield!

```{code-cell} ipython3
%matplotlib inline
for XV in [1.0, 10.0, 1000.0]:
    # XV = 5 * corner_resolution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(flat.pts[:, 0], surf_disp, "ko")
    plt.plot(flat.pts[:, 0], analytical, "bo")
    plt.xlabel("$x$")
    plt.ylabel("$u_z$")
    plt.title("Displacement")
    plt.xlim([-XV, XV])
    plt.ylim([-0.6, 0.6])

    plt.subplot(1, 2, 2)
    plt.plot(flat.pts[:, 0], np.log10(np.abs(surf_disp - analytical)))
    plt.xlabel("$x$")
    plt.ylabel(r"$\log_{10}|u_{\textrm{BIE}} - u_{\textrm{analytic}}|$")
    plt.title("Error (number of digits of accuracy)")
    plt.tight_layout()
    plt.xlim([-XV, XV])
    plt.show()
```

remaining parameter list:
- $\kappa$
- qbx order $p$
- qbx distance $r$ (or in the code `mult`), probably just leave this as $L/2$ (where $L$ is the panel length)
- mesh refinement

The two remaining parameters are $\kappa$ and $r$. I've decided on $r=0.5*L_{panel}$, so the only remaining issue is to set $\kappa$ based on the error tolerance $\epsilon$. The relationship will look like $\kappa = f(\epsilon, \textrm{panel shape}, \textrm{kernel})$ because the panel shape and the choice of kernel will also drive the error function. I can look at the Klinteberg paper to get a sense of what the error estimate should look like. Then I can just empirically fit the constants in that error estimate based.

```{code-cell} ipython3
surfs = []
solns = []
qx_ps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

surf_half_L = 2000

for qx_p in qx_ps:
    qx, qw = gauss_rule(qx_p)
    t = sp.var("t")
    control_points = np.array([(0, 0, 2, corner_resolution)])
    fault = stage1_refine((t, t * 0, (t + 1) * -0.5), (qx, qw))
    flat = stage1_refine(
        (t, -t * surf_half_L, 0 * t), (qx, qw), other_surfaces=[fault], control_points=control_points
    )
    expansions = qbx_panel_setup(flat, other_surfaces=[fault], direction=1, p=15)
    fault_stage2, fault_interp_mat = stage2_refine(fault, expansions)
    flat_stage2, flat_interp_mat = stage2_refine(flat, expansions)
    A_raw = qbx_matrix(double_layer_matrix, flat_stage2, flat.pts, expansions)[:, 0, :]
    A = A_raw.dot(flat_interp_mat.toarray())
    B = -qbx_matrix(double_layer_matrix, fault_stage2, flat.pts, expansions)[:, 0, :]
    lhs = np.eye(A.shape[0]) + A
    rhs = B.dot(np.ones(fault_stage2.n_pts))
    surf_disp = np.linalg.solve(lhs, rhs)
    surfs.append(flat)
    solns.append(surf_disp)
```

```{code-cell} ipython3
for i in range(len(qx_ps)):
    remove_end_idx = qx_ps[i] * 2
#     plt.plot(np.log10(np.abs(solns[i] - analytical_fnc(surfs[i].pts[:,0]))))
#     plt.show()
    diff = solns[i][remove_end_idx:-remove_end_idx] - analytical_fnc(surfs[i].pts[remove_end_idx:-remove_end_idx,0])
    l2_err = np.linalg.norm(diff)
    linf_err = np.max(np.abs(diff))
    print(qx_ps[i], l2_err, linf_err)
```
