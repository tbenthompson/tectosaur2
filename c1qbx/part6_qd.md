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
%matplotlib widget
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from common import (
    gauss_rule,
    qbx_matrix,
    symbolic_eval,
    qbx_setup,
    double_layer_matrix,
    PanelSurface,
    panelize_symbolic_surface,
    build_panel_interp_matrix,
)
import sympy as sp
```

```{code-cell} ipython3
%matplotlib inline
```

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
def refine_panels(panels, which):
    new_panels = []
    for i in range(panels.shape[0]):
        if which[i]:
            left, right = panels[i]
            midpt = 0.5 * (left + right)
            new_panels.append([left, midpt])
            new_panels.append([midpt, right])
        else:
            new_panels.append(panels[i])
    new_panels = np.array(new_panels)
    return new_panels
```

```{code-cell} ipython3
import scipy.spatial


def stage1_refine(
    sym_surf,
    quad_rule,
    other_surfaces=[],
    initial_panels=np.array([[-1, 1]]),
    control_points=None,
    max_iter=30,
):
    cur_panels = initial_panels.copy()

    other_surf_trees = []
    for other_surf in other_surfaces:
        other_surf_trees.append(scipy.spatial.KDTree(other_surf.panel_centers))

    if control_points is not None:
        control_tree = scipy.spatial.KDTree(control_points[:, :2])

    for i in range(max_iter):
        cur_surf = panelize_symbolic_surface(
            t, sym_surf[0], sym_surf[1], cur_panels, *quad_rule
        )

        # Step 1) Refine based on radius of curvature
        panel_radius = np.min(cur_surf.radius.reshape((-1, quad_rule[0].shape[0])), axis=1)
        refine_from_radius = cur_surf.panel_length > 0.25 * panel_radius

        # Step 2) Refine based on a nearby user-specified control points.
        if control_points is not None:
            nearby_controls = control_tree.query(cur_surf.panel_centers)
            nearest_control_pt = control_points[nearby_controls[1], :]
            refine_from_control = (
                nearby_controls[0] < 0.5 * cur_surf.panel_length + nearest_control_pt[:, 2]
            ) & (cur_surf.panel_length > nearest_control_pt[:, 3])
        else:
            refine_from_control = np.zeros(cur_surf.n_panels, dtype=bool)

        # Step 3) Refine based on the length scale imposed by other nearby surfaces
        refine_from_nearby = np.zeros(cur_surf.n_panels, dtype=bool)
        for j, other_surf in enumerate(other_surfaces):
            nearby_surf_panels = other_surf_trees[j].query(cur_surf.panel_centers)
            nearby_dist = nearby_surf_panels[0]
            nearby_panel_length = other_surf.panel_length[nearby_surf_panels[1]]
            refine_from_nearby |= (
                0.5 * nearby_panel_length + nearby_dist < cur_surf.panel_length
            )
            

        # Step 4) Ensure that panel length scale doesn't change too rapidly. This 
        # essentially imposes that a panel will be no more than twice the length 
        # of any adjacent panel.
        if cur_surf.n_panels > 1:
            panel_tree = scipy.spatial.KDTree(cur_surf.panel_centers)
            # Use k=2 because the closest panel will be the query panel itself.
            nearby_panels = panel_tree.query(cur_surf.panel_centers, k=2)
            nearby_dist = nearby_panels[0][:, 1]
            nearby_idx = nearby_panels[1][:, 1]
            nearby_panel_length = cur_surf.panel_length[nearby_idx]
            # The criterion will be: self_panel_length + sep < 0.5 * panel_length
            # but since sep = self_dist - 0.5 * panel_length - 0.5 * self_panel_length
            # we can simplify the criterion to:
            # Since the self distance metric is symmetric, we only need to check
            # if the panel is too large.
            fudge_factor = 0.01
            refine_from_self = (
                0.5 * nearby_panel_length + nearby_dist < (1 - fudge_factor) * cur_surf.panel_length
            )
        else:
            refine_from_self = np.zeros(cur_surf.n_panels, dtype=bool)

        refine = (
            refine_from_control
            | refine_from_radius
            | refine_from_self
            | refine_from_nearby
        )
        new_panels = refine_panels(cur_panels, refine)

        #     plt.plot(s.pts[s.panel_start_idxs,0], s.pts[s.panel_start_idxs,1], 'k-*')
        #     plt.show()
        #     print('nearby_controls: ', nearby_controls, 0.5*panel_length, control_points[nearby_controls[1], 2])
        #     print('panel centers', panel_centers)
        #     print('panel length', panel_length)
#         print('control', refine_from_control)
#         print('radius', refine_from_radius)
#         print('self', refine_from_self)
#         print('nearby', refine_from_nearby)
#         print('overall', refine)
#         print('')
#         print('')


        if new_panels.shape[0] == cur_panels.shape[0]:
            if np.any(refine):
                print("WTF")
            print(f"done after n_iterations={i} with n_panels={cur_panels.shape[0]}")
            break
        cur_panels = new_panels
    return cur_surf


qx, qw = gauss_rule(16)
t = sp.var("t")

sym_obs_surf = (-t * 1000, 0 * t)
sym_src_surf = (t * 0, (t + 1) * -0.5)
src_panels = np.array([[-1, 1]])
src_surf = panelize_symbolic_surface(
    t, sym_src_surf[0], sym_src_surf[1], src_panels, qx, qw
)

control_points = np.array([(0, 0, 2, 0.5)])
obs_surf = stage1_refine(sym_obs_surf, (qx, qw), other_surfaces=[src_surf], control_points=control_points)
```

```{code-cell} ipython3
 %matplotlib widget
plt.figure()
#print(cur_panels * 1000)
plt.plot(obs_surf.pts[obs_surf.panel_start_idxs,0], obs_surf.pts[obs_surf.panel_start_idxs,1], 'k-*')
plt.xlim([-25,25])
plt.show()
```

```{code-cell} ipython3
from common import qbx_panel_setup

expansions = qbx_panel_setup(obs_surf, direction=1, p=10)
expansion_tree = scipy.spatial.KDTree(expansions.pts)
print(expansion_tree.query(src_surf.pts)[0])
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
# from common import build_interp_matrix, build_interpolator
# import scipy.sparse

# n_out_panels = 10
# in_order = 16
# out_order = 48

# indptr = np.arange(n_out_panels + 1)
# indices = np.arange(n_out_panels)
# in_nodes = gauss_rule(in_order)[0]
# data = []
# for i in range(n_panels):
#     out_nodes = 0.5 * gauss_rule(out_order)[0]
#     if i % 2 == 0:
#         out_nodes -= 0.5
#     else:
#         out_nodes += 0.5
#     single_panel_interp = build_interp_matrix(build_interpolator(in_nodes), out_nodes)
#     data.append(single_panel_interp)
# shape = (n_out_panels * out_order, n_panels * in_order)
# interp_mat = scipy.sparse.bsr_matrix((data, indices, indptr), shape)

# plt.figure(figsize=(8, 8))
# plt.imshow(np.log10(np.abs(interp_mat.toarray())))
# plt.show()
```

```{code-cell} ipython3
surf = obs_surf
stage2_panels = np.array([np.arange(surf.n_panels), -np.ones(surf.n_panels), np.ones(surf.n_panels)]).T
```

```{code-cell} ipython3
# panel_outer_bounds = obs_surf.panel_bounds[stage2_panels[:,0].astype(int)]
# to_domain = lambda x: panel_outer_bounds[:,0] + (x + 1) * 0.5 * (panel_outer_bounds[:,1] - panel_outer_bounds[:,0])
# quad_panels = np.array([to_domain(stage2_panels[:,1]), to_domain(stage2_panels[:,2])]).T
```

```{code-cell} ipython3
in_order = 16
out_order = 16
in_nodes = gauss_rule(in_order)[0]
out_q = gauss_rule(out_order)

in_panel_idx = stage2_panels[:, 0].astype(int)
left = stage2_panels[:,1][:, None]
right = stage2_panels[:,2][:, None]
out_relative_nodes = (left + (right - left) * (out_q[0][None,:] + 1) * 0.5)
panel_parameter_width = surf.panel_bounds[:, 1] - surf.panel_bounds[:, 0]
out_nodes = surf.panel_bounds[in_panel_idx, 0, None] + panel_parameter_domain[in_panel_idx, None] * (out_relative_nodes + 1) * 0.5
```

```{code-cell} ipython3
out_node_wts = ((out_q[1][None,:] * 0.25 * (right - left)) * panel_parameter_width[in_panel_idx, None]).ravel()
```

```{code-cell} ipython3
n_out_panels = stage2_panels.shape[0]
shape = (n_out_panels * out_order, surf.n_panels * in_order)
indptr = np.arange(n_out_panels + 1)
indices = stage2_panels[:, 0].astype(int)
interp_mat_data = []
for i in range(n_out_panels):
    single_panel_interp = build_interp_matrix(build_interpolator(in_nodes), out_relative_nodes[i])
    interp_mat_data.append(single_panel_interp)
interp_mat = scipy.sparse.bsr_matrix((interp_mat_data, indices, indptr), shape)
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(interp_mat.toarray()[:200,:200])
plt.show()
```

```{code-cell} ipython3
stage2_pts = interp_mat.dot(surf.pts)
stage2_jacobians = interp_mat.dot(surf.jacobians)
```

```{code-cell} ipython3
stage2_panel_lengths = np.sum((out_node_wts * stage2_jacobians).reshape((-1, qx.shape[0])), axis=1)
```

```{code-cell} ipython3
# panel_parameter_domain = surf.panel_bounds[:, 1] - surf.panel_bounds[:, 0]
# quad_pts = (
#     panel_bounds[:, 0, None] + panel_parameter_domain[:, None] * (qx[None, :] + 1) * 0.5
# ).flatten()
# pts = np.hstack((surf_vals[0][:, None], surf_vals[1][:, None]))
# quad_wts = (panel_parameter_domain[:, None] * qw[None, :] * 0.5).flatten()
# panel_length = np.sum((quad_wts * jacobians).reshape((-1, qx.shape[0])), axis=1)
```

```{code-cell} ipython3
# qx = 
# 
# quad_pts = (
#     surf.panel_bounds[:, 0, None] + panel_parameter_domain[:, None] * (qx[None, :] + 1) * 0.5
# ).flatten()


min_panel_expansion_dist = np.min(
    expansion_tree.query(stage2_pts)[0].reshape((-1, qx.shape[0])), axis=1
)
# list(zip(min_panel_expansion_dist, panel_length))
fudge_factor = 0.01
refine = min_panel_expansion_dist < (0.5 - fudge_factor) * stage2_panel_lengths
print(refine)
#list(zip(min_panel_expansion_dist, surf.panel_length))
```

```{code-cell} ipython3
new_quad_panels = []
for i in range(stage2_panels.shape[0]):
    if refine[i]:
        left = stage2_panels[i, 1]
        right = stage2_panels[i, 2]
        midpt = 0.5 * (left + right)
        new_quad_panels.append((stage2_panels[i, 0], left, midpt))
        new_quad_panels.append((stage2_panels[i, 0], midpt, right))
    else:
        new_quad_panels.append(stage2_panels[i])
new_quad_panels = np.array(new_panels)
stage2_panels = new_quad_panels
```

```{code-cell} ipython3
new_quad_panels
```

```{code-cell} ipython3
midpt = 
np.array([[stage2_panels[:,1], midpt],[midpt, stage2_panels[:,2]]]).reshape((4,-1)).T.reshape((-1, 2))[np.repeat(refine, 2)]
```

```{code-cell} ipython3
np.repeat(stage2_panels, refine.astype(int) + 1, axis=0)
```

```{code-cell} ipython3
stage1_panel_idxs = np.arange()
panel_parameter_ranges = []
stage1_panel_length = surf.pan
min_panel_expansion_dist < (0.5 - fudge_factor) * panel_length
```

```{code-cell} ipython3
t = sp.var("t")
theta = sp.pi + sp.pi * t
F = 0.98
u = F * sp.cos(theta)
v = F * sp.sin(theta)
x = 0.5 * (
    sp.sqrt(2 + 2 * u * sp.sqrt(2) + u ** 2 - v ** 2)
    - sp.sqrt(2 - 2 * u * sp.sqrt(2) + u ** 2 - v ** 2)
)
y = 0.5 * (
    sp.sqrt(2 + 2 * v * sp.sqrt(2) - u ** 2 + v ** 2)
    - sp.sqrt(2 - 2 * v * sp.sqrt(2) - u ** 2 + v ** 2)
)
x = (1.0 / F) * x * 100000
y = (1.0 / F) * y * 20000 - 20000
```

```{code-cell} ipython3
rounded_corner_box = stage1_refine((x, y), (qx, qw))
```

```{code-cell} ipython3
plt.figure()
plt.plot(rounded_corner_box.pts[rounded_corner_box.panel_start_idxs,0], rounded_corner_box.pts[rounded_corner_box.panel_start_idxs,1], 'k-*')
plt.axis('equal')
plt.show()
```
