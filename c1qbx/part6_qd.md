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
    build_panel_interp_matrix
)
import sympy as sp
```

```{code-cell} ipython3
t = sp.var('t')
theta = sp.pi + sp.pi * t
F = 0.999
u = F * sp.cos(theta)
v = F * sp.sin(theta)
x = 0.5 * (sp.sqrt(2 + 2 * u * sp.sqrt(2) + u ** 2 - v ** 2) - sp.sqrt(2 - 2 * u * sp.sqrt(2) + u**2 - v**2))
y = 0.5 * (sp.sqrt(2 + 2 * v * sp.sqrt(2) - u ** 2 + v ** 2) - sp.sqrt(2 - 2 * v * sp.sqrt(2) - u**2 + v**2))
x = (1.0 / F) * x * 300000
y = (1.0 / F) * y * 30000 - 30000
```

```{code-cell} ipython3
dxdt = sp.diff(x, t)
dydt = sp.diff(y, t)
jacobian = sp.sqrt(dxdt ** 2 + dydt ** 2)
dx2dt2 = sp.diff(dxdt, t)
dy2dt2 = sp.diff(dydt, t)
radius = jacobian ** 3 / (dxdt * dy2dt2 - dydt * dx2dt2)
```

```{code-cell} ipython3
# from dataclasses import dataclass
# from typing import List, Optional

# @dataclass()
# class AdaptiveTreeNode:
#     left: Optional['AdaptiveTreeNode']
#     right: Optional['AdaptiveTreeNode']
#     height: int
#     n_leaves: int

# def new_leaf():
#     return AdaptiveTreeNode(None, None, 0, 1)
        
# @dataclass()
# class AdaptiveTree:
#     root: AdaptiveTreeNode
    
#     def refine(self, should_refine):
        
#         def impose_min_height(n, min_height, side):
#             if n.height >= min_height:
#                 return n
#             if n.height > 0:
#                 if side:
#                     new_right = impose_min_height(n.right, min_height - 1, side)
#                     return process_interior_node(n, n.left, new_right)
#                 else:
#                     new_left = impose_min_height(n.left, min_height - 1, side)
#                     return process_interior_node(n, new_left, n.right)
#             else:
#                 return refine_leaf(n)
                
#         def side_height(n, side):
#             if n.height > 0:
#                 return side_height(n.right if side else n.left, side) + 1
#             else:
#                 return 0
        
#         def process_interior_node(n, new_left, new_right):
#             left_right_height = side_height(new_left, True)
#             right_left_height = side_height(new_right, False)
#             if left_right_height > right_left_height + 1:
#                 new_right = impose_min_height(new_right, left_right_height - 1, False)
#             elif right_left_height > left_right_height + 1:
#                 new_left = impose_min_height(new_left, right_left_height - 1, True)
                
#             new_height = max(new_left.height, new_right.height) + 1
#             new_n_leaves = new_left.n_leaves + new_right.n_leaves
#             return AdaptiveTreeNode(new_left, new_right, new_height, new_n_leaves)
        
#         def refine_leaf(n):
#             new_left = new_leaf()
#             new_right = new_leaf()
#             return AdaptiveTreeNode(new_left, new_right, 1, 2)
        
#         def refine_helper(n, idx):
#             if n.height > 0:
#                 new_left = refine_helper(n.left, idx)
#                 new_right = refine_helper(n.right, idx + n.left.n_leaves)
#                 return process_interior_node(n, new_left, new_right)
#             else:
#                 if should_refine[idx]:
#                     return refine_leaf(n)
#                 else:
#                     return n
        
#         return AdaptiveTree(refine_helper(self.root, 0))

#     def divide_domain(self):
#         panels = []
#         def divide_helper(n, left_edge, right_edge, idx):
            
#             if n.height == 0:
#                 panels.append((left_edge, right_edge))
#             else:
#                 midpt = (left_edge + right_edge) / 2
#                 divide_helper(n.left, left_edge, midpt, idx)
#                 divide_helper(n.right, midpt, right_edge, idx + n.left.n_leaves)

#         divide_helper(self.root, -1, 1, 0)
#         return panels
```

```{code-cell} ipython3
T = AdaptiveTree(new_leaf()).refine([1]).refine([0, 1]).refine([0, 1, 0]).refine([0,0,1,0,0]).refine([0,0,1,0,0,0,0]).refine([0,0,0,1,0,0,0,0])
```

```{code-cell} ipython3
panels = T.divide_domain()
len(panels), np.array(panels)
```

```{code-cell} ipython3
# qx, qw = gauss_rule(16)
# tree = AdaptiveTree(new_leaf())

# for i in range(15):
#     panels = tree.divide_domain()
#     cur_bounds = np.array(panels)
    
#     cur_surf = panelize_symbolic_surface(t, x, y, cur_bounds, qx, qw)

#     R_np = np.abs(symbolic_eval(t, cur_surf.quad_pts, radius))
#     panel_width = cur_surf.panel_bounds[:, 1] - cur_surf.panel_bounds[:, 0]

#     panel_radius = np.min(R_np.reshape((-1, qx.shape[0])), axis=1)
#     panel_length = np.sum((cur_surf.quad_wts * cur_surf.jacobians).reshape((-1, qx.shape[0])), axis=1)
    
#     # We want to refine when the panel has too much curvature. 
#     # One way to state this condition is that the panel length 
#     # must be less than half the panel radius of curvature.
#     refine = panel_length > 0.25 * panel_radius
#     tree = tree.refine(refine)
    
#     if np.sum(refine) == 0:
#         print(f'finished with {cur_bounds.shape[0]} panels')
#         break
        
#     print(f'refined {refine.sum()} panels out of {cur_bounds.shape[0]} total')
    
# #     plt.plot(cur_surf.pts[cur_surf.panel_start_idxs,0], cur_surf.pts[cur_surf.panel_start_idxs,1], 'k-*')
# #     plt.show()
    
# #     plt.plot(panel_radius)
# #     plt.plot(panel_length)
# #     plt.show()
```

```{code-cell} ipython3
plt.figure()
plt.plot(cur_surf.pts[cur_surf.panel_start_idxs,0], cur_surf.pts[cur_surf.panel_start_idxs,1], 'k-*')
plt.show()
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
import scipy.spatial

qx, qw = gauss_rule(16)
t = sp.var('t')

sym_obs_surf = (-t * 1000, 0 * t)
sym_src_surf = (t * 0, (t + 1) * -0.5)
src_panels = np.array([[-1,1]])
src_surf = panelize_symbolic_surface(t, sym_src_surf[0], sym_src_surf[1], src_panels, qx, qw)
```

```{code-cell} ipython3
control_points = np.array([(0, 0, 2, 0.5)])
control_tree = scipy.spatial.KDTree(control_points[:,:2])

s = src_surf
src_panel_centers = np.sum((s.quad_wts[:,None] * s.pts).reshape((-1, qx.shape[0], 2)), axis=1)
src_panel_length = np.sum((s.quad_wts * s.jacobians).reshape((-1, qx.shape[0])), axis=1)
src_surf_tree = scipy.spatial.KDTree(src_panel_centers)
```

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
cur_panels = np.array([[-1, 1]])

for i in range(50):
    cur_obs_surf = panelize_symbolic_surface(t, sym_obs_surf[0], sym_obs_surf[1], cur_panels, qx, qw)

    s = cur_obs_surf
    panel_parameter_domain = s.panel_bounds[:,1] - s.panel_bounds[:,0]
    
    panel_radius = np.min(s.radius.reshape((-1, qx.shape[0])), axis=1)

    refine_from_radius = s.panel_length > 0.25 * panel_radius

    nearby_controls = control_tree.query(s.panel_centers)
    nearest_control_pt = control_points[nearby_controls[1], :]
    refine_from_control = (
        (nearby_controls[0] < 0.5 * panel_length + nearest_control_pt[:, 2]) & 
        (panel_length > nearest_control_pt[:, 3])
    )
    
    nearby_surf_panels = src_surf_tree.query(s.panel_centers)
    refine_from_nearby = nearby_surf_panels[0] < 0.5 * panel_length + 0.5 * src_panel_length[nearby_surf_panels[1]]

    if n_panels > 1:
        panel_tree = scipy.spatial.KDTree(s.panel_centers)
        # Use k=2 because the closest panel will be the query panel.
        nearby_self_panels = panel_tree.query(s.panel_centers, k=2)
        self_dist = nearby_self_panels[0][:,1]
        self_idx = nearby_self_panels[1][:,1]
        self_panel_length = panel_length[self_idx]
        # The criterion will be: self_panel_length + sep < 0.5 * panel_length
        # but since sep = self_dist - 0.5 * panel_length - 0.5 * self_panel_length
        # we can simplify the criterion to:
        # Since the self distance metric is symmetric, we only need to check 
        # if the panel is too large.
        
        # Thinking about doubling
        # edges = 0, 1, 3, 7
        # centers = 0.5, 2, 5
        # self_dist = 4.5
        # self_panel_length = 1
        # panel_length = 4
        # 0.5 + 4.5 < 
        refine_from_self = 0.5 * self_panel_length + self_dist < 0.975 * panel_length
    else:
        refine_from_self = np.zeros(n_panels, dtype=bool)


    refine = refine_from_control | refine_from_radius | refine_from_self | refine_from_nearby
    new_panels = refine_panels(cur_panels, refine)
    
#     print('')
#     print('')
#     plt.plot(s.pts[s.panel_start_idxs,0], s.pts[s.panel_start_idxs,1], 'k-*')
#     plt.show()
#     print('nearby_controls: ', nearby_controls, 0.5*panel_length, control_points[nearby_controls[1], 2])
#     print('panel centers', panel_centers)
#     print('panel length', panel_length)
#     print('control', refine_from_control)
#     print('radius', refine_from_radius)
#     print('self', refine_from_self)
#     print('nearby', refine_from_nearby)
#     print('overall', refine)
    
    if new_panels.shape[0] == cur_panels.shape[0]:
        if np.any(refine):
            print("WTF")
        print(f'done after n_iterations={i} with n_panels={cur_panels.shape[0]}')
        break
    cur_panels = new_panels
    
obs_surf = cur_obs_surf
```

```{code-cell} ipython3
 %matplotlib widget
plt.figure()
print(cur_panels * 1000)
plt.plot(s.pts[s.panel_start_idxs,0], s.pts[s.panel_start_idxs,1], 'k-*')
plt.xlim([-25,25])
plt.show()
```

```{code-cell} ipython3
panel_length
```

```{code-cell} ipython3
from common import qbx_panel_setup
expansions = qbx_panel_setup(obs_surf, direction=1, p=10)
expansion_tree = scipy.spatial.KDTree(expansions.pts)
print(expansion_tree.query(src_surf.pts)[0])
min_panel_expansion_dist = np.min(expansion_tree.query(obs_surf.pts)[0].reshape((-1, qx.shape[0])), axis=1)
#list(zip(min_panel_expansion_dist, panel_length))
min_panel_expansion_dist < (0.5 - fudge_factor) * panel_length
```

```{code-cell} ipython3
list(zip(min_panel_expansion_dist, panel_length))
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
from common import build_interp_matrix, build_interpolator
import scipy.sparse

n_panels = 10
in_order = 16
out_order = 48

indptr = np.arange(n_panels + 1)
indices = np.arange(n_panels)
in_nodes = gauss_rule(in_order)[0]
data = []
for i in range(n_panels):
    out_nodes = 0.5 * gauss_rule(out_order)[0]
    if i % 2 == 0:
        out_nodes -= 0.5
    else:
        out_nodes += 0.5
    single_panel_interp = build_interp_matrix(build_interpolator(in_nodes), out_nodes)
    data.append(single_panel_interp)
shape = (n_panels * out_order, n_panels * in_order)
interp_mat = scipy.sparse.bsr_matrix((data, indices, indptr), shape)

plt.figure(figsize=(8,8))
plt.imshow(np.log10(np.abs(interp_mat.toarray())))
plt.show()
```

```{code-cell} ipython3
surf = obs_surf
stage1_panel_idxs = np.arange()
panel_parameter_ranges = []
stage1_panel_length = surf.pan
min_panel_expansion_dist < (0.5 - fudge_factor) * panel_length
```

```{code-cell} ipython3

```
