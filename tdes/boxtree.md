---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3.9 (XPython)
    language: python
    name: xpython
---

```python
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format='retina'

import cutde
```

```python
surf_L = 4000
n_els_per_dim = 50
mesh_xs = np.linspace(-surf_L, surf_L, n_els_per_dim + 1)
mesh_ys = np.linspace(-surf_L, surf_L, n_els_per_dim + 1)
mesh_xg, mesh_yg = np.meshgrid(mesh_xs, mesh_ys)
surf_pts = np.array([mesh_xg, mesh_yg, 0 * mesh_yg]).reshape((3, -1)).T.copy()
surf_tris = []
nx = ny = n_els_per_dim + 1
idx = lambda i, j: i * ny + j
for i in range(n_els_per_dim):
    for j in range(n_els_per_dim):
        x1, x2 = mesh_xs[i : i + 2]
        y1, y2 = mesh_ys[j : j + 2]
        surf_tris.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
        surf_tris.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
surf_tris = np.array(surf_tris, dtype=np.int64)
surf_tri_pts = surf_pts[surf_tris]
surf_centroids = np.mean(surf_tri_pts, axis=1)

# surf_surf_mats = []
# for d in range(3):
#     fictitious_slip = np.zeros((surf_tris.shape[0], 3))
#     fictitious_slip[:, d] = 1.0
#     surf_centers = np.mean(surf_tri_pts, axis=1)
#     surf_surf_mats.append(
#         cutde.disp_all_pairs(surf_centroids + np.array([0,0,0.01]), surf_pts[surf_tris], fictitious_slip, 0.25)
#     )
# surf_surf_mat = np.array(surf_surf_mats)
# lhs = np.transpose(surf_surf_mat, (1, 3, 2, 0))
# lhs_reordered = np.empty_like(lhs)
# lhs_reordered[:, :, :, 0] = lhs[:, :, :, 1]
# lhs_reordered[:, :, :, 1] = lhs[:, :, :, 0]
# lhs_reordered[:, :, :, 2] = lhs[:, :, :, 2]
# lhs_reordered = lhs_reordered.reshape((surf_tris.shape[0] * 3, surf_tris.shape[0] * 3))
# lhs_reordered += np.eye(lhs_reordered.shape[0])

# A = lhs_reordered
```

```python
element_radius = np.max(np.linalg.norm(surf_tri_pts - surf_centroids[:,None,:], axis=2), axis=1)
```

```python
from dataclasses import dataclass
from typing import Optional

@dataclass()
class TreeNode:
    obj_idxs: np.ndarray
    center: np.ndarray
    radius: float
    is_leaf: bool
    left: Optional['TreeNode']
    right: Optional['TreeNode']

def build_tree(all_objs, all_radii, min_pts_per_box=10, obj_idxs=None):
    if obj_idxs is None:
        obj_idxs = np.arange(all_objs.shape[0])
        
    objs = all_objs[obj_idxs]
    box_center = np.mean(objs, axis=0)
    sep = objs - box_center[None,:]
    box_axis_length = np.max(sep, axis=0)
    box_radius = np.max(np.linalg.norm(sep, axis=1) + all_radii[obj_idxs])
    
    node = TreeNode(obj_idxs, box_center, box_radius, is_leaf=True, left=None, right=None)
    if obj_idxs.shape[0] < min_pts_per_box:
        return node
    split_d = np.argmax(box_axis_length)

    is_left = objs[:, split_d] < box_center[split_d]
    left_obj_idxs = obj_idxs[np.where(is_left)[0]]
    right_obj_idxs = obj_idxs[np.where(~is_left)[0]]
    
    node.is_leaf = False
    node.left = build_tree(all_objs, all_radii, min_pts_per_box, left_obj_idxs)
    node.right = build_tree(all_objs, all_radii, min_pts_per_box, right_obj_idxs)
    
    return node

tree = build_tree(surf_centroids, element_radius, min_pts_per_box = 10)
```

```python
%matplotlib inline
def plot_tree(node, depth, **kwargs):
    if depth == 0:
        circle = plt.Circle(tuple(node.center[:2]), node.radius, fill=False, **kwargs)
        plt.gca().add_patch(circle)
    if node.left is None or depth == 0:
        return
    else:
        plot_tree(node.left, depth - 1, **kwargs)
        plot_tree(node.right, depth - 1, **kwargs)

plt.figure(figsize=(9,9))
for depth in [0,1,2,3,4,5,6,7,8]:
    plt.subplot(3,3,1+depth)
    plot_tree(tree, depth, color='b', linewidth=0.5)
    plt.xlim([tree.center[0] - tree.radius, tree.center[0] + tree.radius])
    plt.ylim([tree.center[1] - tree.radius, tree.center[1] + tree.radius])
plt.show()
```

```python
def _traverse(obs_node, src_node, safety_factor, direct_list, approx_list):
    dist = np.linalg.norm(obs_node.center - src_node.center)
    if dist > safety_factor * (obs_node.radius + src_node.radius):
        # We're far away, use an approximate interaction
        approx_list.append((obs_node, src_node))
    elif obs_node.is_leaf and src_node.is_leaf:
        # If we get here, then we can't split the nodes anymore but they are
        # still close. That means we need to use a exact interaction.
        direct_list.append((obs_node, src_node))
    else:
        # We're close by, so we should recurse and use the child tree nodes.
        # But which node should we recurse with? Or should we recurse with both
        split_src = ((obs_node.radius < src_node.radius) and not src_node.is_leaf) or obs_node.is_leaf;

        if split_src:
            _traverse(obs_node, src_node.left, safety_factor, direct_list, approx_list)
            _traverse(obs_node, src_node.right, safety_factor, direct_list, approx_list)
        else:
            _traverse(obs_node.left, src_node, safety_factor, direct_list, approx_list)
            _traverse(obs_node.right, src_node, safety_factor, direct_list, approx_list)

def traverse(obs_node, src_node, safety_factor=1.5):
    direct_list = []
    approx_list = []
    _traverse(obs_node, src_node, safety_factor, direct_list, approx_list)
    return direct_list, approx_list
```

```python
direct, approx = traverse(tree, tree)
```

```python
def direct_matrix(obs_node, src_node):
    mats = []
    for d in range(3):
        fictitious_slip = np.zeros((src_node.obj_idxs.shape[0], 3))
        fictitious_slip[:, d] = 1.0
        src_tri_pts = surf_tri_pts[src_node.obj_idxs]
        obs_tri_pts = surf_tri_pts[obs_node.obj_idxs]
        obs_pts = np.mean(obs_tri_pts, axis=1) + np.array([0,0,0.01])
        mats.append(cutde.disp_all_pairs(obs_pts, src_tri_pts, fictitious_slip, 0.25))
    M = np.array(mats)
    M = np.transpose(M, (1, 3, 2, 0))
    tmp = M[:,:,:,0].copy()
    M[:,:,:,0] = M[:,:,:,1]
    M[:,:,:,1] = tmp
    return M.reshape((
        obs_node.obj_idxs.shape[0] * 3,
        src_node.obj_idxs.shape[0] * 3
    ))
```

```python
M = direct_matrix(direct[0][0], direct[0][1])
```

```python
%%time
direct_matrices = [direct_matrix(d[0], d[1]) for d in direct]
```

```python
(surf_tris.shape[0] * 3) ** 2 * 8 / 1e9
```

```python
np.sum([d.nbytes for d in direct_matrices]) / 1e9
```

```python
def ACA_vector(shape, get_row, get_col, eps, max_iter=None):
    if max_iter is None:
        max_iter = shape[0]

    us = []
    vs = []
    RIk = np.empty(shape[0])
    RJk = np.empty(shape[0])
    # NOTE: For very high accuracy, Kahan summation may be needed while calculating RIk, RJk, Zappxmag?
    Zappxmag = 0 
    
    Ik = 0
    prev_Ik = [0]
    prev_Jk = []
    next_Ik = [None, 1, 2]
    cur_dim = 0
    
    for k in range(max_iter):

        RIk[:] = get_row(Ik)
        for i in range(len(us)):
            RIk -= us[i][Ik] * vs[i]

        Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
        if RIk[Jk] != 0.0:
            prev_Jk.append(Jk)
            vs.append(RIk / RIk[Jk])

            RJk[:] = get_col(Jk)
            for i in range(len(us)):
                RJk -= vs[i][Jk] * us[i]

            us.append(RJk.copy())
            step_size_sq = np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2)
            Zappxmag += step_size_sq
            for j in range(len(us) - 1):
                Zappxmag += 2 * us[-1].dot(us[j]) * vs[-1].dot(vs[j])

            prev_Ik.append(Ik)
            if step_size_sq > (eps ** 2) * Zappxmag:
                next_Ik[cur_dim] = argmax_not_in_list(np.abs(RJk), prev_Ik)

            print(Ik, Jk, step_size_sq, Zappxmag)
            cur_dim = (cur_dim + 1) % 3
        else:
            prev_Ik.append(Ik)
            next_Ik[cur_dim] = argmax_not_in_list(np.abs(RJk), prev_Ik)

        Ik = None
        for i in range(3):
            if next_Ik[cur_dim] is None:
                cur_dim = (cur_dim + 1) % 3
            else:
                Ik = next_Ik[cur_dim]
                next_Ik[cur_dim] = None
                break

        if Ik is None:
            break
    U = np.array(us).T
    V = np.array(vs)
    return U, V
```

```python
def approx_matrix(obs_node, src_node):
    def get_row(ri):
        
```

```python

```

There are a lot of improvements that we could make here:
* Re-order the elements so that each tree cell has a contiguous block of element indices. This would allow passing around a pointer to the start and end of those blocks instead of needing to pass around the entire list of indices.
* Integrate the 
* There are many other $O(n)$ sparsification methods for these types of problems. The fast multipole method is extremely fast and effective.

```python

```
