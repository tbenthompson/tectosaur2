---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
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

```{code-cell} ipython3
import_and_display_fnc('common', 'refine_panels')
import_and_display_fnc('common', 'stage1_refine')
```

```{code-cell} ipython3
qx, qw = gauss_rule(16)
t = sp.var("t")

sym_obs_surf = (t, -t * 1000, 0 * t)
sym_src_surf = (t, t * 0, (t + 1) * -0.5)
src_panels = np.array([[-1, 1]])
src_surf = panelize_symbolic_surface(
    *sym_src_surf, src_panels, qx, qw
)

control_points = np.array([(0, 0, 2, 0.5)])
obs_surf = stage1_refine(
    sym_obs_surf, (qx, qw), other_surfaces=[src_surf], control_points=control_points
)
```

```{code-cell} ipython3
%matplotlib widget
plt.figure()
plt.plot(obs_surf.pts[obs_surf.panel_start_idxs,0], obs_surf.pts[obs_surf.panel_start_idxs,1], 'k-*')
plt.xlim([-25,25])
plt.show()
```

```{code-cell} ipython3
from common import qbx_panel_setup, build_interp_matrix, build_interpolator

expansions = qbx_panel_setup(obs_surf, direction=1, p=10)
```

```{code-cell} ipython3
import_and_display_fnc('common', 'build_panel_interp_matrix')
import_and_display_fnc('common', 'stage2_refine')
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
stage2_surf = stage2_refine(src_surf, expansions)
```

```{code-cell} ipython3
%matplotlib widget
plt.figure()
plt.plot(stage2_surf.pts[stage2_surf.panel_start_idxs,0], stage2_surf.pts[stage2_surf.panel_start_idxs,1], 'k-*')
plt.plot(expansions.pts[:,0], expansions.pts[:,1], 'r*')
plt.axis('equal')
plt.xlim([-1,1])
plt.ylim([-1,0])
plt.show()
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
rounded_corner_box = stage1_refine((t, x, y), (qx, qw), control_points = [(0,0,10000,5000)], max_radius_ratio=10.0)
```

```{code-cell} ipython3
%matplotlib inline
plt.figure()
plt.plot(
    rounded_corner_box.pts[rounded_corner_box.panel_start_idxs, 0],
    rounded_corner_box.pts[rounded_corner_box.panel_start_idxs, 1],
    "k-*",
)
plt.axis("equal")
plt.show()
```

```{code-cell} ipython3
box_expansions = qbx_panel_setup(rounded_corner_box, direction=1, p=10)
```

```{code-cell} ipython3
stage2_box = stage2_refine(rounded_corner_box, box_expansions)
print(stage2_box.n_panels)
plt.figure()
plt.plot(
    stage2_box.pts[stage2_box.panel_start_idxs, 0],
    stage2_box.pts[stage2_box.panel_start_idxs, 1],
    "k-*",
)
plt.plot(box_expansions.pts[:,0], box_expansions.pts[:,1], 'r*')
plt.axis("equal")
plt.show()
```

```{code-cell} ipython3

```
