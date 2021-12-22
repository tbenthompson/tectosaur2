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

```

```{code-cell} ipython3
t = sp.var("t")
qx, qw = gauss_rule(8)
A = (1 + 0.1 * sp.sin(10 * sp.pi * t))
free = stage1_refine(
    (t, A * sp.cos(sp.pi * t), A * sp.sin(sp.pi * t)), (qx, qw),
    max_radius_ratio=2.0
)
free_expansions = qbx_panel_setup(free, other_surfaces=[], direction=-1, p=10)
plt.figure(figsize=(10,10))
plt.plot(free.pts[:,0], free.pts[:,1], '-*')
plt.plot(free_expansions.pts[:,0], free_expansions.pts[:,1], 'b.')
plt.show()
print(free.n_panels)
```

```{code-cell} ipython3
free_stage2, free_interp_mat = stage2_refine(free, free_expansions)
M = qbx_matrix(double_layer_matrix, free_stage2, free.pts, free_expansions)[:,0,:].dot(free_interp_mat.toarray())
M.dot(np.ones(M.shape[1]))
```
