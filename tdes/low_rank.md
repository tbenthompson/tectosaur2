---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
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

surf_surf_mat = cutde.disp_matrix(surf_centroids + np.array([0,0,0.01]), surf_pts[surf_tris], 0.25)
```

```python
surf_surf_mat.shape
```

```python
lhs_reordered = np.empty_like(surf_surf_mat)
lhs_reordered[:, :, :, 0] = surf_surf_mat[:, :, :, 1]
lhs_reordered[:, :, :, 1] = surf_surf_mat[:, :, :, 0]
lhs_reordered[:, :, :, 2] = surf_surf_mat[:, :, :, 2]
lhs_reordered = lhs_reordered.reshape((surf_tris.shape[0] * 3, surf_tris.shape[0] * 3))
lhs_reordered += np.eye(lhs_reordered.shape[0])

A = lhs_reordered
```

```python
np.linalg.cond(A[:500,:500])
```

```python
plt.figure(figsize=(10,10))
logA = np.log10(A)
logA[np.isnan(logA)] = -5
plt.imshow(logA[:270,:270])
plt.colorbar()
plt.show()
```

```python
rand_cols = np.random.randint(A.shape[1], size=100)
```

```python
N = 1
A_frob = A.shape[1] / N * np.sum(A[:,rand_cols[:N]] ** 2)
```

```python
A_frob_true = np.sum(A ** 2)
A_frob_true
```

```python
A_frob
```

```python
block = A[12000:15000, :3000][::3][:,2::3]
block.shape
```

```python
block
```

```python
U, S, V = np.linalg.svd(block)
```

```python
plt.plot(np.log10(S))
plt.show()
```

```python
true_frob = np.sum(block ** 2)
```

```python
true_frob
```

```python
eps = 1e-10 * np.sqrt(block.size) / A.shape[0] * A_frob
appx_rank = np.where(S < eps)[0][0]
appx_rank
```

```python
2 * appx_rank * block.shape[0]
```

```python
2 * appx_rank * block.shape[0] / (block.shape[0] * block.shape[1])
```

```python
x = np.random.rand(block.shape[1])
```

```python
Uappx = U[:,:appx_rank]
Vappx = S[:appx_rank, None] * V[:appx_rank]
```

```python
%%time
y_true = block.dot(x)
```

```python
%%time
y_appx = Uappx.dot(Vappx.dot(x))
```

```python
speedup = 0.067 / 1.41
speedup
```

```python
y_true[:5], y_appx[:5]
```

```python
rel_err = np.abs((y_appx - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
frob = np.sum(Uappx.dot(Vappx) ** 2)
err_df = pd.DataFrame(
    index=['Rank', 'L2(Ax-y)','L1(Ax-y)','Linf(Ax-y)','Frobenius(A)'], 
    data=dict(SVD=[appx_rank, l2_err, l1_err, linf_err, frob])
)
l2_err, l1_err, linf_err
```

```python
def argmax_not_in_list(arr, disallowed):
    arg_sorted = arr.argsort()
    max_idx = arg_sorted.shape[0] - 1
    while True:
        if arg_sorted[max_idx] in disallowed:
            max_idx -= 1
        else:
            break
    return arg_sorted[max_idx]
```

```python
# SIMPLE 1D VERSION
Z = block
Ik = 0
prev_Ik = [0]
prev_Jk = []
us = []
vs = []
Zappxmag = 0
max_iter = 100
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

for k in range(max_iter):
    RIk[:] = Z[Ik,:]
    for i in range(k):
        RIk -= us[i][Ik] * vs[i]
    
    Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
    prev_Jk.append(Jk)

    RJk[:] = Z[:,Jk]
    for i in range(k):
        RJk -= vs[i][Jk] * us[i]
    
    vs.append(RIk / RIk[Jk])
    us.append(RJk.copy())
    step_size_sq = np.sum(us[k] ** 2) * np.sum(vs[k] ** 2)
    Zappxmag += step_size_sq
    for j in range(k - 1):
        Zappxmag += 2 * us[k].dot(us[j]) * vs[k].dot(vs[j])
    print(
        f'row={Ik:3d}, col={Jk:3d}, '
        f'step size={step_size_sq:1.3e}, '
        f'approximate matrix norm={Zappxmag:1.3e}'
    )
    
    if step_size_sq < (eps ** 2) * Zappxmag:
        break
    Ik = argmax_not_in_list(np.abs(RJk), prev_Ik)
    prev_Ik.append(Ik)
```

```python
len(us)
```

```python
np.sum((us[0][:,None] * vs[0][None,:]) ** 2)
```

```python
U_ACA = np.array(us).T
V_ACA = np.array(vs)
y_aca = U_ACA.dot(V_ACA.dot(x))
```

```python
rel_err = np.abs((y_aca - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
frob = np.sum(U_ACA.dot(V_ACA) ** 2)
err_df['ACA'] = [len(us), l2_err, l1_err, linf_err, frob]
l2_err, l1_err, linf_err
```

# SVD Recompression

```python
UQ, UR = np.linalg.qr(U_ACA)
VQ, VR = np.linalg.qr(V_ACA.T)
```

```python
W,SIG,Z = np.linalg.svd(UR.dot(VR.T))
```

```python
alpha = np.sqrt(Z.size) / A.shape[0]
```

```python
beta = (1 - alpha) / (1 + alpha * eps)
```

```python
beta
```

```python
r = np.argmax(SIG < eps * SIG[0])
r
```

```python
U_ACA2 = UQ.dot(W[:,:r] * SIG[:r])
V_ACA2 = Z[:r,:].dot(VQ.T)

y_aca_2 = U_ACA2.dot(V_ACA2.dot(x))

rel_err = np.abs((y_aca_2 - y_true) / y_true)
l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
frob = np.sum(U_ACA2.dot(V_ACA2) ** 2)
err_df[f'Recompress'] = [r, l2_err, l1_err, linf_err, frob]
print(r, l2_err, l1_err, linf_err)
```

MAKE A TABLE HERE: True SVD, ACA, ACA + Recompression

```python
eps
```

```python
err_dfT = err_df.T
err_dfT['True Frobenius(A)'] = np.sum(block ** 2)
err_dfT['Frobenius(A) Error'] = np.abs(err_dfT['True Frobenius(A)'] - err_dfT['Frobenius(A)'])
err_dfT
```

# A randomized handling of the vector problem

```python
block_xyz = A[12000:15000, :3000]
```

```python
x = np.random.rand(block_xyz.shape[1])
y_true = block_xyz.dot(x)
```

```python
frob_true = np.sum(block_xyz ** 2)
frob_true
```

```python

```

```python
def vector_ACA(mat):
    Z = block_xyz.copy()
    Ik = 0
    prev_Ik = [0]
    prev_Jk = []
    us = []
    vs = []
    Zappxmag = 0
    max_iter = 100
    RIk = np.empty_like(Z[0,:])
    RJk = np.empty_like(Z[:,0])

    for k in range(max_iter):
        RIk[:] = Z[Ik,:]
        for i in range(len(us)):
            RIk -= us[i][Ik] * vs[i]

        Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
        prev_Jk.append(Jk)

        RJk[:] = Z[:,Jk]
        for i in range(len(us)):
            RJk -= vs[i][Jk] * us[i]
            
        if RIk[Jk] == 0:
            continue

        vs.append(RIk / RIk[Jk])
        us.append(RJk.copy())
        step_size_sq = np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2)
        Zappxmag += step_size_sq
        for j in range(k - 1):
            Zappxmag += 2 * us[-1].dot(us[j]) * vs[-1].dot(vs[j])
#         print(
#             f'row={Ik:3d}, col={Jk:3d}, '
#             f'step size={step_size_sq:1.3e}, '
#             f'approximate matrix norm={Zappxmag:1.3e}'
#         )

        if step_size_sq < (eps ** 2) * Zappxmag:
            break_steps += 1
        else:
            break_steps = 0

        if break_steps > 5:
            break

        if k % 2 == 0:
            Ik = argmax_not_in_list(np.abs(RJk), prev_Ik)
        else:
            Ik = np.random.randint(Z.shape[0])
        prev_Ik.append(Ik)
        
    U = np.array(us).T
    V = np.array(vs)
    return U, V
```

```python
linf_errs = []
l1_errs = []
l2_errs = []
nappx = []
for i in range(10):
    U_ACA, V_ACA = vector_ACA(block_xyz)
    UQ, UR = np.linalg.qr(U_ACA)
    VQ, VR = np.linalg.qr(V_ACA.T)
    W,SIG,Z = np.linalg.svd(UR.dot(VR.T))
    r = np.argmax(SIG < eps * SIG[0])
    U_ACA2 = UQ.dot(W[:,:r] * SIG[:r])
    V_ACA2 = Z[:r,:].dot(VQ.T)
    
    y_aca = U_ACA2.dot(V_ACA2.dot(x))

    rel_err = np.abs((y_aca - y_true) / y_true)

    l2_err = np.sqrt(np.mean(rel_err ** 2))
    l1_err = np.mean(rel_err)
    linf_err = np.max(rel_err)
    frob = np.sum(U_ACA.dot(V_ACA) ** 2)
    err_df['ACA'] = [len(us), l2_err, l1_err, linf_err, frob]
    l2_err, l1_err, linf_err
    
    nappx.append(U_ACA2.shape[1])
    linf_errs.append(linf_err)
    l2_errs.append(l2_err)
    l1_errs.append(l1_err)
```

```python
nappx
```

```python
l2_errs
```

```python
plt.figure()
plt.hist(np.log10(l1_errs))
plt.figure()
plt.hist(np.log10(l2_errs))
plt.figure()
plt.hist(np.log10(linf_errs))
plt.figure()
plt.hist(nappx)
```

# An older randomized handling of the vector problem

```python
block_xyz = A[12000:15000, :3000]
```

```python
x = np.random.rand(block_xyz.shape[1])
y_true = block_xyz.dot(x)
```

```python
frob_true = np.sum(block_xyz ** 2)
frob_true
```

```python

```

```python
# SIMPLE 1D VERSION
Z = block_xyz.copy()
n_sample = (np.sqrt(Z.shape[0]) * 5).astype(int)
Ik_plan = np.random.randint(0, Z.shape[0], size=n_sample)
Jk_plan = np.random.randint(0, Z.shape[0], size=n_sample)
prev_Jk_subset_idx = []
prev_Ik_subset_idx = []
us = []
vs = []
Zappxmag = 0
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

for k in range(Ik_plan.shape[0]):
#     Ik_subset_idx = argmax_not_in_list(RJk[Ik_plan], prev_Ik_subset_idx)
#     prev_Ik_subset_idx.append(Ik_subset_idx)
#     Ik = Ik_plan[Ik_subset_idx]
    Ik = Ik_plan[k]
    
    RIk[:] = Z[Ik,:]
    for i in range(len(us)):
        RIk -= us[i][Ik] * vs[i]
    
    Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
    prev_Jk.append(Jk)
#     Jk_subset_idx = argmax_not_in_list(np.abs(RIk)[Jk_plan], prev_Jk_subset_idx)
#     prev_Jk_subset_idx.append(Jk_subset_idx)
#     Jk = Jk_plan[Jk_subset_idx]
    
    if RIk[Jk] == 0:
        continue

    RJk[:] = Z[:,Jk]
    for i in range(len(us)):
        RJk -= vs[i][Jk] * us[i]
    
    vs.append(RIk / RIk[Jk])
    us.append(RJk.copy())
    step_size_sq = np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2)
    Zappxmag += step_size_sq
    for j in range(len(us) - 1):
        Zappxmag += 2 * us[-1].dot(us[j]) * vs[-1].dot(vs[j])
    print(
        f'row={Ik:3d}, col={Jk:3d}, '
        f'step size={step_size_sq:1.3e}, '
        f'approximate matrix norm={Zappxmag:1.3e}'
    )
    
    if step_size_sq < (eps ** 2) * Zappxmag:
        break_steps += 1
    else:
        break_steps = 0
        
    if break_steps > 5:
        break
```

```python
len(us)
```

```python
eps
```

```python
U_ACA = np.array(us).T
V_ACA = np.array(vs)
y_aca = U_ACA.dot(V_ACA.dot(x))
```

```python
rel_err = np.abs((y_aca - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
frob = np.sum(U_ACA.dot(V_ACA) ** 2)
err_df['ACA'] = [len(us), l2_err, l1_err, linf_err, frob]
l2_err, l1_err, linf_err
```

# Handling the vector problem

```python
block_xyz = A[12000:15000, :3000]
```

```python
frob_true = np.sum(block_xyz ** 2)
frob_true
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
eps = 1e-3
U_ACA, V_ACA = ACA_vector(
    block_xyz.shape, 
    lambda ri: block_xyz[ri,:], 
    lambda ci: block_xyz[:,ci], 
    eps
)
```

```python
U_ACA.shape
```

```python
np.random.seed(1)
x = np.random.rand(block_xyz.shape[1])
y_true = block_xyz.dot(x)
```

```python
y_aca = U_ACA.dot(V_ACA.dot(x))
```

```python
rel_err = np.abs((y_aca - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
l2_err, l1_err, linf_err
```

SVD Recompression

```python
UQ, UR = np.linalg.qr(U_ACA)
VQ, VR = np.linalg.qr(V_ACA.T)
```

```python
W,SIG,Z = np.linalg.svd(UR.dot(VR.T))
```

```python
r_from_tolerance = np.argmax(SIG < eps/10)
r_from_tolerance
```

```python
for r in [r_from_tolerance + 10, r_from_tolerance]:
    U_ACA2 = UQ.dot(W[:,:r] * SIG[:r])
    V_ACA2 = Z[:r,:].dot(VQ.T)

    y_aca_2 = U_ACA2.dot(V_ACA2.dot(x))

    rel_err = np.abs((y_aca_2 - y_true) / y_true)
    l2_err = np.sqrt(np.mean(rel_err ** 2))
    l1_err = np.mean(rel_err)
    linf_err = np.max(rel_err)
    frob = np.sum(U_ACA2.dot(V_ACA2) ** 2)
    #err_df[f'Recompress{r}'] = [r, l2_err, l1_err, linf_err, frob]
    print(r, l2_err, l1_err, linf_err)
```

```python
(U_ACA2.size + V_ACA2.size) / block_xyz.size
```

# JUNK

```python
def argmax_not_in_list(arr, disallowed):
    arg_sorted = arr.argsort()
    max_idx = arg_sorted.shape[0] - 1
    while True:
        if arg_sorted[max_idx] in disallowed:
            max_idx -= 1
        else:
            break
    return arg_sorted[max_idx]

# A GOOD VECTOR VERSION
Z = block_xyz
Ik0 = 0
prev_Ik = []
prev_Jk = []
us = []
vs = []
Zappxmag = 0
eps = 1e-10 * np.sqrt(Z.size) / A.shape[0] * A_frob
max_iter = 150#appx_rank * 10
RIk3 = np.empty_like(Z[:3,:])
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

for k in range(max_iter): 
    RIk3[:,:] = Z[Ik0:Ik0+3,:]
    for i in range(len(us)):
        RIk3 -= us[i][Ik0:Ik0+3][:,None] * vs[i]
        
    Jk0 = argmax_not_in_list(np.max(np.abs(RIk3), axis=0), prev_Jk)
    Jk0 -= Jk0 % 3
    prev_Jk.extend([Jk0 + d for d in range(3)])
    
    done = True
    for d in range(3):
        RIk[:] = Z[Ik0+d,:]
        RJk[:] = Z[:,Jk0+d]
        for i in range(len(us)):
            RIk -= us[i][Ik0+d] * vs[i]
            RJk -= vs[i][Jk0+d] * us[i]
    
        vs.append(RIk / RIk[Jk0+d])
        us.append(RJk.copy())
        
        step_size_sq = np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2)
        Zappxmag += step_size_sq
        for j in range(len(us) - 1):
            Zappxmag += 2 * us[-1].dot(us[j]) * vs[-1].dot(vs[j])
        
        if step_size_sq > (eps ** 2) * Zappxmag:
            done = False
            
        print(
            f'row={Ik0+d:3d}, col={Jk0+d:3d}, '
            f'step size={step_size_sq:1.3e}, '
            f'approximate matrix norm={Zappxmag:1.3e}'
        )
    
    if done:
        break
        
    prev_Ik.extend([Ik0 + d for d in range(3)])
    Ik0 = argmax_not_in_list(np.abs(RJk), prev_Ik)
    Ik0 -= Ik0 % 3
```

```python
# A GOOD VECTOR VERSION
Z = block
Ik = 0
prev_Ik = [0]
prev_Jk = []
us = []
vs = []
Zappxmag = 0
max_iter = appx_rank * 10
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

dimensionality = 3
last_RJk = [None, None, None]
next_Ik = [None, 1, 2]
eps = 1e-9 * np.sqrt(Z.size) / A.shape[0] * A_frob
print(eps)
k = 0
cur_dim = 0
while True:
#     if k >= max_iter:
#         break
        
    RIk[:] = Z[Ik,:]
    for i in range(k):
        RIk -= us[i][Ik] * vs[i]
    
    Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
    if RIk[Jk] != 0.0:
        prev_Jk.append(Jk)
        vs.append(RIk / RIk[Jk])

        RJk[:] = Z[:,Jk]
        for i in range(k):
            RJk -= vs[i][Jk] * us[i]

        us.append(RJk.copy())
        last_RJk[cur_dim] = us[-1]
        step_size_sq = np.sum(us[k] ** 2) * np.sum(vs[k] ** 2)
        Zappxmag += step_size_sq
        for j in range(k - 1):
            Zappxmag += 2 * us[k].dot(us[j]) * vs[k].dot(vs[j])
        
        prev_Ik.append(Ik)
        if step_size_sq > (eps ** 2) * Zappxmag:
            next_Ik[cur_dim] = argmax_not_in_list(np.abs(RJk), prev_Ik)
                
        print(Ik, Jk, step_size_sq, Zappxmag)
        k += 1
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
    print(next_Ik, Ik)
    if Ik is None:
        break
```

```python
Z = block
dimensionality = 3
Ik = 0
prev_Ik = [0]
prev_Jk = []
us = []
vs = []
Zappxmag = 0
max_iter = appx_rank * 10
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

Ik_stack = [1, 2]
eps = 1e-6
k = 0
while True:
    if k >= max_iter:
        break
        
    RIk[:] = Z[Ik,:]
    for i in range(k):
        RIk -= us[i][Ik] * vs[i]
    
    Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
    if RIk[Jk] != 0.0:
        prev_Jk.append(Jk)
        vs.append(RIk / RIk[Jk])

        RJk[:] = Z[:,Jk]
        for i in range(k):
            RJk -= vs[i][Jk] * us[i]

        us.append(RJk.copy())
        step_size_sq = np.sum(us[k] ** 2) * np.sum(vs[k] ** 2)
        Zappxmag += step_size_sq
        for j in range(k - 1):
            Zappxmag += 2 * us[k].dot(us[j]) * vs[k].dot(vs[j])
        
        if step_size_sq > (eps ** 2) * Zappxmag:
            Ik_stack = []
            new_Ik = np.abs(RJk).argmax()
            new_Ik_x = new_Ik - (new_Ik % dimensionality)
            for d in range(dimensionality):
                Ik_stack.append(new_Ik_x + d)
                
        print(Ik, Jk, step_size_sq, Zappxmag)
        k += 1

    if len(Ik_stack) == 0:
        break
        
    prev_Ik.append(Ik)
    while Ik in prev_Ik or Ik >= Z.shape[0]:
        if len(Ik_stack) == 0:
            Ik = argmax_not_in_list(np.abs(RJk), prev_Ik)
        else:
            Ik = Ik_stack.pop()
```

```python
Z = block
Ik = 0
prev_Ik = [0]
prev_Jk = []
us = []
vs = []
Zappxmag = 0
max_iter = appx_rank * 10
RIk = np.empty_like(Z[0,:])
RJk = np.empty_like(Z[:,0])

next_Ik = [1, 2]
eps = 1e-6
k = 0
cur_dim = 0
while True:
    if k >= max_iter:
        break
        
    RIk[:] = Z[Ik,:]
    for i in range(k):
        RIk -= us[i][Ik] * vs[i]
    
    Jk = argmax_not_in_list(np.abs(RIk), prev_Jk)
    if RIk[Jk] != 0.0:
        prev_Jk.append(Jk)
        vs.append(RIk / RIk[Jk])

        RJk[:] = Z[:,Jk]
        for i in range(k):
            RJk -= vs[i][Jk] * us[i]

        us.append(RJk.copy())
        last_RJk[cur_dim] = us[-1]
        step_size_sq = np.sum(us[k] ** 2) * np.sum(vs[k] ** 2)
        Zappxmag += step_size_sq
        for j in range(k - 1):
            Zappxmag += 2 * us[k].dot(us[j]) * vs[k].dot(vs[j])
        
        prev_Ik.append(Ik)
        if step_size_sq > (eps ** 2) * Zappxmag:
            next_Ik.insert(0, argmax_not_in_list(np.abs(RJk), prev_Ik))
                
        print(Ik, Jk, step_size_sq, Zappxmag)
        k += 1
    else:
        prev_Ik.append(Ik)
        next_Ik.insert(0, argmax_not_in_list(np.abs(last_RJk[cur_dim]), prev_Ik))
    
    if len(next_Ik) == 0:
        break
    Ik = next_Ik.pop()
```
