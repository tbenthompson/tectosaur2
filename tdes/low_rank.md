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

# Low rank approximation of BEM matrices with adaptive cross approximation (ACA).

In the last section, I demonstrated how to avoid constructing the full dense BEM matrix by regenerating the matrix whenever they are needed. This can be helpful for reducing memory costs and, in some situations, actually results in a faster solver too. Here, I'll improve upon that solution by demonstrating how the off-diagonal blocks of a BEM matrix can be compressed via a low-rank approximation. The result will be $O(n\log n )$ solution methods that can scale up to millions of elements. Hierarchical matrices, tree-codes, fast multipole methods and several other techniques all make use of this concept.

To start out, let's generate, yet again, a simple self-interaction matrix for a free surface. I hid these cells since the code is nothing new.

```python tags=[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%config InlineBackend.figure_format='retina'

import cutde
```

```python tags=[]
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

lhs_reordered = np.empty_like(surf_surf_mat)
lhs_reordered[:, :, :, 0] = surf_surf_mat[:, :, :, 1]
lhs_reordered[:, :, :, 1] = surf_surf_mat[:, :, :, 0]
lhs_reordered[:, :, :, 2] = surf_surf_mat[:, :, :, 2]
lhs_reordered = lhs_reordered.reshape((surf_tris.shape[0] * 3, surf_tris.shape[0] * 3))
lhs_reordered += np.eye(lhs_reordered.shape[0])

A = lhs_reordered
```

## Near-field vs far-field

But, this time, let's dig in and investigate the matrix itself. In particular, we'll start by looking at two blocks of the matrix. 
1. A "near-field" block will contain the diagonal of the matrix. Remember that the diagonal of the matrix consists of entries representing the displacement at the center of the same element on which the slip occurred. You can see the bright yellow diagonal in the figure below. The coefficients also decay rapidly away from the diagonal. 
2. The other, a "far-field" block will consist of matrix entries coming from interactions between observation points and source elements that are very far from each other. In the figure below, there's no intense variation in coefficients. 

The thing to notice here is that there is, in some sense, just a lot more going on in the near-field matrix.

```python
nrows = 150
near_field_block = A[:nrows,:nrows]
far_field_block = A[-nrows:, :nrows]

log_near = np.log10(near_field_block)
log_near[np.isnan(log_near)] = -10
log_far = np.log10(far_field_block)
log_far[np.isnan(log_far)] = -10

fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(log_near, vmin=-10, vmax=1)
plt.title('Near-field')
plt.subplot(1,2,2)
ims = plt.imshow(log_far, vmin=-10, vmax=1)
plt.title('Far-field')
cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.75])
cb = fig.colorbar(ims, cax=cbar_ax)
plt.show()
```

Another way of visualizing this same "a lot going on" property would be to look at the action of the matrix. So, we'll apply both the nearfield and far-field blocks to a vector with random elements. The difference is striking: the output from the near-field matrix preserves the "randomness" of the input whereas the far-field matrix actually smooths out the input dramatically. 

```python
x = np.random.rand(nrows)
y1 = near_field_block.dot(x)
y2 = far_field_block.dot(x)

plt.figure(figsize = (8, 4))
plt.subplot(1,2,1)
plt.plot(np.log10(np.abs(y1[2::3])))
plt.title('Near-field')
plt.subplot(1,2,2)
plt.plot(np.log10(np.abs(y2[2::3])))
plt.title('Far-field')
plt.show()
```

A third and more rigorous way of discussing this same property is to look at the singular values of the two matrix blocks.

```python
U_near, S_near, V_near = np.linalg.svd(near_field_block)
U_far, S_far, V_far = np.linalg.svd(far_field_block)
```

The plot below shows the $\log_{10}$ ratio of each singular value to the first singular value. Things to note here:
* The near-field singular values do not decay. They are almost all between 1.58 and 1.41. This reflects the behavior above where the randomness of the `x` vector was mostly preserved.
* The far-field singular values decay very quickly. The majority of the singular values are smaller than 1e-6. In other words, the matrix has a "1e-6 approximate rank" of 24 despite technically have a rank of 150. That means that if we only care about accuracy up to about 6 digits, then we can compress this matrix by a factor of 6. 

```python
# The 24th singular value is below 1e-6.
S_far[24] / S_far[0]
```

```python
plt.plot(np.log10(S_near / S_near[0]), label = 'nearfield')
plt.plot(np.log10(S_far / S_far[0]), label = 'farfield')
plt.legend()
plt.show()
```

## Approximation with the singular value decomposition (SVD)

So, if these off-diagonal blocks of the matrix have singular values that decay quite quickly, can we approximate these blocks with just a few singular values? The answer is an emphatic yes and this idea is the basis of low-rank approximation methods. Let's explore this idea a bit more with a larger block of the matrix. Because it's also off-diagonal, this block shows the same singular value magnitude decay that we saw before. 

```python
block = A[-3000:,:3000]
U, S, V = np.linalg.svd(block)

plt.plot(np.log10(S / S[0]))
plt.show()
```

So, what if we take all the singular values greater than $10^{-5} * S_0$ in magnitude. There are only 74 such singular values, meaning that we can compress this matrix by a factor of 10.

```python
appx_rank = np.where(S < 1e-6 * S[0])[0][0]
appx_rank
```

And let's take a look at the performance and error resulting from applying this low rank approximation to a random vector. First, I'll create an random input vector `x`. Then, I'll form two matrices `Uappx` and `Vappx`. You can sort of think of these as the entrance and exit to our low-dimensional space that very efficiently represents the TDE interaction. Mutliplying `Vappx` by the 1500-dimensional `x` returns a 74-dimensional vector and then we expand back to 1500 dimensional by multiplying by `Uappx`. 

```python
x = np.random.rand(block.shape[1])

Uappx = U[:,:appx_rank]
Vappx = S[:appx_rank, None] * V[:appx_rank]
```

Next, I'll calculate the correct matrix vector product, `y_true` and record the runtime. 

```python
full_time = %timeit -o block.dot(x)
y_true = block.dot(x)
```

And then, calculate the low-rank matrix vector product, `y_appx` and record the runtime.

```python
lowrank_time = %timeit -o Uappx.dot(Vappx.dot(x))
y_appx = Uappx.dot(Vappx.dot(x))
```

A cursory comparison of the output suggests that the approximation is very very accurate.

```python
print(y_true[:5], y_appx[:5])

speedup = full_time.best / lowrank_time.best
memory_reduction = block.nbytes / (Uappx.nbytes + Vappx.nbytes)
speedup, memory_reduction
```

```python
from myst_nb import glue
my_variable = "here is some text!"
glue("cool_text", my_variable)
```

```{glue:}`cool_text`
```


 And the runtime of the low-rank version is {{speedup}}


```{code-cell}
print("Hallo!")
```


Let's do a bit more detailed investigation of the error and look at the $L^1$, $L^2$ and $L^{\infty}$ error. We'll also compare the Frobenius norm of the approximated matrix with the Frobenius norm of the original matrix. 

```python
rel_err = np.abs((y_appx - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
print(f'L1        = {l1_err},  L2        = {l2_err},   Linf  = {linf_err}')

frob = np.sum(Uappx.dot(Vappx) ** 2)
true_frob = np.sum(block ** 2)
frob, true_frob, np.abs(frob - true_frob)
print(f'appx frob = {frob}  true_frob = {true_frob},  error = {np.abs(frob-true_frob)}')
```

It looks like this SVD approximation worked out really well here! We're getting tolerable matrix-vector product errors that are all on the same order of magnitude as the threshold we used for the singular value cutoff. And the approximate matrix itself is extremely similar to the original matrix.

Before we move on, I'll record these error values in a dataframe so that it's easy to compare with the fancier methods in the next two sections.

```python
err_df = pd.DataFrame(
    index=['Rank', 'L2(Ax-y)','L1(Ax-y)','Linf(Ax-y)','Frobenius(A)'], 
    data=dict(true=[block.shape[0], 0, 0, 0, true_frob])
)
err_df['SVD'] = [appx_rank, l2_err, l1_err, linf_err, frob]
```

```python
err_df.T
```

## Adaptive cross approximation (ACA)

So, we've managed to create an extremely efficient approximation our off-diagonal matrix block by using the SVD. That's definitely useful for computing fast matrix-vector products or even for computing a LU decomposition. But, it still suffers from the need to compute the entire matrix block in the first place. If we're going to be throwing all that information away immediately after computing the SVD, is there a way to avoid computing the dense matrix block in the first place? There are several solutions to this problem including randomized SVDs, but the most useful solution for our setting is the adaptive cross approximation (ACA) method. These algorithms depending on the ability to compute arbitrary individual matrix entries without computing the entire matrix. By making certain assumptions about the structure of a matrix, we can be confident that an accurate approximation can be constructed from just a few entries. 

The basic idea of ACA is to approximate a matrix with the as a rank 1 outer product of one row and one column of that same matrix. And then iteratively use this process to construct a approximation of arbitrary precision. Ideally, at each step, we will choose the best fitting row and column. After the first iteration, we are no longer trying to approximate the original matrix, but instead the residual matrix formed by the difference between the original matrix and the current approximation matrix. Eventually, assuming certain matrix properties that are proven true for BEM problem, the procedure will converge. 

For the sake of real-world usage, the description above will be sufficient, but

## ACA and ACA+, the details
The simple version, runs like (following CITE GRASEDYCK 2005):

**ACA with full pivoting**: Given a matrix $M \in \mathbb{R}^{n x m}$, we'll construct an approximation like $\sum_{k}^{r} u_k v_k^T$. The task will be to construct $u_k$ and $v_k$ on each iteration such that the Frobenius norm error eventually converges. To do that, the key will be to iteratively form rank-1 approximations to the residual matrix. The residual matrix is the matrix forming the difference between $M$ and the current approximation and can be written as $R_{ij} = M_{ij} - \sum_{k}^{r} u_{kj} v_{ki}$ which represents the entry-wise difference between the target matrix and the current approximation. The goal will be to have $R$ satisfy $\|R\|_2 < \epsilon\|M\|_2$ where $\epsilon$ is a user-specified accuracy parameter. To do this, during each iteration:
1. Determine a "pivot" $(i^*, j^*)$ as the indices that maximize $R_{ij}$.
2. Assign: 

    \begin{align}
    u_{kj} &= R_{i^*,j} / R_{i^*,j^*} \\
    v_{ki} &= R_{i,j^*}
    \end{align}
3. Update $R$ to account for the new rank-1 update to the approximation. 
4. If the magnitude of the $u_k v_k$ update is small enough, stop. Otherwise, return to step 1.

The reason the algorithm is called "ACA with full pivoting" is because we're allowing the algorithm to choose an arbitrary $(i^*, j^*)$ in the first step. However, that is impossible in our real-world setting because we don't have all the entries of $M$. 

**ACA+**: So, instead, most real-world application use either ACA with partial pivoting or the ACA+ algorithm. Here, I'll introduce the modifications necessary for ACA+. The main distinction is that instead of searching over all matrix indices in step 1, we will search over a subset specified by a random row and a random row. 

Before starting the iteration, we choose a random row, $i_{\mathrm{ref}}$ and random $j_{\mathrm{ref}}$. And we will maintain the corresponding row and column of the residual matrix, $R$. At the start of the algorithm $R_{i_{\mathrm{ref}}, j} = M_{i_{\mathrm{ref}}, j}$ and $R_{i, j_{\mathrm{ref}}} = M_{i, _{\mathrm{ref}}j}$ 

Then, the iteration proceeds like:
1. Find the index $j^*$ that maximizes $R_{i_{\mathrm{ref}}, j}$ and the index $i^*$ that maximizes $R_{i, j_{\mathrm{ref}}}$. Essentially, we are finding the largest entries in each of these vectors. 
2. If $R_{i_{\mathrm{ref}}, j^*} > R_{i^*, j_{\mathrm{ref}}}$ then, we compute column corresponding to $j^*$ or vice-versa or the row corresponding to $i^*$. Essentially, we are determining here whether we should pivot first based on the row or the column. 
3. If we pivoted based on column, we should now have computed a new residual row $R_{i,j^*} = M_{i,j^*} - \sum_{k}^{r} u_{kj^*} v_{ki}$. Find the missing pivot index now by maximizing $R_{i,j^*}$ to get $i^*$. Or if we pivoted on the row, we will have a new residual column $R_{i^*,j} = M_{i^*,j} - \sum_{k}^{r} u_{kj} v_{ki^*}$ and we maximize $R_{i^*,j}$ to get $j^*$. The idea here is to finish the pivot operation from step 2 by pivoting in the dimension that we have not considered yet. At each step, we are essentially trying to find the largest residual matrix element out of all the entries we have seen so far with the goal of getting as close as possible to the full pivoting algorithm without actually needing to calculate all the residual matrix elements. In particular, note how we calculate the residual row (column) here by compute the original matrix row and then subtracting the row (column) of the current approximation. This is critical since it means that we're only compunting a single row or column of the original matrix. (Aside: Why am I referring to the identification of the largest entries as "pivoting"? This is by analogy to various matrix operations like LU decomposition where the numerical stability is best when the largest entries are handled first. ACA can also be reframed as an iterative triangular decomposition of a matrix.)
4. Next, just like in the full pivoting algorithm, assign: 

    \begin{align}
    u_{kj} &= R_{i^*,j} / R_{i^*,j^*} \\
    v_{ki} &= R_{i,j^*}
    \end{align}
5. And update the $R_{i_{\mathrm{ref}}, j}$ row and $R_{i, j_{\mathrm{ref}}}$ by subtracting the new terms of the approximation.
6. Finally, if the magnitude of the $u_k v_k$ update is small enough, stop. Otherwise, return to step 1.

I've deliberately left some of the details vague in order to make the salient features of the algorithm more prominent. But, below, I'm going to go through a full implementation of the algorithm so hopefully that will clear up any of the details.


## Implementing ACA+

An implementation of ACA+ is below. I've put lots of comments throughout to help explain the details. But, if you don't want to dive in on the details here, 

```python
def ACA_plus(M, eps, verbose=False):
    # a quick helper function that will help find the largest entry in 
    # an array while excluding some list of `disallowed`  entries.
    def argmax_not_in_list(arr, disallowed):
        arg_sorted = arr.argsort()
        max_idx = arg_sorted.shape[0] - 1
        while True:
            if arg_sorted[max_idx] in disallowed:
                max_idx -= 1
            else:
                break
        return arg_sorted[max_idx]

    M = block # The matrix we'd like to approximate.
    us = [] # The left vectors of the approximation
    vs = [] # The right vectors of the approximation
    prevIstar = [] # Previously used i^* pivots
    prevJstar = [] # Previously used j^* pivots

    # Re-usable function for finding a reference row and updating 
    # it with respect to the already constructed approximation.
    def reset_reference_row():
        while True:
            Iref = np.random.randint(M.shape[0])
            # Given the vector nature of the problem, we're going to grab
            # an entire 3D "row", so let's get the first index
            Iref -= Iref % 3
            # It's important to avoid re-using a row.
            if not ((Iref + 0) in prevIstar or (Iref + 1) in prevIstar or (Iref + 2) in prevIstar):
                break
                
        # Grab the "row" (actually three rows corresponding to the 
        # x, y, and z components for a single observation point)
        # And, of course, since we want a row of the residual matrix, not the
        # original matrix, we need to subtract the terms of the approximation
        out = M[Iref:Iref+3,:].copy()
        for i in range(len(us)):
            out -= us[i][Iref:Iref+3][:,None] * vs[i][None,:]
        return out, Iref

    # Same function as above but for the column
    def reset_reference_col():
        while True:
            Jref = np.random.randint(M.shape[1])
            Jref -= Jref % 3
            if not ((Jref + 0) in prevJstar or (Jref + 1) in prevJstar or (Jref + 2) in prevJstar):
                break
        out = M[:,Jref:Jref+3].copy()
        for i in range(len(us)):
            out -= vs[i][Jref:Jref+3][None,:] * us[i][:,None]
        return out, Jref

    # The Frobenius norm of the approximation matrix. This will be updated as
    # we construct the approximation iteratively.
    appx_frob = 0

    # If we haven't converged before running for max_iter, we'll stop anyway.
    max_iter = 250

    # Create a buffer for storing the R_{i^*,j} and R_{i, j^*}
    RIstar = np.zeros_like(M[0,:])
    RJstar = np.zeros_like(M[:,0])

    # Choose our starting reference row and column.
    RIref, Iref = reset_reference_row()
    RJref, Jref = reset_reference_col()

    for k in range(max_iter):
        # These two lines find the column in RIref with the largest entry (step 1 above). 
        maxabsRIref = np.max(np.abs(RIref), axis=0)
        Jstar = argmax_not_in_list(maxabsRIref, prevJstar)

        # And these two find the row in RJref with the largest entry (step 1 above). 
        maxabsRJref = np.max(np.abs(RJref), axis=1)
        Istar = argmax_not_in_list(maxabsRJref, prevIstar)

        # Check if we should pivot first based on row or based on column (step 2 above)
        Jstar_val = maxabsRIref[Jstar]
        Istar_val = maxabsRJref[Istar]
        if Istar_val > Jstar_val:
            # If we pivot first on the row, then calculate the corresponding row
            # of the residual matrix by first grabbing the row of the original 
            # matrix and then subtracting the current approximation.
            RIstar[:] = M[Istar,:]
            # Subtracting the Istar-th row of the current approximation
            for i in range(len(us)):
                RIstar -= us[i][Istar] * vs[i]

            # Then find the largest entry in that row vector to identify which 
            # column to pivot on. (See step 3 above)
            Jstar = argmax_not_in_list(np.abs(RIstar), prevJstar)

            # Calculate the column by grabbing from the original matrix and 
            # subtracting the matrix approximation we've constructed so far
            RJstar[:] = M[:,Jstar]
            for i in range(len(us)):
                RJstar -= vs[i][Jstar] * us[i]
        else:
             # If we pivot first on the column, then calculate the corresponding column
            # of the residual matrix by first grabbing the row of the original 
            # matrix and then subtracting the current approximation
            RJstar[:] = M[:,Jstar]
            # Subtract the Jstar-th column of the current approximation
            for i in range(len(us)):
                RJstar -= vs[i][Jstar] * us[i]

            # Then find the largest entry in that row vector to identify which 
            # column to pivot on.  (See step 3 above)
            Istar = argmax_not_in_list(np.abs(RJstar), prevIstar)

            # Calculate the row by grabbing from the original matrix and 
            # subtracting the matrix approximation we've constructed so far
            RIstar[:] = M[Istar,:]
            for i in range(len(us)):
                RIstar -= us[i][Istar] * vs[i]

        # Record the pivot row and column so that we don't re-use them.
        prevIstar.append(Istar)
        prevJstar.append(Jstar)

        # Add the new rank-1 outer product to the approximation (see step 4 above)
        vs.append(RIstar / RIstar[Jstar])
        us.append(RJstar.copy())

        # If we pivoted on the reference row, then choose a new reference row.
        # Remember that we are using a x,y,z vector "row" or 
        # set of 3 rows in an algebraic sense.
        if Iref <= Istar < Iref + 3:
            RIref, Iref = reset_reference_row()
        else:
            # If we didn't change the reference row of the residual matrix "R",
            # update the row to account for the new components of the approximation.
            RIref -= us[-1][Iref:Iref+3][:,None] * vs[-1][None,:]

        # If we pivoted on the reference column, then choose a new reference column.
        # Remember that we are using a x,y,z vector "column" or 
        # set of 3 columns in an algebraic sense.
        if Jref <= Jstar < Jref + 3:
            RJref, Jref = reset_reference_col()
        else:
            # If we didn't change the reference column of the residual matrix "R",
            # update the column to account for the new components of the approximation.
            RJref -= vs[-1][Jref:Jref+3][None,:] * us[-1][:,None]

        # How "large" was this update to the approximation?
        step_size_sq = np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2)

        # Update the Frobenius norm of our approximate matrix. Essentially,
        # how big are the entries?
        appx_frob += step_size_sq
        for j in range(len(us) - 1):
            appx_frob += 2 * us[-1].dot(us[j]) * vs[-1].dot(vs[j])
        if verbose:
            print(
                f'pivot row={Istar:4d}, pivot col={Jstar:4d}, '
                f'step size={step_size_sq:1.3e}, '
                f'approximate matrix frobenius norm={appx_frob:1.3e}'
            )

        # The convergence criteria will be whether the squared "size" of the current
        # update is less than (eps ** 2) times the Frobenius norm of the approximation.
        # This essentially means that the update is small enough that the approximation
        # is within epsilon of the true matrix. 
        if step_size_sq < (eps ** 2) * appx_frob:
            break
            
    # Return the left and right approximation matrices.
    # The approximate is such that:
    # M ~ U_ACA.dot(V_ACA)
    U_ACA = np.array(us).T
    V_ACA = np.array(vs)
    
    return U_ACA, V_ACA
```



```python
U_ACA, V_ACA = ACA_plus(block, 1e-6, verbose=True)
```

A few thoughts on the output here:
1. Clearly, the process is converging. You can see the step size 
1. The set of pivot rows seems quite varied while the set of pivot columns seems concentrated in a few regions (2700-3000, 0-300). Perhaps this just has to do with which elements are closest to the others? I'm not sure!
Clearly, the process is converging! And quite quickly. As we'll see below, the ACA+ algorithm is achieving similar levels of accuracy to the SVD with only 50% more 

```python
y_aca = U_ACA.dot(V_ACA.dot(x))
```

```python
rel_err = np.abs((y_aca - y_true) / y_true)

l2_err = np.sqrt(np.mean(rel_err ** 2))
l1_err = np.mean(rel_err)
linf_err = np.max(rel_err)
frob = np.sum(U_ACA.dot(V_ACA) ** 2)
err_df['ACA'] = [U_ACA.shape[1], l2_err, l1_err, linf_err, frob]
l2_err, l1_err, linf_err
```

```python
terms = []
err = []
for i in range(500):
    U, V = ACA_plus(block)
    terms.append(U.shape[1])
    y_aca = U.dot(V.dot(x))
    rel_err = np.abs((y_aca - y_true) / y_true)
    err.append(np.mean(rel_err))
```

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(terms,bins=15)
plt.xlabel('rank of approximation')
plt.subplot(1,2,2)
plt.hist(np.log10(err), bins=15)
plt.xlabel('$\log_{10}(\|E\|^1)$')
plt.show()
```

## SVD Recompression

```python
UQ, UR = np.linalg.qr(U_ACA)
VQ, VR = np.linalg.qr(V_ACA.T)

W,SIG,Z = np.linalg.svd(UR.dot(VR.T))
```

```python
r = np.where(SIG < 1e-6 * SIG[0])[0][0]
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

```python
eps
```

```python
err_dfT = err_df.T
err_dfT['True Frobenius(A)'] = np.sum(block ** 2)
err_dfT['Frobenius(A) Error'] = np.abs(err_dfT['True Frobenius(A)'] - err_dfT['Frobenius(A)'])
err_dfT
```

```python
def SVD_recompress(U_ACA, V_ACA, eps):
    UQ, UR = np.linalg.qr(U_ACA)
    VQ, VR = np.linalg.qr(V_ACA.T)
    W,SIG,Z = np.linalg.svd(UR.dot(VR.T))
    r = np.where(SIG < 1e-6 * SIG[0])[0][0]
    U = UQ.dot(W[:,:r] * SIG[:r])
    V = Z[:r,:].dot(VQ.T)
    return U, V
```

```python
terms = []
err = []
for i in range(50):
    U, V = SVD_recompress(*ACA_plus(block, 1e-6), 1e-6)
    terms.append(U.shape[1])
    y_aca = U.dot(V.dot(x))
    rel_err = np.abs((y_aca - y_true) / y_true)
    err.append(np.mean(rel_err))
```

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(terms,bins=15)
plt.xlabel('rank of approximation')
plt.subplot(1,2,2)
plt.hist(np.log10(err), bins=15)
plt.xlabel('$\log_{10}(\|E\|^1)$')
plt.show()
```
