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
%load_ext autoreload
%autoreload 2
```

# More quadrature by expansion (QBX) examples for the Laplace equation: fun with screw dislocations

## Antiplane shear

A basic result in linear elastic earthquake modeling is the representation of displacement from slip on an infinitely long strike-slip fault. Because the fault is infinitely long, all displacements is fault parallel. Suppose we're looking at a cross-section in the $x,y$ plane with all displacement occuring in the $z$ direction. Then the displacement vector is $\mathbf{u} = (0, 0, u_z)$. And the strain state is:
\begin{equation}
\boldsymbol{\varepsilon} = \begin{bmatrix}
0 & 0 & \epsilon_{13} \\
0 & 0 & \epsilon_{23}\\
 \epsilon_{13}    &    \epsilon_{23}      & 0\end{bmatrix}
\end{equation}
 
This special state is called "antiplane shear". We can simplify the equations of linear elasticity to be in terms of a vector strain, $\epsilon_z = (\epsilon_{xz}, \epsilon_{yz})$ and vector stress, $\sigma_z = (2\mu\epsilon_{xz}, 2\mu\epsilon_{yz})$. Combined with Newtons law, we get the result that $u_z$ is a solution to the Laplace equation:

\begin{equation}
\nabla^2 u_z = 0
\end{equation}

As a result, we can describe the elastic behavior of infinitely long strike-slip faults (aka a "screw dislocation") in terms of solutions to the Laplace equation. 

Below, we're going to use QBX to compute the displacements and stresses resulting from slip on infinitely long strike-slip faults with fun shapes. In particular, the "double layer" integral we computed in part 1 will compute displacement in the volume from the input slip on the fault. We'll also introduce the "hypersingular" integral to calculate stresses from slip.

```{code-cell} ipython3
import warnings
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format='retina'
```

I've moved the functions we wrote in the previous part into `common.py`. We'll import them!

```{code-cell} ipython3
import common
```

## Barycentric Lagrange interpolation

+++

When we compute a boundary integral, there are two sources of error: the surface approximation error and the quadrature error. We've been focusing so far on the quadrature error because it can be reduced dramatically with better algorithms, especially in the singular or near-singular case. The surface approximation error is handled simply through using a higher resolution approximation to the surface -- for example, represent a circle with 100 points instead of 50. However, below, it will be nice to be able to hold the surface approximation error constant while reducing the quadrature error to zero. But, in the integration techniques we have been using, the quadrature error and the surface error are inextricably linked. When we increase from using 50 to 100 points to integrate a function over a circle, we have been improving both the surface approximation and also using a more accurate quadrature rule. 

To separate the two components, we'll interpolate points from a low order surface approximation in order to calculate the locations of quadrature points for a higher order integral approximation. To make the difference more concrete... Before, we would calculate new point locations for a circle by calculating $(cos \theta, sin \theta)$. Now, we will calculate the new point from a polynomial interpolation of the $n$ existing points $\sum_{i}^n c_i p_i(x)$. In some sense, this is also more realistic. In a real-world application, we normally have a data-derived surface representation that we can't improve. On the other hand, even in that real world setting, we *can* add more quadrature points by interpolating on that surface. But adding more quadrature points won't make the surface itself any more accurate.

To do this, it's going to be helpful to have some functions for polynomial interpolation! We'll use the `scipy.interpolate.BarycentricInterpolator` implementation of [barycentric Lagrange interpolation](https://people.maths.ox.ac.uk/trefethen/barycentric.pdf){cite:p}`Berrut2004`. I strongly recommend that paper if you've never run into barycentric Lagrange interpolation before!

+++

Below is a little check to make sure our interpolation snippet is working correctly. We interpolate $sin(5x)$ (plotted with a solid black line) on a grid of 7 points (blue dots) and plot the resulting approximate function (red dashes line). This isn't a rigorous check, but it seems like it's working! Convergence is very fast if we increase the interpolation order, but I've left out a demonstration of that.

```{code-cell} ipython3
from scipy.interpolate import BarycentricInterpolator

# First, form the interpolating polynomial
qx, _ = common.gauss_rule(7)
fqx = np.sin(5 * qx)
I = BarycentricInterpolator(qx, fqx)

# Then, evaluate the polynomial at a bunch of points for plotting.
xs = np.linspace(-1, 1, 200)
v = I(xs)

plt.plot(qx, fqx, "bo", markersize=10)
plt.plot(xs, v, "r--")
plt.plot(xs, np.sin(5 * xs), "k-")
plt.show()
```

Let's move this into a couple helper functions for interpolating functions and surface.

```{code-cell} ipython3
def interp_fnc(f, in_xhat, out_xhat):
    I = BarycentricInterpolator(in_xhat, f)
    return I(out_xhat)


def interp_surface(in_surf, in_xhat, out_xhat):
    out = []
    # So far, we've defined surfaces as five element tuples consisting of:
    # (x, y, normal_x, normal_y, jacobian)
    for f in in_surf[:5]:
        out.append(interp_fnc(f, in_xhat, out_xhat))
    return out
```

## Hypersingular stress integrals

+++

The second missing piece is a set of tools for computing stresses in the volume given an input fault slip. Continuing in the antiplane strain setting, what we want is to compute the gradient of displacement times the shear modulus. 

\begin{equation}
(\sigma_{xz}, \sigma_{yz}) = (\mu \frac{\partial \phi}{\partial x}, \mu \frac{\partial \phi}{\partial y})
\end{equation}

To simplify things a bit here, I'm going to just compute the first compute $\sigma_{xz}$. This is equivalent to computing a traction with the normal vector equal to $(1, 0)$.

The hypersingular integral will computes $\sigma_{xz}$ for us given the source slip distribution. Since, we already built all the components of a QBX algorithm for the double layer case, we can now just write the naive integrator for a new kernel and everything works perfectly. In the cell below, I've implemented a naive integrator for the hypersingular integral. 

```{margin}
As a reminder, By "naive integrator", I just mean the non-QBX integration function that would be the equivalent of the `double_layer_matrix` function from the previous section. 
```

Why is this kernel called "hypersingular"? Because the kernel behaves like $O(\frac{1}{r^2})$ in 2D. (Add a foot note on weakly singular vs strongly singular vs hypersingular.). This makes the integral especially difficult for many traditional integration methods. As you'll see below, this is not a barrier for QBX and we are able to calculate the integral extremely accurately even right on the surface.

```{code-cell} ipython3
def hypersingular_matrix(surface, quad_rule, obsx, obsy):
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    dx = obsx[:, None] - srcx[None, :]
    dy = obsy[:, None] - srcy[None, :]
    r2 = dx ** 2 + dy ** 2

    A = 2 * (dx * srcnx[None, :] + dy * srcny[None, :]) / r2
    C = 1.0 / (2 * np.pi * r2)
    out = np.empty((obsx.shape[0], 2, surface[0].shape[0]))

    # The definition of the hypersingular kernel.
    # unscaled sigma_xz component
    out[:, 0, :] = srcnx[None, :] - A * dx
    # unscaled sigma_xz component
    out[:, 1, :] = srcny[None, :] - A * dy

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (curve_jacobian * quad_rule[1][None, :]))[:, None, :]
```

```{margin}
By leaving out the shear modulus, I'm implicitly assuming that $\mu = 1$. You can just imagine that we're solving a nondimensionalized version of the problem. This is quite common because scaling the displacement and stress to lie in a similar range of values can improve the numerical condition of some problems. 
```

Finally, I'll write a pretty big function that is going to produce nice figures for comparing QBX against a naive computation. The function is written to be independent of the surface and kernel function. It also accepts QBX parameters. As a reminder, `offset_mult` is a multiplier for how far off the surface the QBX expansion centers are placed. `kappa` is the upsampling rate in case we want to use a higher order quadrature for computing QBX coefficients than for representing the surface. And `qbx_p` is the order of the power series expansion. Please look through the function! Very little is new compared to section 1.

```{code-cell} ipython3
def interior_eval(
    kernel,
    src_surface,
    src_quad_rule,
    src_slip,
    obsx,
    obsy,
    offset_mult,
    kappa,
    qbx_p,
    visualize_centers=False,
):
    n_qbx = src_surface[0].shape[0] * kappa
    quad_rule_qbx = common.gauss_rule(n_qbx)
    surface_qbx = common.interp_surface(src_surface, src_quad_rule[0], quad_rule_qbx[0])
    slip_qbx = common.interp_fnc(src_slip, src_quad_rule[0], quad_rule_qbx[0])

    # This is new! We'll have two sets of QBX expansion centers on each side
    # of the surface. The direction parameter simply multiplies the surface
    # offset. So, -1 put the expansion the same distance on the other side
    # of the surface.
    qbx_center_x1, qbx_center_y1, qbx_r1 = common.qbx_choose_centers(
        src_surface, src_quad_rule, mult=offset_mult, direction=1.0
    )
    qbx_center_x2, qbx_center_y2, qbx_r2 = common.qbx_choose_centers(
        src_surface, src_quad_rule, mult=offset_mult, direction=-1.0
    )
    qbx_center_x = np.concatenate([qbx_center_x1, qbx_center_x2])
    qbx_center_y = np.concatenate([qbx_center_y1, qbx_center_y2])
    qbx_r = np.concatenate([qbx_r1, qbx_r2])

    if visualize_centers:
        plt.plot(surface_qbx[0], surface_qbx[1], "k-")
        plt.plot(qbx_center_x, qbx_center_y, "r.")
        plt.show()

    Qexpand = common.qbx_expand_matrix(
        kernel,
        surface_qbx,
        quad_rule_qbx,
        qbx_center_x,
        qbx_center_y,
        qbx_r,
        qbx_p=qbx_p,
    )
    qbx_coeffs = Qexpand.dot(slip_qbx)
    out = common.qbx_interior_eval(
        kernel,
        src_surface,
        src_quad_rule,
        src_slip,
        obsx,
        obsy,
        qbx_center_x,
        qbx_center_y,
        qbx_r,
        qbx_coeffs,
    )
    return out


def qbx_example(
    kernel, surface_fnc, n, offset_mult, kappa, qbx_p, vmin=None, vmax=None
):
    def slip_fnc(xhat):
        # This must be zero at the endpoints!
        return np.cos(xhat * np.pi) + 1.0

    quad_rule_low = common.gauss_rule(n)
    surface_low = surface_fnc(quad_rule_low[0])
    slip_low = slip_fnc(surface_low[0])

    nobs = 400
    zoomx = [-1.5, 1.5]
    zoomy = [-1.5, 1.5]
    xs = np.linspace(*zoomx, nobs)
    ys = np.linspace(*zoomy, nobs)
    obsx, obsy = np.meshgrid(xs, ys)

    low_vals = (
        kernel(
            surface=surface_low,
            obsx=obsx.flatten(),
            obsy=obsy.flatten(),
            quad_rule=quad_rule_low,
        )
        .dot(slip_low)[:, 0]
        .reshape(obsx.shape)
    )

    n = 2000
    quad_rule_high = common.gauss_rule(n)
    surface_high = interp_surface(surface_low, quad_rule_low[0], quad_rule_high[0])
    slip_high = interp_fnc(slip_low, quad_rule_low[0], quad_rule_high[0])
    high_vals = (
        kernel(
            surface=surface_high,
            obsx=obsx.flatten(),
            obsy=obsy.flatten(),
            quad_rule=quad_rule_high,
        )
        .dot(slip_high)[:, 0]
        .reshape(obsx.shape)
    )

    qbx_vals = interior_eval(
        kernel,
        surface_low,
        quad_rule_low,
        slip_low,
        obsx,
        obsy,
        offset_mult,
        kappa,
        qbx_p,
        visualize_centers=True,
    )[:, :, 0]

    if vmin is None:
        vmin = -1.0
    if vmax is None:
        vmax = 1.0
    levels = np.linspace(vmin, vmax, 16)

    plt.figure(figsize=(16, 12))
    plt.subplot(2, 3, 1)
    plt.title("Naive solution")
    cntf = plt.contourf(obsx, obsy, low_vals, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        low_vals,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)

    plt.subplot(2, 3, 2)
    plt.title('"Accurate" solution')
    cntf = plt.contourf(obsx, obsy, high_vals, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        high_vals,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.colorbar(cntf)
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)

    plt.subplot(2, 3, 3)
    plt.title("Naive error")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        logerror = np.log10(np.abs(low_vals - high_vals))
    logerror[np.isinf(logerror)] = -12.0
    cntf = plt.contourf(
        obsx, obsy, logerror, levels=np.linspace(-12, 0, 13), extend="both"
    )
    plt.contour(
        obsx,
        obsy,
        logerror,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=np.linspace(-12, 0, 13),
        extend="both",
    )
    cb = plt.colorbar(cntf)
    cb.set_label("$\log_{10}(|\hat{u} - \hat{u}_{\\textrm{naive}}|)$", fontsize=14)
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.tight_layout()

    plt.subplot(2, 3, 4)
    plt.title("QBX solution")
    cntf = plt.contourf(obsx, obsy, qbx_vals, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        qbx_vals,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)

    plt.subplot(2, 3, 5)
    plt.title('"Accurate" solution')
    cntf = plt.contourf(obsx, obsy, high_vals, levels=levels, extend="both")
    plt.contour(
        obsx,
        obsy,
        high_vals,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=levels,
        extend="both",
    )
    plt.colorbar(cntf)
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)

    plt.subplot(2, 3, 6)
    plt.title("QBX error")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        logerror = np.log10(np.abs(qbx_vals - high_vals))
    logerror[np.isinf(logerror)] = -12.0
    cntf = plt.contourf(
        obsx, obsy, logerror, levels=np.linspace(-12, 0, 13), extend="both"
    )
    plt.contour(
        obsx,
        obsy,
        logerror,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        levels=np.linspace(-12, 0, 13),
        extend="both",
    )
    cb = plt.colorbar(cntf)
    cb.set_label("$\log_{10}(|\hat{u} - \hat{u}_{\\textrm{QBX}}|)$", fontsize=14)
    plt.plot(surface_high[0], surface_high[1], "k-", linewidth=1.5)
    plt.xlim(zoomx)
    plt.ylim(zoomy)
    plt.tight_layout()
    plt.show()
```

## Displacement from a line source

For the remainder of this part, we'll compute displacements and stresses for a three simple geometries of increasing difficulty. 

For each plot, there will be a geometry summary that's just a black line showing the location of the "fault" and red dots showing the location of the QBX expansions. 

Then, we'll plot three solutions: the naive solution, a high accuracy naive solution (from 2000 points) and a QBX solution. We'll also plot the $\log_{10}$ error for both the naive and QBX solutions.

```{code-cell} ipython3
def line(q):
    # Remember the surface tuple format is:
    # (coord_x, coord_y, normal_x, normal_y, jacobian)
    return q, 0 * q, 0 * q, np.ones_like(q), np.ones_like(q)


qbx_example(
    common.double_layer_matrix, surface_fnc=line, n=16, offset_mult=5, kappa=3, qbx_p=15
)
```

## Stress from a line source

```{code-cell} ipython3
qbx_example(
    hypersingular_matrix, surface_fnc=line, n=32, offset_mult=5, kappa=5, qbx_p=15
)
```

## Displacement from an arc source

```{code-cell} ipython3
def arc(q):
    t = 0.5 * np.pi * q + 0.5 * np.pi
    x = np.cos(t)
    y = np.sin(t)
    nx = x.copy()
    ny = y.copy()
    y -= np.mean(y)
    return x, y, nx, ny, np.full_like(x, 0.5 * np.pi)


qbx_example(
    common.double_layer_matrix, surface_fnc=arc, n=16, offset_mult=4, kappa=3, qbx_p=20
)
```

## Stress from an arc source

```{code-cell} ipython3
qbx_example(
    hypersingular_matrix, surface_fnc=arc, n=32, offset_mult=4, kappa=5, qbx_p=20
)
```

## Displacement from a challenging wavy source

```{code-cell} ipython3
def wavy(q):
    t = (q + 1) * 2 * np.pi
    x, y = q, np.sin(t)

    dxdt = 1.0
    dydt = np.cos(t)
    ddt_norm = np.sqrt(dxdt ** 2 + dydt ** 2)
    dxdt /= ddt_norm
    dydt /= ddt_norm
    return x, y, dydt, -dxdt, 2 * np.pi * ddt_norm


qbx_example(
    common.double_layer_matrix,
    surface_fnc=wavy,
    n=256,
    offset_mult=2.5,
    kappa=5,
    qbx_p=15,
)
```

## Stress from a challenging wavy source

```{code-cell} ipython3
qbx_example(
    hypersingular_matrix, surface_fnc=wavy, n=256, offset_mult=2.5, kappa=5, qbx_p=15
)
```
