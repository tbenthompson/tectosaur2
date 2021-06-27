import numpy as np
import scipy.linalg
from scipy.spatial import cKDTree
from scipy.interpolate import BarycentricInterpolator
import sympy as sp
import matplotlib.pyplot as plt

# the n-point gauss quadrature rule on [-1, 1], returns tuple of (points,
# weights)
def gauss_rule(n):
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0 : (n - 1)] = k / np.sqrt(4 * k * k - 1)  # noqa: E203
    x, V = scipy.linalg.eig_banded(a_band, lower=True)
    w = 2 * np.real(np.power(V[0, :], 2))
    return x, w


# the n-point trapezoidal rule on [-1, 1], returns tuple of (points, weights)
def trapezoidal_rule(n):
    return np.linspace(-1.0, 1.0, n + 1)[:-1], np.full(n, 2.0 / n)


# our simple curve functions will return (x, y, nx, ny, jacobian)
# because the input quadrature rule is on the domain [-1, 1], the
# jacobian of the transformation for a circle with radius 1 is
# constant and equal to pi.
def circle(quad_pts):
    theta = np.pi * (quad_pts + 1)
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y, x, y, np.pi


def double_layer_matrix(surface, quad_rule, obsx, obsy):
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    dx = obsx[:, None] - srcx[None, :]
    dy = obsy[:, None] - srcy[None, :]
    r2 = dx ** 2 + dy ** 2

    # The double layer potential
    integrand = -1.0 / (2 * np.pi) * (dx * srcnx[None, :] + dy * srcny[None, :]) / r2

    return (integrand * curve_jacobian * quad_rule[1][None, :])[:, None, :]


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


def qbx_choose_centers(surface, quad_rule, mult=5.0, direction=1.0):
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    # The expansion center will be offset from the surface in the direction of
    # (srcnx, srcny)
    quad_pt_spacing = curve_jacobian * np.full_like(quad_rule[1], np.mean(quad_rule[1]))
    qbx_r = mult * quad_pt_spacing
    center_x = srcx + direction * qbx_r * srcnx
    center_y = srcy + direction * qbx_r * srcny
    return center_x, center_y, qbx_r


def qbx_expand_matrix(kernel, surface, quad_rule, center_x, center_y, qbx_r, qbx_p=5):
    # Instead of computing for a single expansion center, we'll do it for many
    # at once.  There will be one expansion center for each point on the input
    # surface.  We'll also compute the matrix form so that we can apply it
    # multiply times for different source functions.
    srcx, srcy, srcnx, srcny, curve_jacobian = surface

    qbx_nq = 2 * qbx_p + 1
    qbx_qx, qbx_qw = trapezoidal_rule(qbx_nq)
    qbx_qw *= np.pi
    qbx_theta = np.pi * (qbx_qx + 1)

    # The coefficient integral points will have shape (number of expansions,
    # number of quadrature points).
    qbx_eval_r = qbx_r * 0.5
    qbx_x = center_x[:, None] + qbx_eval_r[:, None] * np.cos(qbx_theta)[None, :]
    qbx_y = center_y[:, None] + qbx_eval_r[:, None] * np.sin(qbx_theta)[None, :]

    Keval = kernel(surface, quad_rule, qbx_x.flatten(), qbx_y.flatten())
    kernel_ndim = Keval.shape[1]
    qbx_u_matrix = Keval.reshape((*qbx_x.shape, kernel_ndim, srcx.shape[0]))

    # Compute the expansion coefficients in matrix form.
    alpha = np.empty(
        (center_x.shape[0], kernel_ndim, qbx_p, srcx.shape[0]), dtype=np.complex128
    )
    for L in range(qbx_p):
        C = 1.0 / (np.pi * (qbx_eval_r ** L))
        if L == 0:
            C /= 2.0
        oscillatory = qbx_qw[None, :, None] * np.exp(-1j * L * qbx_theta)[None, :, None]
        alpha[:, :, L, :] = C[:, None, None] * np.sum(
            qbx_u_matrix * oscillatory[:, :, None], axis=1
        )
    return alpha


def qbx_eval_matrix(obsx, obsy, center_x, center_y, qbx_p=5):
    """
    If center_x and center_y, the QBX centers, have shape (n_centers,),
    then obsx and obsy should have shape (n_pts_per_center, n_centers)
    """
    # Construct a matrix that evaluates the QBX expansions. This should look
    # very similar to the single-expansion case above.
    obs_complex = obsx + obsy * 1j
    qbx_center = center_x + center_y * 1j
    sep = obs_complex - qbx_center[None, :]
    out = np.empty((obsx.shape[0], obsx.shape[1], qbx_p), dtype=np.complex)
    for L in range(qbx_p):
        out[:, :, L] = sep ** L
    return out


def qbx_interior_eval(
    kernel,
    surface,
    quad_rule,
    density,
    obsx,
    obsy,
    qbx_center_x,
    qbx_center_y,
    qbx_r,
    qbx_coeffs,
):
    # Build a KDTree for doing nearest neighbor searches amongst the QBX centers
    center_pts = np.array([qbx_center_x, qbx_center_y]).T
    qbx_centers_tree = cKDTree(center_pts)

    # And also for doing nearest neighbor searches on the source surface.
    surface_pts = np.array([surface[0], surface[1]]).T
    surface_tree = cKDTree(surface_pts)

    lookup_pts = np.array([obsx.flatten(), obsy.flatten()]).T

    # Identify the distance to the closest expansion, which expansion that is,
    # and the distance to the surface.
    dist_to_expansion, closest_expansion = qbx_centers_tree.query(lookup_pts)
    dist_to_surface, _ = surface_tree.query(lookup_pts)

    # Only use QBX if point is close enough to the surface and the point is
    # close enough to its respective QBX expansion center To measure "close
    # enough", we use qbx_r, which is the distance from the surface.
    use_qbx = (dist_to_expansion < qbx_r[closest_expansion]) & (
        dist_to_surface < qbx_r[closest_expansion]
    )

    # And we identify which expansion centers are ever used, and how many times.
    qbx_centers_used, center_counts = np.unique(
        closest_expansion[use_qbx], return_counts=True
    )

    # This part is slightly complex. The vectorization in qbx_eval_matrix means
    # that for each QBX center, we need to compute the same number of
    # observation points. So, we find the maximum number of observation points
    # for any expansion center. qbx_eval_pts is going to be the list of points
    # for each expansion center orig_pt_idxs is a mapping back to which indices
    # those points correspond to in the original obsx and obsy input arrays.
    # Because some expansion centers won't use the full n_max_per_qbx_center
    # observation points, orig_pt_idxs equals -1 by default. This will be used
    # later to identify which entries are valid and which are just
    # "vectorization junk".
    n_max_per_qbx_center = np.max(center_counts)
    qbx_eval_pts = np.zeros((n_max_per_qbx_center, qbx_centers_used.shape[0], 2))
    orig_pt_idxs = np.full(
        (n_max_per_qbx_center, qbx_centers_used.shape[0]), -1, dtype=np.int
    )
    for (
        i,
        c,
    ) in enumerate(qbx_centers_used):
        # So, for each QBX center, we find the observation points that use it.
        idxs = np.where((closest_expansion == c) & use_qbx)[0]
        orig_pt_idxs[: idxs.shape[0], i] = idxs
        qbx_eval_pts[: idxs.shape[0], i] = lookup_pts[
            orig_pt_idxs[: idxs.shape[0], i], :
        ]

    # Now, we get to actually computing integrals.  First, compute the brute
    # force integral for every observation point. We'll just overwrite the ones
    # using QBX next.
    out = kernel(
        surface=surface, obsx=obsx.flatten(), obsy=obsy.flatten(), quad_rule=quad_rule
    ).dot(density)

    # This is the matrix that maps from QBX coeffs to observation point
    Q = qbx_eval_matrix(
        qbx_eval_pts[:, :, 0],
        qbx_eval_pts[:, :, 1],
        qbx_center_x[qbx_centers_used],
        qbx_center_y[qbx_centers_used],
        qbx_p=qbx_coeffs.shape[2],
    )

    # And perform a summation over the terms in each QBX. axis=2 is the
    # summation over the l index in the alpha expansion coefficients.
    out_for_qbx_points = np.sum(
        np.real(Q[:, :, None, :] * qbx_coeffs[qbx_centers_used][None, :]), axis=3
    )

    # Finally, use the QBX evaluation where appropriate. If orig_pt_idxs == -1,
    # the entries are vectorization junk.
    out[orig_pt_idxs[orig_pt_idxs >= 0]] = out_for_qbx_points[orig_pt_idxs >= 0]

    return out.reshape(*obsx.shape, out.shape[1])


def interaction_matrix(
    kernel, obs_surface, obs_quad_rule, src_surface, src_quad_rule, qbx_p=5
):
    qbx_center_x, qbx_center_y, qbx_r = qbx_choose_centers(obs_surface, obs_quad_rule)
    qbx_expand = qbx_expand_matrix(
        kernel,
        src_surface,
        src_quad_rule,
        qbx_center_x,
        qbx_center_y,
        qbx_r,
        qbx_p=qbx_p,
    )
    qbx_eval = qbx_eval_matrix(
        obs_surface[0][None, :],
        obs_surface[1][None, :],
        qbx_center_x,
        qbx_center_y,
        qbx_p=qbx_p,
    )[0]
    return (
        np.real(np.sum(qbx_eval[:, None, :, None] * qbx_expand, axis=2)),
        (qbx_center_x, qbx_center_y, qbx_r, qbx_expand, qbx_eval),
    )


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
    quad_rule_qbx=None,
    surface_qbx=None,
    slip_qbx=None,
    qbx_center_x=None,
    qbx_center_y=None,
    qbx_r=None,
    visualize_centers=False,
):
    if quad_rule_qbx is None:
        n_qbx = src_surface[0].shape[0] * kappa
        quad_rule_qbx = gauss_rule(n_qbx)
        surface_qbx = interp_surface(src_surface, src_quad_rule[0], quad_rule_qbx[0])
        slip_qbx = interp_fnc(src_slip, src_quad_rule[0], quad_rule_qbx[0])

    # This is new! We'll have two sets of QBX expansion centers on each side
    # of the surface. The direction parameter simply multiplies the surface
    # offset. So, -1 put the expansion the same distance on the other side
    # of the surface.
    if qbx_center_x is None:
        qbx_center_x1, qbx_center_y1, qbx_r1 = qbx_choose_centers(
            src_surface, src_quad_rule, mult=offset_mult, direction=1.0
        )
        qbx_center_x2, qbx_center_y2, qbx_r2 = qbx_choose_centers(
            src_surface, src_quad_rule, mult=offset_mult, direction=-1.0
        )
        qbx_center_x = np.concatenate([qbx_center_x1, qbx_center_x2])
        qbx_center_y = np.concatenate([qbx_center_y1, qbx_center_y2])
        qbx_r = np.concatenate([qbx_r1, qbx_r2])

    if visualize_centers:
        plt.plot(surface_qbx[0], surface_qbx[1], "k-")
        plt.plot(qbx_center_x, qbx_center_y, "r.")
        plt.show()

    Qexpand = qbx_expand_matrix(
        kernel,
        surface_qbx,
        quad_rule_qbx,
        qbx_center_x,
        qbx_center_y,
        qbx_r,
        qbx_p=qbx_p,
    )
    qbx_coeffs = Qexpand.dot(slip_qbx)
    out = qbx_interior_eval(
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


def barycentric_eval(eval_pts, interp_pts, interp_wts, fnc_vals):
    dist = eval_pts[:, None] - interp_pts[None, :]
    kernel = interp_wts[None, :] / dist
    return (kernel.dot(fnc_vals)) / np.sum(kernel, axis=1)


def interp_fnc(f, in_xhat, out_xhat):
    permutation = np.random.permutation(in_xhat.shape[0])
    permuted_in_xhat = in_xhat[permutation]
    C = (np.max(permuted_in_xhat) - np.min(permuted_in_xhat)) / 4.0
    Cinv = 1.0 / C
    I = BarycentricInterpolator(Cinv * permuted_in_xhat, f[permutation])
    return I(Cinv * out_xhat)


def interp_surface(in_surf, in_xhat, out_xhat):
    out = []
    # So far, we've defined surfaces as five element tuples consisting of:
    # (x, y, normal_x, normal_y, jacobian)
    for f in in_surf[:5]:
        out.append(interp_fnc(f, in_xhat, out_xhat))
    return out


def symbolic_surface(t, x, y):
    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    ddt_norm = sp.simplify(sp.sqrt(dxdt ** 2 + dydt ** 2))
    dxdt /= ddt_norm
    dydt /= ddt_norm

    return x, y, dydt, -dxdt, ddt_norm


def symbolic_eval(t, tvals, exprs):
    out = []
    for e in exprs:
        result = sp.lambdify(t, e, "numpy")(tvals)
        if isinstance(result, float) or isinstance(result, int):
            result = np.full_like(tvals, result)
        out.append(result)
    return out
