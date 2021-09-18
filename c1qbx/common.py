import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate
import sympy as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass()
class Surface:
    """
    A surface consists of:

    `quad_pts` and `quad_wts`: a set of quadrature nodes and weights specifying
        where in [-1, 1] parameter space each interpolation point lies.

    `pts`: a set of interpolation points that define the mesh,

    `normals`: the normal vectors to the surface at the points,

    `jacobians`: the determinant of the curve's jacobian at the points

    """

    quad_pts: np.ndarray
    quad_wts: np.ndarray
    pts: np.ndarray
    normals: np.ndarray
    jacobians: np.ndarray

    @property
    def n_pts(
        self,
    ):
        return self.pts.shape[0]


def circle(n):
    """
    Construct a circular surface with the normal vectors pointing outwards.

    The natural choice for quadrature on a circle is a trapezoidal rule due
    to its exponential convergence on a periodic domain.
    """

    quad_pts, quad_wts = trapezoidal_rule(n)

    # Convert from [-1,1] to [0,2*pi]
    theta = np.pi * (quad_pts + 1)
    pts = np.hstack([np.cos(theta)[:, None], np.sin(theta)[:, None]])
    normals = pts
    jacobians = np.full(pts.shape[0], np.pi)
    return Surface(quad_pts, quad_wts, pts, normals, jacobians)

def discretize_symbolic_surface(quad_pts, quad_wts, t, x, y):
    """
    Given a sympy parameteric expression for the x and y coordinates of a surface, we construct the points and normals and jacobians of that surface.
    `quad_pts`: A 1D array of the quadrature points in the domain [-1, 1].
    `quad_wts`: The weights of the quadrature rule.
    `t`: The parameter of the `x` and `y` expression. Expected to vary in [-1, 1]
    `x` and `y`: The parametric definition of the surface.
    """

    # Both the normal and jacobian will depend on the surface derivatives
    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    jacobian = sp.simplify(sp.sqrt(dxdt ** 2 + dydt ** 2))

    # The normal vector is the normalized derivative vector.
    nx = -dydt / jacobian
    ny = dxdt / jacobian

    # At this point we have the points and normals and jacobians.  But they are
    # all still symbolic! So, we need to evaluate the expressions at the
    # specified quadrature points.
    surf_vals = [symbolic_eval(t, quad_pts, v) for v in [x, y, nx, ny, jacobian]]

    # And create the surface object.
    pts = np.hstack((surf_vals[0][:, None], surf_vals[1][:, None]))
    normals = np.hstack((surf_vals[2][:, None], surf_vals[3][:, None]))
    jacobians = surf_vals[4]

    return Surface(quad_pts, quad_wts, pts, normals, jacobians)

def line(n, xy1, xy2):
    q = gauss_rule(n)
    t = sp.var('t')
    x = xy1[0] + 0.5 * (t + 1) * (xy2[0] - xy1[0])
    y = xy1[1] + 0.5 * (t + 1) * (xy2[1] - xy1[1])
    return discretize_symbolic_surface(*q, t, x, y)


def pts_grid(xs, ys):
    """
    Takes two 1D arrays specifying X and Y values and returns a
    (xs.size * ys.size, 2) array specifying the grid of points in the Cartesian
    product of `xs` and `ys`.
    """
    return np.hstack([v.ravel()[:, None] for v in np.meshgrid(xs, ys)])


def gauss_rule(n):
    """
    The n-point gauss quadrature rule on [-1, 1].
    Returns tuple of (points, weights)
    """
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0 : (n - 1)] = k / np.sqrt(4 * k * k - 1)  # noqa: E203
    x, V = scipy.linalg.eig_banded(a_band, lower=True)
    w = 2 * np.real(np.power(V[0, :], 2))
    return x, w


def trapezoidal_rule(n):
    """
    The n-point trapezoidal rule on [-1, 1].
    Returns tuple of (points, weights)
    """
    return np.linspace(-1.0, 1.0, n + 1)[:-1], np.full(n, 2.0 / n)


def barycentric_eval(eval_pts, interp_pts, interp_wts, fnc_vals):
    dist = eval_pts[:, None] - interp_pts[None, :]
    kernel = interp_wts[None, :] / dist
    return (kernel.dot(fnc_vals)) / np.sum(kernel, axis=1)


def build_interpolator(in_xhat):
    """
    Interpolate the function f(in_xhat) at the values f(out_xhat).

    `f`: An array consisting of f(in_xhat[i])
    `in_xhat`: The function inputs for which values are known already.
    `out_xhat`: The function inputs for which values are desired.

    Note that the in_xhat ordering is randomly permuted. This is a simple trick
    to improve numerical stability. A PR has been merged to scipy to implement
    this permutation within scipy.interpolate.BarycentricInterpolator but the new
    functionality has not yet been released. The Cinv scaling is also included
    in the PR.
    """
    permutation = np.random.permutation(in_xhat.shape[0])
    permuted_in_xhat = in_xhat[permutation]
    C = (np.max(permuted_in_xhat) - np.min(permuted_in_xhat)) / 4.0
    Cinv = 1.0 / C
    I = scipy.interpolate.BarycentricInterpolator(
        Cinv * permuted_in_xhat, np.zeros_like(in_xhat)
    )
    I.Cinv = Cinv
    I.permutation = permutation
    return I

def interpolate_fnc(interpolator, f, out_xhat):
    interpolator.set_yi(f[interpolator.permutation])
    return interpolator(interpolator.Cinv * out_xhat)

def build_interp_matrix(interpolator, out_xhat):
    # This code is based on the code in
    # scipy.interpolate.BarycentricInterpolator._evaluate but modified to
    # construct a matrix.
    dist = (interpolator.Cinv * out_xhat[:,None]) - interpolator.xi

    # Remove zeros so we don't divide by zero.
    z = (dist == 0)
    dist[z] = 1

    # The barycentric interpolation formula
    dist = interpolator.wi / dist
    interp_matrix = dist / np.sum(dist, axis=-1)[:,None]

    # Handle points where out_xhat is in an entry in interpolator.xi
    r = np.nonzero(z)
    interp_matrix[r[:-1]] = 0
    interp_matrix[r[:-1], r[-1]] = 1.0

    # Invert the permutation so that the matrix can be used without extra knowledge
    inv_permutation = np.empty_like(interpolator.permutation)
    inv_permutation[interpolator.permutation] = np.arange(inv_permutation.shape[0])
    return interp_matrix[:, inv_permutation]


def interpolate_surface(in_surf, out_quad_pts, out_quad_wts):
    """
    Interpolate each component of a surface: the pts, normals and jacobians
    """
    interpolator = build_interpolator(in_surf.quad_pts)
    return Surface(
        out_quad_pts,
        out_quad_wts,
        interp_fnc(interpolator, in_surf.pts, out_quad_pts),
        interp_fnc(interpolator, in_surf.normals, out_quad_pts),
        interp_fnc(interpolator, in_surf.jacobians, out_quad_pts),
    )


def symbolic_eval(t, tvals, e):
    result = sp.lambdify(t, e, "numpy")(tvals)
    if isinstance(result, float) or isinstance(result, int):
        result = np.full_like(tvals, result)
    return result


def single_layer_matrix(source, obs_pts):

    dx = obs_pts[:, 0, None] - source.pts[None, :, 0]
    dy = obs_pts[:, 1, None] - source.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2
    G = (1.0 / (4 * np.pi)) * np.log(r2)

    return (G * source.jacobians * source.quad_wts[None, :])[:, None, :]


def double_layer_matrix(source, obs_pts):
    """
    Compute the entries of the matrix that forms the double layer potential.
    """
    dx = obs_pts[:, 0, None] - source.pts[None, :, 0]
    dy = obs_pts[:, 1, None] - source.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2

    # The double layer potential
    integrand = (
        -1.0
        / (2 * np.pi * r2)
        * (dx * source.normals[None, :, 0] + dy * source.normals[None, :, 1])
    )

    return (integrand * source.jacobians * source.quad_wts[None, :])[:, None, :]


def adjoint_double_layer_matrix(source, obs_pts):
    dx = obs_pts[:, None,0] - source.pts[None, :,0]
    dy = obs_pts[:, None,1] - source.pts[None, :,1]
    r2 = dx ** 2 + dy ** 2

    out = np.empty((obs_pts.shape[0], 2, source.n_pts))
    out[:, 0, :] = dx
    out[:, 1, :] = dy

    C = -1.0 / (2 * np.pi * r2)

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (source.jacobians * source.quad_wts[None, :]))[:, None, :]


def hypersingular_matrix(source, obs_pts):
    dx = obs_pts[:, 0, None] - source.pts[None, :, 0]
    dy = obs_pts[:, 1, None] - source.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2

    A = 2 * (dx * source.normals[None, :, 0] + dy * source.normals[None, :, 1]) / r2
    C = 1.0 / (2 * np.pi * r2)
    out = np.empty((obs_pts.shape[0], 2, source.n_pts))

    # The definition of the hypersingular kernel.
    # unscaled sigma_xz component
    out[:, 0, :] = source.normals[None, :, 0] - A * dx
    # unscaled sigma_xz component
    out[:, 1, :] = source.normals[None, :, 1] - A * dy

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (source.jacobians * source.quad_wts[None, :]))[:, None, :]


@dataclass()
class QBXExpansions:
    """
    There are three main operations involving QBX expansions:
    1. Setup via `qbx_setup`: This identifies the proper locations for the expansions.
    2. Expansions via `qbx_expand_matrix`: Given a kernel and source surface,
       construct a matrix that computes the coefficients of the power series from
       the source density field.
    3. Evaluation via `qbx_eval_matrix` and `qbx_interior_eval_matrix`: Constructing
       a matrix that computes integral values at observation points given an already
       computed expansion.

    This might make more sense in reference to a simple Taylor series:
    f(x_i) = C_0 + C_1(x_i - x0) + C_2(x_i - x0)^2 + ...

    1. "Setup", is choosing x0.
    2. "Expansion" is calculating the coefficients C_j
    3. "Evaluation" is calculating f(x_i)

    A QBX power series expansion is defined in reference to an underlying
    source surface.

    `pts`: The expansion centers that are offset from the source surface.

    `r`: The distance from the source surface to the expansion center.

    `p`: The whole set of expansions has a constant `p` which is the
        number of terms in the expansion (in principle, there is no reason that the
        order could not vary from expansion to expansion, but keeping it constant
        simplifies the implementation.)
    """

    pts: np.ndarray
    r: np.ndarray
    p: int

    @property
    def N(self):
        return self.pts.shape[0]


def qbx_setup(source, mult=5.0, direction=0, p=5, r = None):
    """
    Set up the expansion centers for a source surface. The centers will be
    offset by a distance proportional to the local jacobian of the surface.

    By default, two sets of expansion centers will be created: one set on each
    side of the source surface.  The expansion center will be offset from the
    surface in the direction of `surface.normals`.

    `source`: The source surface over which integrals are computed.

    `mult`: A multiplier for the offset.

    `direction`: The direction to offset the expansion centers. By default this
    is zero and as a result, expansion centers will be created on both sides of
    the source surface.

    `p`: The order of QBX expansion.
    """

    # We want the expansion to be further away when the surface points are far
    # from each other and closer when the surface points are close to each
    # other. Scaling by the local jacobian divided by the number of points
    # achieves this. The factor of 0.5 comes in because the quadrature domain
    # is [-1, 1] which has a total length of 2.0 and our goal is essentially to
    # determine how much of that domain each point consumes.
    #
    # NOTE: This is just a heuristic and more complex ways of choosing r might
    # be justified in more complex situations.
    if r is None:
        r = mult * source.jacobians / (source.n_pts / 2.0)

    if direction == 0:
        centers1 = source.pts + r[:, None] * source.normals
        centers2 = source.pts - r[:, None] * source.normals
        centers = np.concatenate((centers1, centers2))
        r = np.concatenate((r, r))
    else:
        centers = source.pts + direction * r[:, None] * source.normals

    return QBXExpansions(centers, r, p)


def qbx_expand_matrix(kernel, source, expansions):
    """
    Given a kernel and source surface, construct a matrix that computes the
    coefficients of the QBX power series from the source density field.

    `kernel`: The kernel function in the integrals.

    `source`: The source surface over which integrals are computed.

    `expansions`: The location of QBX expansions.
    """

    # We'll compute for many expansion centers at once.  There will be one
    # expansion center for each point on the input source surface.  We'll also
    # compute the matrix form so that we can apply it multiply times for
    # different source functions.

    # Construct the quadrature points on the circles surrounding each expansion.
    qbx_nq = 2 * expansions.p + 1
    qbx_qx, qbx_qw = trapezoidal_rule(qbx_nq)
    qbx_qw *= np.pi
    qbx_theta = np.pi * (qbx_qx + 1)

    # The evaluation radius for each center. Ideally the radius is
    # small enough that all the points on the boundary of the expansion circle
    # are still far enough from the source surface.
    qbx_eval_r = expansions.r * 0.5

    # The points where we evaluate source integrals! This will have a shape:
    # (expansions.N, qbx_nq, 2)
    qbx_eval_pts = np.tile(
        expansions.pts.copy()[:, None, :], (1, qbx_theta.shape[0], 1)
    )
    qbx_eval_pts[:, :, 0] += qbx_eval_r[:, None] * np.cos(qbx_theta)[None, :]
    qbx_eval_pts[:, :, 1] += qbx_eval_r[:, None] * np.sin(qbx_theta)[None, :]

    # Evaluate the integrals!
    Keval = kernel(source, qbx_eval_pts.reshape((-1, 2)))
    kernel_ndim = Keval.shape[1]
    qbx_u_matrix = Keval.reshape((expansions.N, qbx_nq, kernel_ndim, source.n_pts))

    # Compute the expansion coefficients in matrix form.
    alpha = np.empty(
        (expansions.pts.shape[0], kernel_ndim, expansions.p, source.n_pts),
        dtype=np.complex128,
    )
    for L in range(expansions.p):
        C = 1.0 / (np.pi * (qbx_eval_r ** L))
        if L == 0:
            C /= 2.0
        oscillatory = qbx_qw[None, :, None] * np.exp(-1j * L * qbx_theta)[None, :, None]
        alpha[:, :, L, :] = C[:, None, None] * np.sum(
            qbx_u_matrix * oscillatory[:, :, None], axis=1
        )
    return alpha


def qbx_eval_matrix(obs_pts_per_expansion, expansions):
    """
    Construct a matrix evaluating the QBX integrals from `expansions` to `obs_pts`.

    `obs_pts_per_expansion`: an array of observation points for each expansion. Expected to
    have a shape like (M, expansions.N).

    `expansions`: The QBX expansions.
    """

    obs_complex = obs_pts_per_expansion[:, :, 0] + obs_pts_per_expansion[:, :, 1] * 1j
    qbx_center = expansions.pts[:, 0] + expansions.pts[:, 1] * 1j
    sep = obs_complex - qbx_center[None, :]
    out = np.empty(
        (obs_pts_per_expansion.shape[0], obs_pts_per_expansion.shape[1], expansions.p),
        dtype=np.complex,
    )
    for L in range(expansions.p):
        out[:, :, L] = sep ** L
    return out


@dataclass()
class QBXInteriorEval:
    matrix: np.ndarray
    centers_used: np.ndarray
    obs_pt_idxs: np.ndarray


def qbx_interior_eval_matrix(
    kernel,
    source,
    obs_pts,
    expansions,
):
    """
    This function identifies which expansion center to use for which
    observation points by finding which expansion is closest. And then we
    construct a QBX evaluation matrix to perform the evaluation.
    """

    # Build a KDTree for doing nearest neighbor searches amongst the QBX centers
    centers_tree = scipy.spatial.cKDTree(expansions.pts)

    # And also for doing nearest neighbor searches on the source surface.
    source_tree = scipy.spatial.cKDTree(source.pts)

    # Identify the distance to the closest expansion, which expansion that is,
    # and the distance to the source surface.
    dist_to_expansion, closest_expansion = centers_tree.query(obs_pts)
    dist_to_source, _ = source_tree.query(obs_pts)

    # Only use QBX if point is close enough to the surface or the point is
    # close enough to its respective QBX expansion center To measure "close
    # enough", we use r, which is the distance from the surface.
    # Why do we use "OR" here?
    # 1) If a point is near an expansion, we should use the expansion
    # regardless of the source surface. This maintains the "global QBX"
    # property.
    # 2) If a point is near a surface, we cannot use a naive
    # integration so we must use the closest expansion point available.
    # TODO: Another task here is to decide which side of a surface the
    # observation point and the expansion are on. It's important to only use
    # expansions on the same side as the observation point. On the other hand,
    # I'm not sure how important or relevant this issue is.
    use_qbx = (dist_to_expansion <= expansions.r[closest_expansion]) | (
        dist_to_source <= expansions.r[closest_expansion]
    )

    # And we identify which expansion centers are ever used, and how many times.
    centers_used, center_counts = np.unique(
        closest_expansion[use_qbx], return_counts=True
    )

    # This part is slightly complex. The vectorization in qbx_eval_matrix means
    # that for each QBX center, we need to compute the same number of
    # observation points. So, we find the maximum number of observation points
    # for any expansion center. eval_pts is going to be the list of points
    # for each expansion center obs_pt_idxs is a mapping back to which indices
    # those points correspond to in the original obsx and obsy input arrays.
    # Because some expansion centers won't use the full n_max_per_qbx_center
    # observation points, obs_pt_idxs equals -1 by default. This will be used
    # later to identify which entries are valid and which are just
    # "vectorization junk".
    n_max_per_center = np.max(center_counts)
    eval_pts = np.zeros((n_max_per_center, centers_used.shape[0], 2))
    obs_pt_idxs = np.full((n_max_per_center, centers_used.shape[0]), -1, dtype=np.int)

    for (i, c) in enumerate(centers_used):
        # So, for each QBX center, we find the observation points that use it.
        idxs = np.where((closest_expansion == c) & use_qbx)[0]
        obs_pt_idxs[: idxs.shape[0], i] = idxs
        eval_pts[: idxs.shape[0], i] = obs_pts[obs_pt_idxs[: idxs.shape[0], i], :]

    # This is the matrix that maps from QBX coeffs to observation point
    Q = qbx_eval_matrix(
        eval_pts,
        QBXExpansions(
            expansions.pts[centers_used],
            expansions.r[centers_used],
            expansions.p,
        ),
    )
    return QBXInteriorEval(Q, centers_used, obs_pt_idxs)


def qbx_matrix(kernel, source, obs_pts, expansions):
    expand = qbx_expand_matrix(kernel, source, expansions)
    interior = qbx_interior_eval_matrix(kernel, source, obs_pts, expansions)

    out_for_qbx_points = np.sum(
        np.real(
            interior.matrix[:, :, None, :, None]
            * expand[interior.centers_used][None, :, :]
        ),
        axis=3,
    )

    entries_used = interior.obs_pt_idxs >= 0


    out = np.empty((obs_pts.shape[0], expand.shape[1], source.n_pts))

    # Which observation points used QBX? Use the QBX results for those!
    obs_pt_qbx = interior.obs_pt_idxs[entries_used]
    out[obs_pt_qbx] = out_for_qbx_points[entries_used]

    # Which observations did not need QBX? Use a naive integrator for those by
    # calling the kernel directly!
    obs_pt_not_qbx = np.setdiff1d(np.arange(obs_pts.shape[0]), obs_pt_qbx)
    out[obs_pt_not_qbx] = kernel(source, obs_pts[obs_pt_not_qbx])

    return out


def qbx_self_interaction_matrix(kernel, obs_surface, src_surface, expansions):
    expand_mat = qbx_expand_matrix(kernel, src_surface, expansions)
    eval_mat = qbx_eval_matrix(obs_surface.pts[None,:], expansions)[0]
    I = np.real(np.sum(eval_mat[:, None, :, None] * expand_mat, axis=2))
    return I, expand_mat, eval_mat


@dataclass()
class PanelSurface:
    # Hierarchy: Boundary -> Element -> Panel -> Point
    # A boundary consists of several segments.
    # An element consists of a single parametrized curve that might be composed of several panels.
    # A panel consists of a quadrature rule defined over a subset of a parametrized curve.
    quad_pts: np.ndarray
    quad_wts: np.ndarray
    pts: np.ndarray
    normals: np.ndarray
    jacobians: np.ndarray
    radius: np.ndarray
    panel_bounds: np.ndarray
    panel_start_idxs: np.ndarray
    panel_sizes: np.ndarray
    panel_centers: np.ndarray
    panel_length: np.ndarray

    @property
    def n_pts(self):
        return self.pts.shape[0]

    @property
    def n_panels(self):
        return self.panel_bounds.shape[0]


def panelize_symbolic_surface(t, x, y, panel_bounds, qx, qw):
    """
    Construct a surface out of a symbolic parametrized curve splitting the curve parameter at
    `panel_bounds` into subcomponent. `panel_bounds` is expected to be a list of ranges of
    the parameter `t` that spans from [-1, 1]. For example:
    `panel_bounds = [(-1,-0.5),(-0.5,0),(0,1)]` would split the surface into three panels extending from
    1. t = -1 to t = -0.5
    2. t=-0.5 to t=0
    3. t=0 to t=1.
    """
    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    jacobian = sp.sqrt(dxdt ** 2 + dydt ** 2)
    
    dx2dt2 = sp.diff(dxdt, t)
    dy2dt2 = sp.diff(dydt, t)
    # A small factor is added to the radius of curvature denominator 
    # so that we don't divide by zero when a surface is flat.
    radius = jacobian ** 3 / (dxdt * dy2dt2 - dydt * dx2dt2 + 1e-16)

    nx = -dydt / jacobian
    ny = dxdt / jacobian

    quad_pts = []

    panel_parameter_width = panel_bounds[:, 1] - panel_bounds[:, 0]

    quad_pts = (
        panel_bounds[:, 0, None] + panel_parameter_width[:, None] * (qx[None, :] + 1) * 0.5
    ).flatten()
    quad_wts = (panel_parameter_width[:, None] * qw[None, :] * 0.5).flatten()
    surf_vals = [symbolic_eval(t, quad_pts, v) for v in [x, y, nx, ny, jacobian, radius]]

    # And create the surface object.
    pts = np.hstack((surf_vals[0][:, None], surf_vals[1][:, None]))
    normals = np.hstack((surf_vals[2][:, None], surf_vals[3][:, None]))
    jacobians = surf_vals[4]
    radius_of_curvature = surf_vals[5]

    panel_sizes = np.full(panel_bounds.shape[0], qx.shape[0])
    panel_start_idxs = np.cumsum(panel_sizes) - qx.shape[0]
    panel_centers = (1.0 / panel_parameter_width[:, None]) * np.sum(
        (quad_wts[:,None] * pts).reshape((-1, qx.shape[0], 2)), 
        axis=1
    )
    panel_length = np.sum((quad_wts * jacobians).reshape((-1, qx.shape[0])), axis=1)
    
    return PanelSurface(
        quad_pts,
        quad_wts,
        pts,
        normals,
        jacobians,
        radius_of_curvature,
        panel_bounds,
        panel_start_idxs,
        panel_sizes,
        panel_centers,
        panel_length
    )


def build_panel_interp_matrix(surface_in, surface_out):
    """
    Construct a matrix interpolating the values of some function from surface_in
    to surface_out. The function assumes that underlying surface is the same
    between both surfaces and that the number of panels is the same. The only
    difference is the order of approximation on each panel.
    """
    out = np.zeros((surface_out.n_pts, surface_in.n_pts))
    for i in range(surface_in.n_panels):
        n_in = surface_in.panel_sizes[i]
        start_in = surface_in.panel_start_idxs[i]
        end_in = start_in + n_in

        n_out = surface_out.panel_sizes[i]
        start_out = surface_out.panel_start_idxs[i]
        end_out = start_out + n_out

        in_quad_pts = surface_in.quad_pts[start_in:end_in]
        out_quad_pts = surface_out.quad_pts[start_out:end_out]
        chunk = build_interp_matrix(build_interpolator(in_quad_pts), out_quad_pts)
        out[start_out:end_out, start_in:end_in] = chunk
    return out

def qbx_panel_setup(source, mult=1.0, direction=0, p=5):
    r = (
        np.repeat((source.panel_bounds[:, 1] - source.panel_bounds[:, 0]), source.panel_sizes)
        * source.jacobians
        * (0.5 * mult)
    )
    return qbx_setup(source, direction=direction, r=r, p=p)
