from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import scipy.spatial
import sympy as sp


@dataclass()
class PanelSurface:
    # A panel consists of a quadrature rule defined over a subset of a parametrized curve.

    qx: np.ndarray
    qw: np.ndarray
    interp_wts: np.ndarray

    quad_pts: np.ndarray
    quad_wts: np.ndarray
    pts: np.ndarray
    normals: np.ndarray
    jacobians: np.ndarray
    radius: np.ndarray

    panel_bounds: np.ndarray
    panel_edges: np.ndarray
    panel_parameter_width: np.ndarray
    panel_start_idxs: np.ndarray
    panel_sizes: np.ndarray
    panel_centers: np.ndarray
    panel_length: np.ndarray

    def __init__(
        self,
        quad_rule,
        quad_pts,
        quad_wts,
        pts,
        normals,
        jacobians,
        radius,
        panel_bounds,
        panel_edges,
    ):
        self.quad_rule = quad_rule
        self.qx = quad_rule[0]
        self.qw = quad_rule[1]
        self.interp_wts = quad_rule[2]

        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        self.pts = pts
        self.normals = normals
        self.jacobians = jacobians
        self.radius = radius

        # TODO: replace panel_bounds with panel_edges
        self.panel_bounds = panel_bounds

        self.panel_edges = panel_edges
        self.panel_parameter_width = self.panel_bounds[:, 1] - self.panel_bounds[:, 0]
        self.panel_sizes = np.full(self.panel_bounds.shape[0], self.panel_order)
        self.panel_start_idxs = np.cumsum(self.panel_sizes) - self.panel_order
        self.panel_length = np.sum(
            (self.quad_wts).reshape((-1, self.panel_order)), axis=1
        )
        self.panel_centers = (1.0 / self.panel_length[:, None]) * np.sum(
            (self.quad_wts[:, None] * self.pts).reshape((-1, self.panel_order, 2)),
            axis=1,
        )
        self.panel_radius = np.min(self.radius.reshape((-1, self.qx.shape[0])), axis=1)

    @property
    def panel_order(self):
        return self.qx.shape[0]

    @property
    def n_pts(self):
        return self.pts.shape[0]

    @property
    def n_panels(self):
        return self.panel_bounds.shape[0]


def concat_meshes(meshes):
    return PanelSurface(
        (meshes[0].qx, meshes[0].qw, meshes[0].interp_wts),
        np.concatenate([s.quad_pts for s in meshes]),
        np.concatenate([s.quad_wts for s in meshes]),
        np.concatenate([s.pts for s in meshes]),
        np.concatenate([s.normals for s in meshes]),
        np.concatenate([s.jacobians for s in meshes]),
        np.concatenate([s.radius for s in meshes]),
        np.concatenate([s.panel_bounds for s in meshes]),
        None,
    )


def panelize_symbolic_surface(t, x, y, quad_rule, panel_bounds=None, n_panels=None):
    """
    Construct a surface out of a symbolic parametrized curve splitting the curve parameter at
    `panel_bounds` into subcomponents. `panel_bounds` is expected to be a list of ranges of
    the parameter `t` that spans from [-1, 1]. For example:
    `panel_bounds = [(-1,-0.5),(-0.5,0),(0,1)]` would split the surface into three panels extending from
    1. t = -1 to t = -0.5
    2. t=-0.5 to t=0
    3. t=0 to t=1.

    If `panel_bounds` is None, then the `n_panels` parameter is used the evenly
    split the parameter space into panels.
    """
    if panel_bounds is None:
        assert n_panels is not None
        panel_edges = np.linspace(-1, 1, n_panels + 1)
        panel_bounds = np.stack((panel_edges[:-1], panel_edges[1:]), axis=1)

    dxdt = sp.diff(x, t)
    dydt = sp.diff(y, t)

    jacobian = sp.sqrt(dxdt ** 2 + dydt ** 2)

    dx2dt2 = sp.diff(dxdt, t)
    dy2dt2 = sp.diff(dydt, t)

    # A small factor is added to the radius of curvature denominator
    # so that we don't divide by zero when a surface is flat.
    radius = np.abs(jacobian ** 3 / (dxdt * dy2dt2 - dydt * dx2dt2 + 1e-16))

    nx = -dydt / jacobian
    ny = dxdt / jacobian

    quad_pts = []

    panel_parameter_width = panel_bounds[:, 1] - panel_bounds[:, 0]

    qx = quad_rule[0]
    qw = quad_rule[1]
    quad_pts = (
        panel_bounds[:, 0, None]
        + panel_parameter_width[:, None] * (qx[None, :] + 1) * 0.5
    ).flatten()

    surf_vals = [
        symbolic_eval(t, quad_pts, v) for v in [x, y, nx, ny, jacobian, radius]
    ]
    panel_edge_parameters = np.concatenate((panel_bounds[:, 0], panel_bounds[-1:, 1]))
    panel_edges = np.array(
        [symbolic_eval(t, panel_edge_parameters, v) for v in [x, y]]
    ).T

    pts = np.hstack((surf_vals[0][:, None], surf_vals[1][:, None]))
    normals = np.hstack((surf_vals[2][:, None], surf_vals[3][:, None]))
    jacobians = surf_vals[4]
    radius_of_curvature = surf_vals[5]

    quad_wts = (panel_parameter_width[:, None] * qw[None, :] * 0.5).ravel() * jacobians

    return PanelSurface(
        quad_rule,
        quad_pts,
        quad_wts,
        pts,
        normals,
        jacobians,
        radius_of_curvature,
        panel_bounds,
        panel_edges,
    )


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


def refine_surfaces(
    sym_surfs,
    quad_rule,
    other_surfaces=[],
    initial_panels=None,
    max_curvature=0.25,
    control_points=None,
    max_iter=30,
):
    """
    Why does `refine_surfaces` process multiple surfaces at once? Why not just
    process each surface individually? If two surfaces are very close to each
    other and one has panels that are ten times larger than the panels of the
    other surface, then the numerical integration will struggle to compute
    accurate results. As a result, it's important that either the mesh set up
    function handles all the relevant surfaces at once or the user explicitly
    ensures that nearby panels are similar in size.
    """
    n_surfs = len(sym_surfs)

    # cur_panels will track the current refinement level of the panels in each surface
    # The default initial state is that one panels covers the entire curve: [(-1, 1)]
    if initial_panels is None:
        cur_panels = []
        for s in sym_surfs:
            cur_panels.append(np.array([[-1.0, 1.0]]))
    else:
        cur_panels = [ps.copy() for ps in initial_panels]

    # Construct KDtrees from any "other_surfaces" so that we can quickly
    # determine how far away their panels are from our surfaces of interest.
    other_surf_trees = []
    for other_surf in other_surfaces:
        other_surf_trees.append(scipy.spatial.KDTree(other_surf.panel_centers))

    if control_points is not None:
        control_points = np.asarray(control_points)

    # Step 0) Create a PanelSurface from the current set of panels.
    # Note that this step would need to look different if the surface were
    # defined from an input segment geometry rather than from a symbolic
    # curve specification.
    cur_surfs = [
        panelize_symbolic_surface(*sym_surfs[j], quad_rule, cur_panels[j])
        for j in range(len(sym_surfs))
    ]

    for i in range(max_iter):

        # We'll track whether any surface was refined this iteration. If a surface
        # was refined, keep going. Otherwise, we'll exit.
        did_refine = False

        for j in range(n_surfs):
            # Step 1) Refine based on radius of curvature
            # The absolute value
            refine_from_radius = (
                cur_surfs[j].panel_length > max_curvature * cur_surfs[j].panel_radius
            )

            # Step 2) Refine based on a nearby user-specified control points.
            if control_points is not None:
                dist = np.linalg.norm(
                    cur_surfs[j].panel_centers[:, None, :]
                    - control_points[None, :, :2],
                    axis=2,
                )
                # A frequent situation is that a control point will lie exactly on the boundary between two panels. In this edge case, we *do* want to refine both the touching panels. But, floating point error can make this difficult. As a result, I've added a small fudge factor to expand the effect radius of the control point by a small amount.
                fudge_factor = 1.001
                max_dist = (
                    0.5 * cur_surfs[j].panel_length[:, None]
                    + control_points[None, :, 2]
                )
                close = dist <= fudge_factor * max_dist
                refine_from_control = np.sum(
                    close
                    & (cur_surfs[j].panel_length[:, None] > control_points[None, :, 3]),
                    axis=1,
                )
            else:
                refine_from_control = np.zeros(cur_surfs[j].n_panels, dtype=bool)

            # Step 3) Ensure that panel length scale doesn't change too rapidly. This
            # imposes that a panel will be no more than twice the length
            # of any nearby panel, including panels from surfaces provided in the
            # "other_surfaces" argument.
            refine_from_self = np.zeros(cur_surfs[j].n_panels, dtype=bool)
            for k, other_surf in enumerate(other_surfaces + cur_surfs):
                if k == j and cur_surfs[j].n_panels <= 1:
                    continue

                panel_tree = scipy.spatial.KDTree(other_surf.pts)
                nearby_panels = panel_tree.query_ball_point(
                    cur_surfs[j].panel_centers, 1.2 * cur_surfs[j].panel_length
                )

                for panel_idx in range(cur_surfs[j].n_panels):
                    cmp_panels = np.unique(
                        np.array(nearby_panels[panel_idx], dtype=np.int32)
                        // other_surf.panel_order
                    )
                    refine_from_self[panel_idx] |= np.any(
                        cur_surfs[j].panel_length[panel_idx]
                        > 2 * other_surf.panel_length[cmp_panels]
                    )

            refine = refine_from_control | refine_from_radius | refine_from_self
            new_panels = refine_panels(cur_panels[j], refine)

            # TODO: add a callback for debugging? or some logging?
            #         plt.plot(cur_surf.pts[cur_surf.panel_start_idxs,0], cur_surf.pts[cur_surf.panel_start_idxs,1], 'k-*')
            #         plt.show()
            #             print('panel centers', cur_surfs[j].panel_centers)
            #             print('panel length', cur_surfs[j].panel_length)
            #             print('panel radius', panel_radius)
            #             print('control', refine_from_control)
            #             print('radius', refine_from_radius)
            #             print('self', refine_from_self)
            #             print('nearby', refine_from_nearby)
            #             print('overall', refine)
            #             print('')
            #             print('')

            if new_panels.shape[0] == cur_panels[j].shape[0]:
                continue

            did_refine = True
            cur_panels[j] = new_panels

            # Step 5) If
            cur_surfs[j] = panelize_symbolic_surface(
                *sym_surfs[j],
                quad_rule,
                cur_panels[j],
            )

        if not did_refine:
            #             for j in range(n_surfs):
            #                 print(
            #                     f"done after n_iterations={i} with n_panels={cur_panels[j].shape[0]}"
            #                 )
            break

    return cur_surfs[0] if len(cur_surfs) == 1 else cur_surfs


def unit_circle(quad_rule, max_curvature=0.5, control_points=None):
    t = sp.var("t")
    return refine_surfaces(
        [
            (t, sp.cos(sp.pi * t), sp.sin(sp.pi * t)),
        ],
        quad_rule,
        max_curvature=max_curvature,
        control_points=control_points,
    )


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
    return x, w, build_interp_wts(x)


def trapezoidal_rule(n):
    """
    The n-point trapezoidal rule on [-1, 1].
    Returns tuple of (points, weights)
    """
    return np.linspace(-1.0, 1.0, n + 1)[:-1], np.full(n, 2.0 / n), np.ones(n)


def build_interp_wts(x):
    """
    Construct the barycentric interpolation weights for a particular set of
    interpolation points on a non-periodic interval.
    IMPORTANT: this will not work correctly for interpolation on a periodic
    interval. This is important when using a trapezoidal quadrature rule on a
    periodic surface.
    """
    dist = x[:, None] - x[None, :]
    np.fill_diagonal(dist, 1.0)
    weights = 1.0 / np.prod(dist, axis=1)
    return weights


def barycentric_eval(eval_pts, interp_pts, interp_wts, fnc_vals):
    dist = eval_pts[:, None] - interp_pts[None, :]
    kernel = interp_wts[None, :] / dist
    return (kernel.dot(fnc_vals)) / np.sum(kernel, axis=1)


def barycentric_deriv(eval_pts, interp_pts, interp_wts, fnc_vals):
    """
    Evaluate the derivative of a function via barycentric interpolation.
    """
    dist = eval_pts[:, None] - interp_pts[None, :]
    kernel = interp_wts[None, :] / dist
    dkernel = -interp_wts[None, :] / (dist ** 2)
    return (
        np.sum(kernel, axis=1) * dkernel.dot(fnc_vals)
        - kernel.dot(fnc_vals) * np.sum(dkernel, axis=1)
    ) / (np.sum(kernel, axis=1) ** 2)


def build_interp_matrix(in_xhat, in_bary_wts, out_xhat):
    # This code is based on the code in
    # scipy.interpolate.BarycentricInterpolator._evaluate but modified to
    # construct a matrix.
    dist = out_xhat[:, None] - in_xhat

    # Remove zeros so we don't divide by zero.
    z = dist == 0
    dist[z] = 1

    # The barycentric interpolation formula
    dist = in_bary_wts / dist
    interp_matrix = dist / np.sum(dist, axis=-1)[:, None]

    # Handle points where out_xhat is in an entry in interpolator.xi
    r = np.nonzero(z)
    interp_matrix[r[:-1]] = 0
    interp_matrix[r[:-1], r[-1]] = 1.0

    return np.ascontiguousarray(interp_matrix)


def symbolic_eval(t, tvals, e):
    result = sp.lambdify(t, e, "numpy")(tvals)
    if isinstance(result, float) or isinstance(result, int):
        result = np.full_like(tvals, result)
    return result


def build_panel_interp_matrix(in_n_panels, in_qx, in_bary_wts, panel_idxs, out_qx):
    n_out_panels = out_qx.shape[0]
    shape = (n_out_panels * out_qx.shape[1], in_n_panels * in_qx.shape[0])
    indptr = np.arange(n_out_panels + 1)
    indices = panel_idxs
    interp_mat_data = []
    for i in range(n_out_panels):
        single_panel_interp = build_interp_matrix(in_qx, in_bary_wts, out_qx[i])
        interp_mat_data.append(single_panel_interp)
    return scipy.sparse.bsr_matrix((interp_mat_data, indices, indptr), shape)


def apply_interp_mat(mat, interp_mat):
    if mat.ndim == 4:
        reshaped = np.transpose(mat, (0, 1, 3, 2)).reshape((-1, mat.shape[2]))
    else:
        reshaped = mat
    out = scipy.sparse.bsr_matrix.dot(reshaped, interp_mat)
    if mat.ndim == 4:
        return np.transpose(
            out.reshape(
                (mat.shape[0], mat.shape[1], mat.shape[3], interp_mat.shape[1])
            ),
            (0, 1, 3, 2),
        )
    else:
        return out


def upsample(surf, kappa):
    """
    Upsample every panel in surf by a factor of kappa.

    For example, if kappa = 3, and the original quadrature order was 10, then
    every panel will be replaced by a new panel with quadrature order 30.
    """
    stage2_panels = np.empty((surf.n_panels, 3))
    stage2_panels[:, 0] = np.arange(surf.n_panels)
    stage2_panels[:, 1] = -1
    stage2_panels[:, 2] = 1
    src_refined, interp_mat = upsample_per_panel(
        surf, stage2_panels, gauss_rule(surf.panel_order * kappa)
    )
    return src_refined, interp_mat


def upsample_per_panel(surf, panel_spec, quad_rule):
    """
    Upsample panels individually.

    New panels are defined by the panel_spec array where:
    panel_spec[:, 0] is which panel in original surf object corresponds to the new panel
    panel_spec[:, 1] is the left edge in parameter space of the new panel in
        terms of the original panel.
    panel_spec[:, 2] is the right edge in parameter space of the new panel in
        terms of the original panel

    The quad_rule parameter allows specifying the quadrature rule on the new panels.

    The combination of panel_spec and quad_rule allows doing either "h" or "p"
    refinement. For example, specifying two rows of panel_spec as
    [[0, -1, 0], [0, 0, 1]]
    would split the original panel 0 into two new panels that are half as long.
    """
    in_panel_idx = panel_spec[:, 0].astype(int)
    left_param = panel_spec[:, 1][:, None]
    right_param = panel_spec[:, 2][:, None]

    qx = quad_rule[0]
    qw = quad_rule[1]
    out_relative_nodes = (
        left_param + (right_param - left_param) * (qx[None, :] + 1) * 0.5
    )

    interp_mat = build_panel_interp_matrix(
        surf.n_panels,
        surf.qx,
        surf.interp_wts,
        panel_spec[:, 0].astype(int),
        out_relative_nodes,
    )

    quad_pts = (
        surf.panel_bounds[in_panel_idx, 0, None]
        + surf.panel_parameter_width[in_panel_idx, None]
        * (out_relative_nodes + 1)
        * 0.5
    ).ravel()

    pts = interp_mat.dot(surf.pts)
    normals = interp_mat.dot(surf.normals)
    jacobians = interp_mat.dot(surf.jacobians)
    radius = interp_mat.dot(surf.radius)

    panel_bounds = (
        surf.panel_bounds[in_panel_idx, 0, None]
        + (panel_spec[:, 1:] + 1) * 0.5 * surf.panel_parameter_width[in_panel_idx, None]
    )

    quad_wts = (
        (qw[None, :] * 0.25 * (right_param - left_param))
        * surf.panel_parameter_width[in_panel_idx, None]
    ).ravel() * jacobians

    return (
        PanelSurface(
            quad_rule,
            quad_pts,
            quad_wts,
            pts,
            normals,
            jacobians,
            radius,
            panel_bounds,
            None,
        ),
        interp_mat,
    )


def pts_grid(xs, ys):
    """
    Takes two 1D arrays specifying X and Y values and returns a
    (xs.size * ys.size, 2) array specifying the grid of points in the Cartesian
    product of `xs` and `ys`.
    """
    return np.hstack([v.ravel()[:, None] for v in np.meshgrid(xs, ys)])
