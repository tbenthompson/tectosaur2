import warnings

import numpy as np
import quadpy
import scipy.spatial

from tectosaur2.mesh import build_interp_matrix, concat_meshes

from ._ext import (
    choose_expansion_circles,
    identify_nearfield_panels,
    local_qbx_integrals,
    nearfield_integrals,
)


class Kernel:
    def __init__(self, d_cutoff=2.0, d_up=4.0, d_qbx=0.5, max_p=50, default_tol=1e-13):
        self.d_cutoff = d_cutoff
        self.d_up = d_up
        self.d_qbx = d_qbx
        self.max_p = max_p
        self.default_tol = default_tol
        if not hasattr(self, "parameters"):
            self.parameters = np.array([], dtype=np.float64)

    def direct(self, obs_pts, src):
        return (
            self.kernel(obs_pts, src.pts, src.normals)
            * src.quad_wts[None, None, :, None]
        )


def integrate_term(
    K,
    obs_pts,
    *srcs,
    limit_direction=1.0,
    tol=None,
    singularities=None,
    safety_mode=False,
    return_report=False,
    farfield="direct"
):
    # STEP 0: Prepare the inputs.
    obs_pts = np.asarray(obs_pts, dtype=np.float64)
    if tol is None:
        tol = K.default_tol

    for s in srcs[1:]:
        if np.any(s.qx != srcs[0].qx):
            raise ValueError(
                "All input sources must use the same panel quadrature rule."
            )

    if singularities is None:
        singularities = np.zeros(shape=(0, 2))
    singularities = np.asarray(singularities, dtype=np.float64)

    # STEP 1: construct the nearfield matrix.
    qbx_nearfield_mat, report = integrate_nearfield(
        K,
        obs_pts,
        concat_meshes(srcs),
        limit_direction,
        tol,
        singularities,
        safety_mode,
    )
    report["srcs"] = srcs
    report["obs_pts"] = obs_pts

    # STEP 2: slice the matrix into its constituent terms.
    # NOTE: this should probably be removed because it doesn't play nicely with
    # using an fmm or hmatrix for the farfield. how would we "slice" the
    # resulting matrix? but this slicing is actually quite necessary for the
    # current way of solving boundary value problems. what to do about this??
    # perhaps it is still possible to slice an hmatrix.
    # I think the answer is to never combine the hmatrices in the first place.
    # The nearfield matrix construction will be combined, but can be separated
    # at a later time.
    mats = []
    col_idx = 0
    for s in srcs:
        # STEP 2a: construct the farfield matrix and combine with the nearfield matrix
        if farfield == "hmatrix":
            # M = HMatrix()
            raise ValueError("Unimplemented")
        elif farfield == "direct":
            M = K.direct(obs_pts, s)
            M += qbx_nearfield_mat[:, :, col_idx : col_idx + s.n_pts, :]
        else:
            raise ValueError("Unsupported farfield acceleration type.")
        mats.append(M)
        col_idx += s.n_pts

    if len(mats) == 1:
        mats = mats[0]
    if return_report:
        return mats, report
    else:
        return mats


def integrate_nearfield(
    K, obs_pts, src, limit_direction, tol, singularities, safety_mode
):
    report = dict()

    # STEP 2: figure out which observation points need to use QBX and which need
    # to use nearfield integration
    src_tree = scipy.spatial.KDTree(src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts, workers=-1)
    closest_panel = closest_idx // src.panel_order
    closest_panel_length = src.panel_length[closest_panel]
    use_qbx = closest_dist < K.d_qbx * closest_panel_length
    use_nearfield = (closest_dist < K.d_up * closest_panel_length) & (~use_qbx)

    # Currently I use a kronrod rule with order one greater than the underlying
    # number of points per panel. This is to avoid the points colliding which
    # makes the code a bit simpler. Also, the underlying number of points per
    # panel provides some information about the smoothness of the integrand.
    #
    # However, using a kronrod rule with the base rule equal to the number of
    # quadrature points per panel would optimize the nearfield/QBX integrals
    # because no interpolation would be necessary unless the accuracy is
    # poor.
    #
    # I also set the minimum order equal to six. Using a low order quadrature
    # rule in the adaptive integration is really slow.
    kronrod_n = max(src.qx.shape[0], 6)
    kronrod_rule = quadpy.c1.gauss_kronrod(kronrod_n)
    kronrod_qx = kronrod_rule.points
    kronrod_qw = kronrod_rule.weights
    gauss_rule = quadpy.c1.gauss_legendre(kronrod_n)
    gauss_qx = gauss_rule.points
    kronrod_qw_gauss = gauss_rule.weights
    np.testing.assert_allclose(gauss_qx, kronrod_qx[1::2], atol=1e-10)

    n_qbx = np.sum(use_qbx)
    report["n_qbx"] = n_qbx
    if n_qbx == 0:
        qbx_entries = []
        qbx_mapped_rows = []
        qbx_cols = []
        precorrect_entries = []
        precorrect_mapped_rows = []
        precorrect_cols = []
    else:
        qbx_obs_pts = obs_pts[use_qbx]
        qbx_src_pt_indices = closest_idx[use_qbx]
        qbx_closest_pts = src.pts[qbx_src_pt_indices]
        qbx_normals = src.normals[qbx_src_pt_indices]
        qbx_panel_L = closest_panel_length[use_qbx]

        # TODO: use ckdtree directly via its C++/cython interface to avoid
        # python list construction
        qbx_panel_src_pts = src_tree.query_ball_point(
            qbx_obs_pts,
            (K.d_cutoff + 0.5) * qbx_panel_L,
            return_sorted=True,
            workers=-1,
        )

        (
            qbx_panels,
            qbx_panel_starts,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
        ) = identify_nearfield_panels(
            n_qbx,
            qbx_panel_src_pts,
            src.n_panels,
            src.panel_order,
        )

        # STEP 3: find expansion centers/radii
        # In most cases, the simple expansion center will be best. This default
        # choice is determined by simply moving away from the nearest source
        # surface in the direction of that source surface's normal.
        #
        # But, sometimes, the resulting expansion center will either be
        # 1) too close to another portion of the source surface.
        # 2) too close to a user-specified singularity.
        # In those
        #
        # TODO: it would be possible to implement a limit_direction='best'
        # option that chooses the side that allows the expansion point to be
        # further from the source surfaces and then returns the side used. then,
        # external knowledge of the integral equation could be used to handle
        # the jump relation and gather the value on the side the user cares
        # about

        # qbx_normals contains the normal vector from the nearest source surface point.
        # First, we need to determine whether the observation point is on the
        # positive or negative side of the source surface.
        direction_dot = np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1)
        direction = np.sign(direction_dot)
        # If the observation point is precisely *on* the source surface, we use
        # the user-specified limit_direction parameter to determine which side
        # of the source surface to expand on.
        on_surface = np.abs(direction) < 1e-13
        direction[on_surface] = limit_direction

        # This section of code identifies the singularities that are near each
        # observation point. These will be necessary to avoid placing expansion
        # centers too close to singularities.
        singularity_safety_ratio = 3.0
        singularity_tree = scipy.spatial.KDTree(singularities)
        nearby_singularities = singularity_tree.query_ball_point(
            qbx_obs_pts, (singularity_safety_ratio + 0.5) * qbx_panel_L, workers=-1
        )
        # We pack the nearby singularity data into an efficient pair of arrays:
        # - for observation point 3, the set of nearby singularities will be
        #   contained in the slice:
        #     start = nearby_singularities_starts[3]
        #     end = nearby_singularities_starts[4]
        #     slice = nearby_singularities[start:end]
        nearby_singularities_starts = np.zeros(n_qbx + 1, dtype=int)
        nearby_singularities_starts[1:] = np.cumsum(
            [len(ns) for ns in nearby_singularities]
        )
        nearby_singularities = np.concatenate(
            nearby_singularities, dtype=int, casting="unsafe"
        )

        n_interp = 30
        Im = build_interp_matrix(src.qx, src.interp_wts, np.linspace(-1, 1, n_interp))
        # Produce a first "default" guess of where the expansion centers should
        # be. The offset distance in the direction of the normal vector will be
        # half the length of the closest panel. Based on the literature, this
        # forms a nice balance between requiring low order quadrature to compute
        # expansion terms while also requiring a fairly small number of expanion
        # terms for good accuracy.
        exp_rs = qbx_panel_L * 0.5 * np.abs(direction)
        offset_vector = np.sign(direction[:, None]) * qbx_normals
        exp_centers = qbx_obs_pts + offset_vector * exp_rs[:, None]

        # Now that we have collected all the relevant directional, singularity
        # and nearfield panel information, we can finally calculate the ideal
        # location for each expansion center.
        choose_expansion_circles(
            exp_centers,
            exp_rs,
            qbx_obs_pts,
            offset_vector,
            closest_panel[use_qbx].copy(),
            src.pts,
            Im,
            qbx_panels,
            qbx_panel_starts,
            singularities,
            nearby_singularities,
            nearby_singularities_starts,
            nearby_safety_ratio=1.5 if safety_mode else 0.9999,
            singularity_safety_ratio=singularity_safety_ratio,
        )

        if safety_mode:
            # The test_density specifies a source density function that will be
            # multiplied by the matrix entries in order to determine the error
            # in the adaptive QBX order choice. This is necessary because the
            # rigorous error bounds are defined in terms of full integration or
            # matrix vector products rather than matrix entries.
            #
            # With safety_mode=False, the test function is all ones. This
            # essentially assumes that we will be integrating a smooth density.
            #
            # With safety_mode=True, the test function is designed so that there
            # will be a step function at the boundaries between panels. This
            # forces the integration to use higher order expansions at those points and
            # results in a matrix that properly integrates density functions
            # that are discontinuous at panel boundaries. (Discontinuities
            # within a panel would be nigh impossible to integrate correctly
            # because the design of a panel inherently assumes that the density
            # is smooth per panel. If you need discontinuities within a panel, I
            # would encourage you to use more low order panels, perhaps even
            # linear panels, N=2.)
            #
            # TODO: ideally, we'd use some sort of graph coloring here but a
            # random value per panel is highly likely to be good enough because
            # it have a step function in the right places
            # TODO: another cool feature here would be to allow the user to pass
            # in a test_density and then automatically identify where the test
            # density has step functions and edit nearby_safety_ratio for those
            # intersections and then use the test_density for computing
            # integration error
            test_density = np.repeat(
                np.random.rand(src.n_panels),
                src.panel_order * K.src_dim,
            )
        else:
            test_density = np.ones(src.n_pts * K.obs_dim * K.src_dim)

        # step 4: QBX integrals
        qbx_entries = np.zeros(
            (qbx_panels.shape[0] * src.panel_order * K.obs_dim * K.src_dim)
        )
        qbx_rows = np.empty_like(qbx_entries, dtype=np.int64)
        qbx_cols = np.empty_like(qbx_entries, dtype=np.int64)
        (
            report["p"],
            report["qbx_integration_error"],
            report["qbx_n_subsets"],
        ) = local_qbx_integrals(
            K,
            qbx_entries,
            qbx_rows,
            qbx_cols,
            qbx_obs_pts,
            src,
            test_density,
            kronrod_qx,
            kronrod_qw,
            kronrod_qw_gauss,
            exp_centers,
            exp_rs,
            tol,
            qbx_panels,
            qbx_panel_starts,
        )

        # The integration_error is the maximum error per observation point from
        # any of the integrals passed to the adaptive quadrature routine.
        report["qbx_integration_failed"] = (
            report["qbx_integration_error"] > tol
        ).astype(bool)
        if np.any(report["qbx_integration_failed"]):
            warnings.warn(
                "Some integrals failed to converge during adaptive integration. "
                "This an indication of a problem in either the integration or the "
                "problem formulation."
            )

        report["max_order_reached"] = report["p"] == K.max_p
        if np.any(report["max_order_reached"]):
            warnings.warn(
                "Some expanded integrals reached maximum expansion order."
                " These integrals may be inaccurate."
            )

        # step 6: subtract off the direct term whenever a QBX integral is used.
        precorrect_entries = np.zeros_like(qbx_entries)
        precorrect_rows = np.zeros_like(qbx_rows)
        precorrect_cols = np.zeros_like(qbx_cols)
        nearfield_integrals(
            K,
            precorrect_entries,
            precorrect_rows,
            precorrect_cols,
            qbx_obs_pts,
            src,
            src.qx,
            src.qw,
            src.qw,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
            -1.0,
            0.0,
            adaptive=False,
        )
        qbx_obs_idx_map = np.arange(obs_pts.shape[0])[use_qbx]
        qbx_mapped_rows = (
            qbx_obs_idx_map[qbx_rows // K.obs_dim] * K.obs_dim + qbx_rows % K.obs_dim
        )
        precorrect_mapped_rows = (
            qbx_obs_idx_map[precorrect_rows // K.obs_dim] * K.obs_dim
            + precorrect_rows % K.obs_dim
        )

        report["use_qbx"] = use_qbx
        report["exp_centers"] = exp_centers
        report["exp_rs"] = exp_rs
        report["closest_src_pts"] = qbx_closest_pts
        report["direction"] = direction
        report["on_surface"] = on_surface

    n_nearfield = np.sum(use_nearfield)
    report["n_nearfield"] = n_nearfield
    report["use_nearfield"] = use_nearfield
    if n_nearfield == 0:
        nearfield_entries = []
        nearfield_mapped_rows = []
        nearfield_cols = []
    else:
        nearfield_obs_pts = obs_pts[use_nearfield]

        obs_tree = scipy.spatial.KDTree(nearfield_obs_pts)
        panel_obs_pts = obs_tree.query_ball_point(
            src.panel_centers, K.d_up * src.panel_length
        )
        panel_obs_pts_starts = np.zeros(src.n_panels + 1, dtype=int)
        panel_obs_pts_starts[1:] = np.cumsum([len(p) for p in panel_obs_pts])
        panel_obs_pts = np.concatenate(panel_obs_pts, dtype=int, casting="unsafe")

        nearfield_entries = np.zeros(
            (panel_obs_pts.shape[0] * src.panel_order * K.obs_dim * K.src_dim)
        )
        nearfield_rows = np.empty_like(nearfield_entries, dtype=np.int64)
        nearfield_cols = np.empty_like(nearfield_entries, dtype=np.int64)
        (
            report["nearfield_n_subsets"],
            report["nearfield_integration_error"],
        ) = nearfield_integrals(
            K,
            nearfield_entries,
            nearfield_rows,
            nearfield_cols,
            nearfield_obs_pts,
            src,
            kronrod_qx,
            kronrod_qw,
            kronrod_qw_gauss,
            panel_obs_pts,
            panel_obs_pts_starts,
            1.0,
            tol,
            adaptive=True,
        )
        report["nearfield_integration_failed"] = (
            report["nearfield_integration_error"] > tol
        ).astype(bool)
        if np.any(report["nearfield_integration_failed"]):
            warnings.warn(
                "Some integrals failed to converge during adaptive integration. "
                "This an indication of a problem in either the integration or the "
                "problem formulation."
            )

        # setting adaptive=False prevents refinement which is what we want to
        # cancel out the direct component terms
        nearfield_integrals(
            K,
            nearfield_entries,
            nearfield_rows,
            nearfield_cols,
            nearfield_obs_pts,
            src,
            src.qx,
            src.qw,
            src.qw,
            panel_obs_pts,
            panel_obs_pts_starts,
            -1.0,
            0.0,
            adaptive=False,
        )
        nearfield_obs_idx_map = np.arange(obs_pts.shape[0])[use_nearfield]
        nearfield_mapped_rows = (
            nearfield_obs_idx_map[nearfield_rows // K.obs_dim] * K.obs_dim
            + nearfield_rows % K.obs_dim
        )

    qbx_nearfield_mat = (
        scipy.sparse.coo_matrix(
            (
                np.concatenate((qbx_entries, precorrect_entries, nearfield_entries)),
                (
                    np.concatenate(
                        (qbx_mapped_rows, precorrect_mapped_rows, nearfield_mapped_rows)
                    ),
                    np.concatenate((qbx_cols, precorrect_cols, nearfield_cols)),
                ),
            ),
            shape=(
                obs_pts.shape[0] * K.obs_dim,
                src.n_pts * K.src_dim,
            ),
        )
        .toarray()
        .reshape((obs_pts.shape[0], K.obs_dim, src.n_pts, K.src_dim))
    )

    return qbx_nearfield_mat, report
