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
    return_report=False
):
    obs_pts = np.asarray(obs_pts, dtype=np.float64)
    if tol is None:
        tol = K.default_tol

    for s in srcs[1:]:
        if np.any(s.qx != srcs[0].qx):
            raise ValueError(
                "All input sources must use the same panel quadrature rule."
            )

    combined_src = concat_meshes(srcs)

    # step 1: construct the farfield matrix!
    mat = K.direct(obs_pts, combined_src)
    ndim = K.obs_dim * K.src_dim
    report = dict(combined_src=combined_src)
    report["srcs"] = srcs
    report["obs_pts"] = obs_pts

    # step 1: figure out which observation points need to use QBX and which need
    # to use nearfield integration
    src_tree = scipy.spatial.KDTree(combined_src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts, workers=-1)
    closest_panel = closest_idx // combined_src.panel_order
    closest_panel_length = combined_src.panel_length[closest_panel]
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
    kronrod_n = max(combined_src.qx.shape[0], 6)
    kronrod_rule = quadpy.c1.gauss_kronrod(kronrod_n)
    kronrod_qx = kronrod_rule.points
    kronrod_qw = kronrod_rule.weights
    gauss_rule = quadpy.c1.gauss_legendre(kronrod_n)
    gauss_qx = gauss_rule.points
    kronrod_qw_gauss = gauss_rule.weights
    np.testing.assert_allclose(gauss_qx, kronrod_qx[1::2], atol=1e-10)

    n_qbx = np.sum(use_qbx)
    report["n_qbx"] = n_qbx
    if n_qbx > 0:
        qbx_obs_pts = obs_pts[use_qbx]
        qbx_src_pt_indices = closest_idx[use_qbx]
        qbx_closest_pts = combined_src.pts[qbx_src_pt_indices]
        qbx_normals = combined_src.normals[qbx_src_pt_indices]
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
            combined_src.n_panels,
            combined_src.panel_order,
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
        if singularities is None:
            singularities = np.zeros(shape=(0, 2))
        singularities = np.asarray(singularities, dtype=np.float64)
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
        Im = build_interp_matrix(
            combined_src.qx, combined_src.interp_wts, np.linspace(-1, 1, n_interp)
        )
        exp_rs = qbx_panel_L * 0.5 * np.abs(direction)
        offset_vector = np.sign(direction[:, None]) * qbx_normals
        exp_centers = qbx_obs_pts + offset_vector * exp_rs[:, None]
        choose_expansion_circles(
            exp_centers,
            exp_rs,
            qbx_obs_pts,
            offset_vector,
            closest_panel[use_qbx].copy(),
            combined_src.pts,
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
                np.random.rand(combined_src.n_panels), combined_src.panel_order * ndim
            )
        else:
            test_density = np.ones(combined_src.n_pts * ndim)

        # step 5: QBX integrals
        # TODO: This could be replaced by a sparse local matrix.
        qbx_mat = np.zeros((qbx_obs_pts.shape[0], combined_src.n_pts, ndim))
        (
            report["p"],
            report["qbx_integration_error"],
            report["qbx_n_subsets"],
        ) = local_qbx_integrals(
            K.name,
            K.parameters,
            qbx_mat,
            qbx_obs_pts,
            combined_src,
            test_density,
            kronrod_qx,
            kronrod_qw,
            kronrod_qw_gauss,
            exp_centers,
            exp_rs,
            K.max_p,
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
        nearfield_integrals(
            K.name,
            K.parameters,
            qbx_mat,
            qbx_obs_pts,
            combined_src,
            combined_src.qx,
            combined_src.qw,
            combined_src.qw,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
            -1.0,
            0.0,
            adaptive=False,
        )

        mat[use_qbx] += np.transpose(
            qbx_mat.reshape(
                (qbx_obs_pts.shape[0], combined_src.n_pts, K.obs_dim, K.src_dim)
            ),
            (0, 2, 1, 3),
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
    if n_nearfield > 0:
        nearfield_obs_pts = obs_pts[use_nearfield]

        obs_tree = scipy.spatial.KDTree(nearfield_obs_pts)
        panel_obs_pts = obs_tree.query_ball_point(
            combined_src.panel_centers, K.d_up * combined_src.panel_length
        )
        panel_obs_pts_starts = np.zeros(combined_src.n_panels + 1, dtype=int)
        panel_obs_pts_starts[1:] = np.cumsum([len(p) for p in panel_obs_pts])
        panel_obs_pts = np.concatenate(panel_obs_pts, dtype=int, casting="unsafe")

        nearfield_mat = np.zeros((nearfield_obs_pts.shape[0], combined_src.n_pts, ndim))
        (
            report["nearfield_n_subsets"],
            report["nearfield_integration_error"],
        ) = nearfield_integrals(
            K.name,
            K.parameters,
            nearfield_mat,
            nearfield_obs_pts,
            combined_src,
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
            K.name,
            K.parameters,
            nearfield_mat,
            nearfield_obs_pts,
            combined_src,
            combined_src.qx,
            combined_src.qw,
            combined_src.qw,
            panel_obs_pts,
            panel_obs_pts_starts,
            -1.0,
            0.0,
            adaptive=False,
        )
        mat[use_nearfield] += np.transpose(
            nearfield_mat.reshape(
                (nearfield_obs_pts.shape[0], combined_src.n_pts, K.obs_dim, K.src_dim)
            ),
            (0, 2, 1, 3),
        )

    mats = []
    col_idx = 0
    for s in srcs:
        mats.append(mat[:, :, col_idx : col_idx + s.n_pts])
        col_idx += s.n_pts
    if len(mats) == 1:
        mats = mats[0]

    if return_report:
        return mats, report
    else:
        return mats
