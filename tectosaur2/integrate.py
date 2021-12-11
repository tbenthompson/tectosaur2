import warnings

import numpy as np
import quadpy
import scipy.spatial

from tectosaur2.mesh import build_interp_matrix, build_interpolator, concat_meshes

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

    # step 1: figure out which observation points need to use QBX and which need
    # to use nearfield integration
    src_tree = scipy.spatial.KDTree(combined_src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
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
    kronrod_n = max(combined_src.qx.shape[0] + 1, 6)
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
            qbx_obs_pts, (K.d_cutoff + 0.5) * qbx_panel_L, return_sorted=True
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

        # step 3: find expansion centers
        # TODO: it would be possible to implement a limit_direction='best'
        # option that chooses the side that allows the expansion point to be
        # further from the source surfaces and then returns the side used. then,
        # external knowledge of the integral equation could be used to handle
        # the jump relation and gather the value on the side the user cares
        # about
        direction_dot = np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1)
        direction = np.sign(direction_dot)
        on_surface = np.abs(direction) < 1e-13
        direction[on_surface] = limit_direction

        singularity_safety_ratio = 3.0

        if singularities is None:
            singularities = np.zeros(shape=(0, 2))
        singularities = np.asarray(singularities, dtype=np.float64)
        singularity_tree = scipy.spatial.KDTree(singularities)
        nearby_singularities = singularity_tree.query_ball_point(
            qbx_obs_pts, (singularity_safety_ratio + 0.5) * qbx_panel_L
        )
        nearby_singularities_starts = np.zeros(n_qbx + 1, dtype=int)
        nearby_singularities_starts[1:] = np.cumsum(
            [len(ns) for ns in nearby_singularities]
        )
        nearby_singularities = np.concatenate(
            nearby_singularities, dtype=int, casting="unsafe"
        )

        interpolator = build_interpolator(combined_src.qx)
        n_interp = 30
        Im = build_interp_matrix(interpolator, np.linspace(-1, 1, n_interp))
        exp_rs = qbx_panel_L * 0.5
        offset_vector = direction[:, None] * qbx_normals
        exp_centers = qbx_obs_pts + offset_vector * exp_rs[:, None]
        choose_expansion_circles(
            exp_centers,
            exp_rs,
            qbx_obs_pts,
            offset_vector,
            combined_src.pts,
            Im,
            qbx_panels,
            qbx_panel_starts,
            closest_panel,
            singularities,
            nearby_singularities,
            nearby_singularities_starts,
            nearby_safety_ratio=2.0 if safety_mode else 0.9999,
            singularity_safety_ratio=singularity_safety_ratio,
        )

        # step 5: QBX integrals
        # TODO: This could be replaced by a sparse local matrix.
        qbx_mat = np.zeros((qbx_obs_pts.shape[0], combined_src.n_pts, ndim))
        (
            report["p"],
            report["integration_failed"],
            report["n_subsets"],
        ) = local_qbx_integrals(
            K.name,
            K.parameters,
            qbx_mat,
            qbx_obs_pts,
            combined_src,
            kronrod_qx,
            kronrod_qw,
            kronrod_qw_gauss,
            exp_centers,
            exp_rs,
            K.max_p,
            tol,
            safety_mode,
            qbx_panels,
            qbx_panel_starts,
        )
        if np.any(report["integration_failed"]):
            warnings.warn(
                "Some integrals failed to converge during adaptive integration. "
                "This an indication of a problem in either the integration or the "
                "problem formulation."
            )
        if np.any(report["p"] == K.max_p):
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
        nearfield_integrals(
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
