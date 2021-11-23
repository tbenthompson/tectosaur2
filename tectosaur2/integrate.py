import warnings

import numpy as np
import quadpy
import scipy.spatial

from tectosaur2.mesh import concat_meshes

from ._ext import identify_nearfield_panels, local_qbx_integrals, nearfield_integrals


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

    if singularities is not None:
        singularity_tree = scipy.spatial.KDTree(singularities)

    # step 1: figure out which observation points need to use QBX
    src_tree = scipy.spatial.KDTree(combined_src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
    closest_panel_length = combined_src.panel_length[
        closest_idx // combined_src.panel_order
    ]
    use_qbx = closest_dist < K.d_qbx * closest_panel_length
    use_nearfield = (closest_dist < K.d_up * closest_panel_length) & (~use_qbx)

    # Currently I use a kronrod rule with order one greater than the underlying
    # number of points per panel. This is to avoid the points colliding which
    # makes the code a bit simpler.
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
    kronrod_qx, kronrod_qw, kronrod_qw_gauss

    n_qbx = np.sum(use_qbx)
    report["n_qbx"] = n_qbx
    if n_qbx > 0:
        qbx_obs_pts = obs_pts[use_qbx]
        qbx_src_pt_indices = closest_idx[use_qbx]
        qbx_closest_pts = combined_src.pts[qbx_src_pt_indices]
        qbx_normals = combined_src.normals[qbx_src_pt_indices]
        qbx_panel_L = closest_panel_length[use_qbx]

        # step 3: find expansion centers
        # TODO: account for singularities
        exp_rs = qbx_panel_L * 0.5

        direction_dot = (
            np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1) / exp_rs
        )
        direction = np.sign(direction_dot)
        on_surface = np.abs(direction) < 1e-13
        direction[on_surface] = limit_direction

        for j in range(30):
            which_violations = np.zeros(n_qbx, dtype=bool)
            exp_centers = (
                qbx_obs_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
            )

            dist_to_nearest_panel, nearest_idx = src_tree.query(exp_centers, k=2)
            nearby_surface_ratio = 2.1 if safety_mode else 1.0001
            which_violations = dist_to_nearest_panel[
                :, 1
            ] < nearby_surface_ratio * np.abs(exp_rs)
            nearest_not_owner = np.where(nearest_idx[:, 0] != qbx_src_pt_indices)[0]
            which_violations[nearest_not_owner] = True

            if singularities is not None:
                singularity_dist_ratio = 3.0
                dist_to_singularity, _ = singularity_tree.query(exp_centers)
                which_violations |= (
                    dist_to_singularity <= singularity_dist_ratio * np.abs(exp_rs)
                )

            if not which_violations.any():
                break
            exp_rs[which_violations] *= 0.75

        # TODO: use ckdtree directly via its C++/cython interface to avoid
        # python list construction
        qbx_panel_src_pts = src_tree.query_ball_point(
            exp_centers, K.d_cutoff * qbx_panel_L, return_sorted=True
        )

        (
            qbx_panels,
            qbx_panel_starts,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
        ) = identify_nearfield_panels(
            exp_centers,
            qbx_panel_src_pts,
            combined_src.n_panels,
            combined_src.panel_order,
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
