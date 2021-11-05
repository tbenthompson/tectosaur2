import warnings

import numpy as np
import scipy.spatial

from tectosaur2.mesh import PanelSurface

from ._ext import identify_nearfield_panels, local_qbx_integrals


def integrate_term(
    K, obs_pts, *srcs, limit_direction=1.0, tol=1e-13, return_reports=False
):
    for s in srcs[1:]:
        if np.any(s.qx != srcs[0].qx):
            raise ValueError(
                "All input sources must use the same panel quadrature rule."
            )

    combined_src = PanelSurface(
        srcs[0].qx,
        srcs[0].qw,
        np.concatenate([s.quad_pts for s in srcs]),
        np.concatenate([s.quad_wts for s in srcs]),
        np.concatenate([s.pts for s in srcs]),
        np.concatenate([s.normals for s in srcs]),
        np.concatenate([s.jacobians for s in srcs]),
        np.concatenate([s.radius for s in srcs]),
        np.concatenate([s.panel_bounds for s in srcs]),
    )

    # step 1: construct the farfield matrix!
    mat = K.direct(obs_pts, combined_src)
    report = dict()

    # step 1: figure out which observation points need to use QBX
    src_tree = scipy.spatial.KDTree(combined_src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
    closest_panel_length = combined_src.panel_length[
        closest_idx // combined_src.panel_order
    ]
    use_qbx = closest_dist < K.d_qbx * closest_panel_length
    use_nearfield = (closest_dist < K.d_up * closest_panel_length) & (~use_qbx)

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
        direction[np.abs(direction) < 1e-13] = limit_direction

        for j in range(30):
            exp_centers = (
                qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
            )
            dist_to_nearest_panel = src_tree.query(exp_centers)[0]
            # TODO: WRITE A TEST THAT HAS VIOLATIONS
            # The fudge factor helps avoid numerical precision issues. For example,
            # when we offset an expansion center 1.0 away from a surface node,
            # without the fudge factor this test will be checking 1.0 < 1.0, but
            # that is fragile in the face of small 1e-15 sized numerical errors.
            # By simply multiplying by 1.0001, we avoid this issue without
            # introducing any other problems.
            fudge_factor = 1.0001
            which_violations = dist_to_nearest_panel * fudge_factor < np.abs(exp_rs)

            if not which_violations.any():
                break
            exp_rs[which_violations] *= 0.75
        # TODO:
        # TODO: Is this important?
        exp_rs *= 0.9

        # TODO: use ckdtree directly via its C++/cython interface to avoid python
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
        qbx_mat = np.zeros((obs_pts.shape[0], combined_src.n_pts, K.ndim))
        report["p"], report["n_subsets"] = local_qbx_integrals(
            K.name,
            qbx_mat,
            obs_pts,
            combined_src,
            exp_centers,
            exp_rs,
            K.max_p,
            tol,
            qbx_panels,
            qbx_panel_starts,
        )
        if np.any(report["n_subsets"] > 100):
            warnings.warn(
                "Some integrals required a surprising amount of adaptive refinement. "
                "This an indication of a problem in either the integration or the "
                "problem formulation."
            )
        if np.any(report["p"] == K.max_p):
            warnings.warn(
                "Some expanded integrals reached maximum expansion order."
                " These integrals may be inaccurate."
            )

        # step 6: subtract off the direct term whenever a QBX integral is used.
        K.nearfield(
            qbx_mat,
            obs_pts,
            combined_src,
            qbx_panel_obs_pts,
            qbx_panel_obs_pt_starts,
            -1.0,
            0.0,
            adaptive=False,
        )
        mat[use_qbx] += np.transpose(qbx_mat, (0, 2, 1))

        report["use_qbx"] = use_qbx
        report["exp_centers"] = exp_centers
        report["exp_rs"] = exp_rs

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

        nearfield_mat = np.zeros(
            (nearfield_obs_pts.shape[0], combined_src.n_pts, K.ndim)
        )
        K.nearfield(
            nearfield_mat,
            nearfield_obs_pts,
            combined_src,
            panel_obs_pts,
            panel_obs_pts_starts,
            1.0,
            tol,
            adaptive=True,
        )

        # setting d_refine=0.0 prevents refinement which is what we want to
        # cancel out the direct component terms
        K.nearfield(
            nearfield_mat,
            nearfield_obs_pts,
            combined_src,
            panel_obs_pts,
            panel_obs_pts_starts,
            -1.0,
            0.0,
            adaptive=False,
        )
        mat[use_nearfield] += np.transpose(nearfield_mat, (0, 2, 1))

    mats = []
    col_idx = 0
    for s in srcs:
        mats.append(mat[:, col_idx : col_idx + s.n_pts])
        col_idx += s.n_pts

    if return_reports:
        return mats, report
    else:
        return mats
