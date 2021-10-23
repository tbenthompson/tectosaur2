import warnings

import numpy as np
import scipy.spatial

from ._local_qbx import local_qbx_integrals, nearfield_integrals
from .direct import double_layer_matrix
from .mesh import apply_interp_mat, stage2_refine


def local_qbx(
    exp_deriv,
    eval_deriv,
    obs_pts,
    src,
    tol,
    d_cutoff,
    kappa,
    d_up=None,
    on_src_direction=1,
    max_p=50,
    return_report=False,
):
    # step 1: construct the farfield matrix!
    mat = double_layer_matrix(obs_pts, src)

    # step 2: identify QBX observation points.
    if d_up is None:
        d_up = [0.0, 0.5]

    src_tree = scipy.spatial.KDTree(src.pts)
    closest_dist, closest_idx = src_tree.query(obs_pts)
    closest_panel_length = src.panel_length[closest_idx // src.panel_order]
    use_qbx = closest_dist < d_up[-1] * closest_panel_length
    qbx_closest_pts = src.pts[closest_idx][use_qbx]
    qbx_normals = src.normals[closest_idx][use_qbx]
    qbx_obs_pts = obs_pts[use_qbx]
    qbx_L = closest_panel_length[use_qbx]

    # step 3: find expansion centers
    # TODO: account for singularities
    exp_rs = qbx_L * 0.5
    direction_dot = (
        np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1) / exp_rs
    )
    direction = np.sign(direction_dot)
    direction[np.abs(direction) < 1e-13] = on_src_direction
    exp_centers = qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]

    # step 4: find which source panels need to use QBX
    # this information must be propagated to the refined panels.
    qbx_src_pts_unrefined = src_tree.query_ball_point(exp_centers, d_cutoff * qbx_L)

    refined_src, interp_mat, refinement_plan = stage2_refine(
        src, exp_centers, kappa=kappa
    )
    refinement_map = {i: [] for i in range(src.n_panels)}
    # todo: could use np.unique here
    orig_panel = refinement_plan[:, 0].astype(int)
    for i in range(orig_panel.shape[0]):
        refinement_map[orig_panel[i]].append(i)

    qbx_src_panels_refined = []
    qbx_src_panels_unrefined = []
    for i in range(exp_centers.shape[0]):
        unrefined_panels = np.unique(
            np.array(qbx_src_pts_unrefined[i]) // src.panel_order
        )
        qbx_src_panels_unrefined.append(unrefined_panels)
        qbx_src_panels_refined.append(
            np.concatenate([refinement_map[p] for p in unrefined_panels])
        )

    # step 5: QBX integrals
    # TODO: This could be replaced by a sparse local matrix.
    n_qbx = np.sum(use_qbx)
    if n_qbx > 0:
        qbx_mat = np.zeros((qbx_obs_pts.shape[0], 1, refined_src.n_pts))
        p, kappa_too_small = local_qbx_integrals(
            exp_deriv,
            eval_deriv,
            qbx_mat,
            qbx_obs_pts,
            refined_src,
            exp_centers,
            exp_rs,
            max_p,
            tol,
            qbx_src_panels_refined,
        )
        if np.any(kappa_too_small):
            warnings.warn("Some integrals diverged because kappa is too small.")
        qbx_mat = np.ascontiguousarray(apply_interp_mat(qbx_mat, interp_mat))

        # step 6: subtract off the direct term whenever a QBX integral is used.
        nearfield_integrals(qbx_mat, qbx_obs_pts, src, qbx_src_panels_unrefined, -1.0)
        mat[use_qbx] += qbx_mat

    # step 7: nearfield integrals
    use_nearfield = (closest_dist < d_up[0] * closest_panel_length) & (~use_qbx)
    n_nearfield = np.sum(use_nearfield)
    if n_nearfield > 0:
        nearfield_obs_pts = obs_pts[use_nearfield]
        nearfield_L = closest_panel_length[use_nearfield]
        nearfield_src_pts_unrefined = src_tree.query_ball_point(
            nearfield_obs_pts, d_up[0] * nearfield_L
        )

        nearfield_src_panels_refined = []
        nearfield_src_panels_unrefined = []
        for i in range(nearfield_obs_pts.shape[0]):
            unrefined_panels = np.unique(
                np.array(nearfield_src_pts_unrefined[i]) // src.panel_order
            )
            nearfield_src_panels_unrefined.append(unrefined_panels)
            nearfield_src_panels_refined.append(
                np.concatenate([refinement_map[p] for p in unrefined_panels])
            )

        nearfield_mat = np.zeros((nearfield_obs_pts.shape[0], 1, refined_src.n_pts))
        nearfield_integrals(
            nearfield_mat,
            nearfield_obs_pts,
            refined_src,
            nearfield_src_panels_refined,
            1.0,
        )
        nearfield_mat = np.ascontiguousarray(
            apply_interp_mat(nearfield_mat, interp_mat)
        )
        nearfield_integrals(
            nearfield_mat, nearfield_obs_pts, src, nearfield_src_panels_unrefined, -1.0
        )
        mat[use_nearfield] += nearfield_mat

    if return_report:
        report = dict()
        report["stage2_src"] = refined_src
        report["exp_centers"] = exp_centers
        report["exp_rs"] = exp_rs
        report["n_qbx_panels"] = np.sum([len(p) for p in qbx_src_panels_refined])
        report["qbx_src_panels_refined"] = qbx_src_panels_refined
        report["p"] = p
        report["kappa_too_small"] = kappa_too_small
        return mat, report
    else:
        return mat
