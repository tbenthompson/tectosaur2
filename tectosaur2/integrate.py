import warnings
from dataclasses import dataclass

import numpy as np
import scipy.spatial

from tectosaur2.laplace2d import LaplaceKernel
from tectosaur2.mesh import PanelSurface

from ._ext import identify_nearfield_panels, local_qbx_integrals


@dataclass()
class Integral:
    src: PanelSurface
    K: LaplaceKernel
    on_src_direction: float = 1.0
    d_cutoff: float = None
    d_up: float = None
    d_qbx: float = None
    d_refine: float = None
    max_p: int = None

    def get_d_cutoff(self):
        return self.d_cutoff if self.d_cutoff is not None else self.K.d_cutoff

    def get_d_up(self):
        return self.d_up if self.d_up is not None else self.K.d_up

    def get_d_qbx(self):
        return self.d_qbx if self.d_qbx is not None else self.K.d_qbx

    def get_d_refine(self):
        return self.d_refine if self.d_refine is not None else self.K.d_refine

    def get_max_p(self):
        return self.max_p if self.max_p is not None else self.K.max_p


def to_integrals(integrals):
    out = []
    for t in integrals:
        if isinstance(t, Integral):
            out.append(t)
        else:
            out.append(Integral(src=t[0], K=t[1]))
    return out


def integrate(obs_pts, *terms, tol=1e-13, return_reports=False):
    terms = to_integrals(terms)

    n_terms = len(terms)
    n_obs = obs_pts.shape[0]

    # step 1: construct the farfield matrix!
    mats = [t.K.direct(obs_pts, t.src) for t in terms]
    if return_reports:
        reports = [dict() for t in terms]

    # step 1: figure out which observation points need to use QBX
    src_trees = [scipy.spatial.KDTree(t.src.pts) for t in terms]
    closest_dist = np.full(n_obs, np.finfo(np.float64).max)
    closest_idx = np.empty(n_obs, dtype=int)
    closest_src = np.empty_like(closest_idx)
    closest_panel_length = np.empty_like(closest_dist)
    use_qbx = np.zeros(n_obs, dtype=bool)
    use_nearfield = np.zeros(n_obs, dtype=bool)
    for i in range(n_terms):
        src = terms[i].src

        this_closest_dist, this_closest_idx = src_trees[i].query(obs_pts)
        closer = this_closest_dist < closest_dist
        closest_dist[closer] = this_closest_dist[closer]
        closest_src[closer] = i
        closest_idx[closer] = this_closest_idx[closer]
        closest_panel_length[closer] = src.panel_length[
            closest_idx[closer] // src.panel_order
        ]
        this_use_qbx = closest_dist < terms[i].get_d_qbx() * closest_panel_length
        use_qbx |= this_use_qbx
        use_nearfield |= (closest_dist < terms[i].get_d_up() * closest_panel_length) & (
            ~this_use_qbx
        )

    n_qbx = np.sum(use_qbx)
    print(n_qbx)
    qbx_obs_pts = obs_pts[use_qbx]
    if n_qbx > 0:
        qbx_closest_pts = np.empty((n_qbx, 2))
        qbx_normals = np.empty((n_qbx, 2))
        qbx_L = np.empty(n_qbx)
        on_src_direction = np.empty(n_qbx)
        for i, t in enumerate(terms):
            which_pts = closest_src[use_qbx] == i
            pt_indices = closest_idx[use_qbx][which_pts]
            qbx_closest_pts[which_pts] = t.src.pts[pt_indices]
            qbx_normals[which_pts] = t.src.normals[pt_indices]
            qbx_L[which_pts] = closest_panel_length[use_qbx][which_pts]
            on_src_direction[which_pts] = t.on_src_direction

        # step 3: find expansion centers
        # TODO: account for singularities
        exp_rs = qbx_L * 0.5

        direction_dot = (
            np.sum(qbx_normals * (qbx_obs_pts - qbx_closest_pts), axis=1) / exp_rs
        )
        direction = np.sign(direction_dot)
        direction[np.abs(direction) < 1e-13] = on_src_direction

        for j in range(30):
            exp_centers = (
                qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
            )
            which_violations = np.zeros(exp_centers.shape[0], dtype=bool)
            for i in range(n_terms):
                dist_to_nearest_panel = src_trees[i].query(exp_centers)[0]
                # TODO: WRITE A TEST THAT HAS VIOLATIONS
                # The fudge factor helps avoid numerical precision issues. For example,
                # when we offset an expansion center 1.0 away from a surface node,
                # without the fudge factor this test will be checking 1.0 < 1.0, but
                # that is fragile in the face of small 1e-15 sized numerical errors.
                # By simply multiplying by 1.0001, we avoid this issue without
                # introducing any other problems.
                fudge_factor = 1.0001
                which_violations |= dist_to_nearest_panel * fudge_factor < np.abs(
                    exp_rs
                )

            if not which_violations.any():
                break
            exp_rs[which_violations] *= 0.75
        exp_rs *= 0.9

        for i in range(n_terms):
            qbx_mat, p, kappa_too_small = _integrate_qbx(
                qbx_obs_pts, terms[i], exp_centers, exp_rs, qbx_L, src_trees[i], tol
            )
            mats[i][use_qbx] += np.transpose(qbx_mat, (0, 2, 1))
            reports[i]["p"] = p
            reports[i]["kappa_too_small"] = kappa_too_small
            reports[i]["use_qbx"] = use_qbx
            reports[i]["exp_centers"] = exp_centers
            reports[i]["exp_rs"] = exp_rs

    n_nearfield = np.sum(use_nearfield)
    if n_nearfield > 0:
        nearfield_obs_pts = obs_pts[use_nearfield]
        for i in range(n_terms):
            src = terms[i].src

            obs_tree = scipy.spatial.KDTree(nearfield_obs_pts)
            panel_obs_pts = obs_tree.query_ball_point(
                src.panel_centers, terms[i].get_d_up() * src.panel_length
            )

            panel_obs_pts_starts = np.zeros(src.n_panels + 1, dtype=int)
            panel_obs_pts_starts[1:] = np.cumsum([len(p) for p in panel_obs_pts])
            panel_obs_pts = np.concatenate(panel_obs_pts, dtype=int, casting="unsafe")

            K = terms[i].K
            nearfield_mat = np.zeros((nearfield_obs_pts.shape[0], src.n_pts, K.ndim))
            K.nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                src,
                panel_obs_pts,
                panel_obs_pts_starts,
                1.0,
                terms[i].get_d_refine(),
            )

            # setting d_refine=0.0 prevents refinement which is what we want to
            # cancel out the direct component terms
            K.nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                src,
                panel_obs_pts,
                panel_obs_pts_starts,
                -1.0,
                0.0,
            )
            mats[i][use_nearfield] += np.transpose(nearfield_mat, (0, 2, 1))

    if return_reports:
        return mats, reports
    else:
        return mats


def _integrate_qbx(obs_pts, term, exp_centers, exp_rs, exp_panel_L, src_tree, tol):
    # step 4: find which source panels need to use QBX

    # TODO: use ckdtree directly to avoid python
    qbx_panel_src_pts = src_tree.query_ball_point(
        exp_centers, term.get_d_cutoff() * exp_panel_L, return_sorted=True
    )

    (
        qbx_panels,
        qbx_panel_starts,
        qbx_panel_obs_pts,
        qbx_panel_obs_pt_starts,
    ) = identify_nearfield_panels(
        exp_centers,
        qbx_panel_src_pts,
        term.src.n_panels,
        term.src.panel_order,
    )

    # step 5: QBX integrals
    # TODO: This could be replaced by a sparse local matrix.
    qbx_mat = np.zeros((obs_pts.shape[0], term.src.n_pts, term.K.ndim))
    p, kappa_too_small = local_qbx_integrals(
        term.K.exp_deriv,
        term.K.eval_deriv,
        qbx_mat,
        obs_pts,
        term.src,
        exp_centers,
        exp_rs,
        term.get_max_p(),
        tol,
        term.get_d_refine(),
        qbx_panels,
        qbx_panel_starts,
    )
    if np.any(kappa_too_small):
        warnings.warn("Some integrals diverged because kappa is too small.")

    # step 6: subtract off the direct term whenever a QBX integral is used.
    term.K.nearfield(
        qbx_mat,
        obs_pts,
        term.src,
        qbx_panel_obs_pts,
        qbx_panel_obs_pt_starts,
        -1.0,
        0.0,
    )

    return qbx_mat, p, kappa_too_small


def direct(kernel, obs_pts, src):
    return np.transpose(kernel._direct(obs_pts, src), (0, 2, 1))
