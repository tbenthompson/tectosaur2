import warnings

import numpy as np
import scipy.spatial

from ._ext import local_qbx_integrals, nearfield_integrals
from .mesh import apply_interp_mat, stage2_refine


class LaplaceKernel:
    def __init__(self, d_cutoff=1.5, d_up=[0.5, 0.0, 0.0, 4.0], kappa_qbx=4):
        self.d_cutoff = d_cutoff
        self.d_up = d_up
        self.kappa_qbx = kappa_qbx

    def integrate(
        self,
        obs_pts,
        src,
        tol=1e-13,
        max_p=50,
        on_src_direction=1.0,
        return_report=False,
    ):
        # step 1: construct the farfield matrix!
        mat = self._direct(obs_pts, src)

        # step 2: identify QBX observation points.
        src_tree = scipy.spatial.KDTree(src.pts)
        closest_dist, closest_idx = src_tree.query(obs_pts)
        closest_panel_length = src.panel_length[closest_idx // src.panel_order]
        use_qbx = closest_dist < self.d_up[-1] * closest_panel_length
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
        exp_centers = (
            qbx_closest_pts + direction[:, None] * qbx_normals * exp_rs[:, None]
        )

        # step 4: find which source panels need to use QBX
        # this information must be propagated to the refined panels.
        qbx_src_pts_unrefined = src_tree.query_ball_point(
            exp_centers, self.d_cutoff * qbx_L
        )
        qbx_panels = []
        for i in range(exp_centers.shape[0]):
            panels = np.unique(np.array(qbx_src_pts_unrefined[i]) // src.panel_order)
            qbx_panels.append(panels)
        qbx_panel_starts = np.zeros(exp_centers.shape[0] + 1, dtype=int)
        qbx_panel_starts[1:] = np.cumsum([p.shape[0] for p in panels])
        qbx_panels = np.concatenate(qbx_panels)

        refined_src, interp_mat, refinement_plan = stage2_refine(
            src, exp_centers, kappa=self.kappa_qbx
        )
        refinement_map = np.unique(
            refinement_plan[:, 0].astype(int), return_inverse=True
        )

        # step 5: QBX integrals
        # TODO: This could be replaced by a sparse local matrix.
        n_qbx = np.sum(use_qbx)
        if n_qbx > 0:
            qbx_mat = np.zeros((qbx_obs_pts.shape[0], 1, refined_src.n_pts))
            p, kappa_too_small = local_qbx_integrals(
                self.exp_deriv,
                self.eval_deriv,
                qbx_mat,
                qbx_obs_pts,
                refined_src,
                exp_centers,
                exp_rs,
                max_p,
                tol,
                qbx_panels,
                qbx_panel_starts,
                refinement_map,
            )
            if np.any(kappa_too_small):
                warnings.warn("Some integrals diverged because kappa is too small.")
            qbx_mat = np.ascontiguousarray(apply_interp_mat(qbx_mat, interp_mat))

            # step 6: subtract off the direct term whenever a QBX integral is used.
            self._nearfield(qbx_mat, qbx_obs_pts, src, qbx_src_panels_unrefined, -1.0)
            mat[use_qbx] += qbx_mat

        # step 7: nearfield integrals
        use_nearfield = (closest_dist < self.d_up[0] * closest_panel_length) & (
            ~use_qbx
        )
        n_nearfield = np.sum(use_nearfield)
        if n_nearfield > 0:
            nearfield_obs_pts = obs_pts[use_nearfield]
            nearfield_L = closest_panel_length[use_nearfield]
            nearfield_src_pts_unrefined = src_tree.query_ball_point(
                nearfield_obs_pts, self.d_up[0] * nearfield_L
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
            self._nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                refined_src,
                nearfield_src_panels_refined,
                1.0,
            )
            nearfield_mat = np.ascontiguousarray(
                apply_interp_mat(nearfield_mat, interp_mat)
            )
            self._nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                src,
                nearfield_src_panels_unrefined,
                -1.0,
            )
            mat[use_nearfield] += nearfield_mat

        if return_report:
            report = dict()
            report["refined_src"] = refined_src
            report["interp_mat"] = interp_mat

            report["use_qbx"] = use_qbx
            report["n_qbx_panels"] = np.sum([len(p) for p in qbx_src_panels_refined])
            report["qbx_src_panels"] = qbx_src_panels_unrefined
            report["exp_centers"] = exp_centers
            report["exp_rs"] = exp_rs
            report["p"] = p
            report["kappa_too_small"] = kappa_too_small

            report["use_nearfield"] = use_nearfield
            report["n_nearfield_panels"] = np.sum(
                [len(p) for p in nearfield_src_panels_unrefined]
            )
            report["nearfield_src_panels"] = nearfield_src_panels_unrefined
            return mat, report
        else:
            return mat

    def _nearfield(self, mat, obs_pts, src, panels, panel_starts, refinement_map, mult):
        return nearfield_integrals(
            self.name, mat, obs_pts, src, panels, panel_starts, refinement_map, mult
        )


class SingleLayer(LaplaceKernel):
    name = "single_layer"
    exp_deriv = False
    eval_deriv = False

    def _direct(self, obs_pts, src):

        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        r2[r2 == 0] = 1
        G = (1.0 / (4 * np.pi)) * np.log(r2)
        G[r2 == 0] = 0

        return (G * src.jacobians * src.quad_wts[None, :])[:, :, None]


class DoubleLayer(LaplaceKernel):
    name = "double_layer"
    exp_deriv = True
    eval_deriv = False

    def _direct(self, obs_pts, src):
        """
        Compute the entries of the matrix that forms the double layer potential.
        """
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        r2[r2 == 0] = 1

        # The double layer potential
        integrand = (
            -1.0
            / (2 * np.pi * r2)
            * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1])
        )
        integrand[r2 == 0] = 0.0

        return (integrand * src.jacobians * src.quad_wts[None, :])[:, :, None]


class AdjointDoubleLayer(LaplaceKernel):
    name = "adjoint_double_layer"
    exp_deriv = False
    eval_deriv = True

    def _direct(self, obs_pts, src):
        dx = obs_pts[:, None, 0] - src.pts[None, :, 0]
        dy = obs_pts[:, None, 1] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        r2[r2 == 0] = 1

        out = np.empty((obs_pts.shape[0], src.n_pts, 2))
        out[:, :, 0] = dx
        out[:, :, 1] = dy

        C = -1.0 / (2 * np.pi * r2)
        C[r2 == 0] = 0

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, :, None]


class Hypersingular(LaplaceKernel):
    name = "hypersingular"
    exp_deriv = True
    eval_deriv = True

    def _direct(self, obs_pts, src):
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        r2[r2 == 0] = 1

        A = 2 * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1]) / r2
        C = 1.0 / (2 * np.pi * r2)
        C[r2 == 0] = 0
        out = np.empty((obs_pts.shape[0], src.n_pts, 2))

        # The definition of the hypersingular kernel.
        # unscaled sigma_xz component
        out[:, :, 0] = src.normals[None, :, 0] - A * dx
        # unscaled sigma_xz component
        out[:, :, 1] = src.normals[None, :, 1] - A * dy

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, :, None]


single_layer = SingleLayer()
double_layer = DoubleLayer()
adjoint_double_layer = AdjointDoubleLayer()
hypersingular = Hypersingular()
