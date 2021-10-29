import warnings

import numpy as np
import scipy.spatial

from ._ext import identify_nearfield_panels, local_qbx_integrals, nearfield_integrals


class LaplaceKernel:
    def __init__(self, d_cutoff=2.0, d_up=4.0, d_qbx=0.5, d_refine=2.5):
        self.d_cutoff = d_cutoff
        self.d_up = d_up
        self.d_qbx = d_qbx
        self.d_refine = d_refine

    def integrate(
        self,
        obs_pts,
        src,
        tol=1e-13,
        max_p=50,
        on_src_direction=1.0,
        return_report=False,
        d_cutoff=None,
        d_up=None,
        d_qbx=None,
        d_refine=None,
    ):
        if d_cutoff is None:
            d_cutoff = self.d_cutoff
        if d_up is None:
            d_up = self.d_up
        if d_qbx is None:
            d_qbx = self.d_qbx
        if d_refine is None:
            d_refine = self.d_refine

        # step 1: construct the farfield matrix!
        mat = self._direct(obs_pts, src)

        # step 2: identify QBX observation points.
        src_tree = scipy.spatial.KDTree(src.pts)
        closest_dist, closest_idx = src_tree.query(obs_pts)
        closest_panel_length = src.panel_length[closest_idx // src.panel_order]
        use_qbx = closest_dist < d_qbx * closest_panel_length

        n_qbx = np.sum(use_qbx)
        if n_qbx > 0:
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
            (
                qbx_panels,
                qbx_panel_starts,
                qbx_panel_obs_pts,
                qbx_panel_obs_pt_starts,
            ) = identify_nearfield_panels(
                exp_centers, d_cutoff * qbx_L, src_tree, src.panel_order
            )

            # step 5: QBX integrals
            # TODO: This could be replaced by a sparse local matrix.
            qbx_mat = np.zeros((qbx_obs_pts.shape[0], src.n_pts, self.ndim))
            p, kappa_too_small = local_qbx_integrals(
                self.exp_deriv,
                self.eval_deriv,
                qbx_mat,
                qbx_obs_pts,
                src,
                exp_centers,
                exp_rs,
                max_p,
                tol,
                d_refine,
                qbx_panels,
                qbx_panel_starts,
            )
            if np.any(kappa_too_small):
                warnings.warn("Some integrals diverged because kappa is too small.")

            # step 6: subtract off the direct term whenever a QBX integral is used.
            self._nearfield(
                qbx_mat,
                qbx_obs_pts,
                src,
                qbx_panel_obs_pts,
                qbx_panel_obs_pt_starts,
                -1.0,
                0.0,
            )
            mat[use_qbx] += qbx_mat

        # step 7: nearfield integrals
        use_nearfield = (closest_dist < d_up * closest_panel_length) & (~use_qbx)
        n_nearfield = np.sum(use_nearfield)

        if n_nearfield > 0:
            nearfield_obs_pts = obs_pts[use_nearfield]
            nearfield_L = closest_panel_length[use_nearfield]

            obs_tree = scipy.spatial.KDTree(nearfield_obs_pts)
            panel_obs_pts = obs_tree.query_ball_point(
                src.panel_centers, d_up * src.panel_length
            )
            panel_obs_pts_starts = np.zeros(src.n_panels + 1, dtype=int)
            panel_obs_pts_starts[1:] = np.cumsum([len(p) for p in panel_obs_pts])
            panel_obs_pts = np.concatenate(panel_obs_pts, dtype=int, casting="unsafe")

            nearfield_mat = np.zeros((nearfield_obs_pts.shape[0], src.n_pts, self.ndim))
            self._nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                src,
                panel_obs_pts,
                panel_obs_pts_starts,
                1.0,
                d_refine,
            )

            # setting d_refine=0.0 prevents refinement which is what we want to
            # cancel out the direct component terms
            self._nearfield(
                nearfield_mat,
                nearfield_obs_pts,
                src,
                panel_obs_pts,
                panel_obs_pts_starts,
                -1.0,
                0.0,
            )
            mat[use_nearfield] += nearfield_mat

        if return_report:
            report = dict()
            report["n_qbx"] = n_qbx
            report["n_nearfield"] = n_nearfield
            if n_nearfield > 0:
                report["n_nearfield_panels"] = panel_obs_pts_starts[-1]
            if n_qbx > 0:
                report["n_qbx_panels"] = qbx_panel_obs_pt_starts[-1]
            for k in [
                "exp_centers",
                "exp_rs",
                "p",
                "kappa_too_small",
                "use_qbx",
            ]:
                report[k] = locals().get(k, None)
            return np.transpose(mat, (0, 2, 1)), report
        else:
            return np.transpose(mat, (0, 2, 1))

    def _nearfield(self, mat, obs_pts, src, panels, panel_starts, mult, distance_limit):
        return nearfield_integrals(
            self.name, mat, obs_pts, src, panels, panel_starts, mult, distance_limit
        )


class SingleLayer(LaplaceKernel):
    name = "single_layer"
    ndim = 1
    exp_deriv = False
    eval_deriv = False

    def _direct(self, obs_pts, src):

        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        G = (1.0 / (4 * np.pi)) * np.log(r2)
        G[too_close] = 0

        return (G * src.jacobians * src.quad_wts[None, :])[:, :, None]


class DoubleLayer(LaplaceKernel):
    name = "double_layer"
    ndim = 1
    exp_deriv = True
    eval_deriv = False

    def _direct(self, obs_pts, src):
        """
        Compute the entries of the matrix that forms the double layer potential.
        """
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        # The double layer potential
        integrand = (
            -1.0
            / (2 * np.pi * r2)
            * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1])
        )
        integrand[too_close] = 0.0

        return (integrand * src.jacobians * src.quad_wts[None, :])[:, :, None]


class AdjointDoubleLayer(LaplaceKernel):
    name = "adjoint_double_layer"
    ndim = 2
    exp_deriv = False
    eval_deriv = True

    def _direct(self, obs_pts, src):
        dx = obs_pts[:, None, 0] - src.pts[None, :, 0]
        dy = obs_pts[:, None, 1] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        out = np.empty((obs_pts.shape[0], src.n_pts, 2))
        out[:, :, 0] = dx
        out[:, :, 1] = dy

        C = -1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, :, None]


class Hypersingular(LaplaceKernel):
    name = "hypersingular"
    ndim = 2
    exp_deriv = True
    eval_deriv = True

    def _direct(self, obs_pts, src):
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        A = 2 * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1]) / r2
        C = 1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        out = np.empty((obs_pts.shape[0], src.n_pts, 2))

        # The definition of the hypersingular kernel.
        # unscaled sigma_xz component
        out[:, :, 0] = src.normals[None, :, 0] - A * dx
        # unscaled sigma_xz component
        out[:, :, 1] = src.normals[None, :, 1] - A * dy

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, :, None]


single_layer = SingleLayer(d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3)
double_layer = DoubleLayer(d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3)
adjoint_double_layer = AdjointDoubleLayer(
    d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3
)
hypersingular = Hypersingular(d_cutoff=2.0, d_refine=2.5, d_up=1.5, d_qbx=0.4)
