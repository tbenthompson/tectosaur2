import numpy as np

from ._ext import nearfield_integrals


class LaplaceKernel:
    def __init__(self, d_cutoff=2.0, d_up=4.0, d_qbx=0.5, d_refine=2.5, max_p=50):
        self.d_cutoff = d_cutoff
        self.d_up = d_up
        self.d_qbx = d_qbx
        self.d_refine = d_refine
        self.max_p = max_p

    def nearfield(self, mat, obs_pts, src, panels, panel_starts, mult, tol):
        return nearfield_integrals(
            self.name, mat, obs_pts, src, panels, panel_starts, mult, tol
        )


class SingleLayer(LaplaceKernel):
    name = "single_layer"
    ndim = 1
    exp_deriv = False
    eval_deriv = False

    def direct(self, obs_pts, src):

        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        G = (1.0 / (4 * np.pi)) * np.log(r2)
        G[too_close] = 0

        return (G * src.jacobians * src.quad_wts[None, :])[:, None, :]


class DoubleLayer(LaplaceKernel):
    name = "double_layer"
    ndim = 1
    exp_deriv = True
    eval_deriv = False

    def direct(self, obs_pts, src):
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

        return (integrand * src.jacobians * src.quad_wts[None, :])[:, None, :]


class AdjointDoubleLayer(LaplaceKernel):
    name = "adjoint_double_layer"
    ndim = 2
    exp_deriv = False
    eval_deriv = True

    def direct(self, obs_pts, src):
        dx = obs_pts[:, None, 0] - src.pts[None, :, 0]
        dy = obs_pts[:, None, 1] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        out = np.empty((obs_pts.shape[0], 2, src.n_pts))
        out[:, 0, :] = dx
        out[:, 1, :] = dy

        C = -1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :]


class Hypersingular(LaplaceKernel):
    name = "hypersingular"
    ndim = 2
    exp_deriv = True
    eval_deriv = True

    def direct(self, obs_pts, src):
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        A = 2 * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1]) / r2
        C = 1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        out = np.empty((obs_pts.shape[0], 2, src.n_pts))

        # The definition of the hypersingular kernel.
        # unscaled sigma_xz component
        out[:, 0, :] = src.normals[None, :, 0] - A * dx
        # unscaled sigma_xz component
        out[:, 1, :] = src.normals[None, :, 1] - A * dy

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :]


d_refine = 8.0
single_layer = SingleLayer(d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3)
double_layer = DoubleLayer(d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3)
adjoint_double_layer = AdjointDoubleLayer(
    d_cutoff=1.5, d_refine=2.5, d_up=1.5, d_qbx=0.3
)
hypersingular = Hypersingular(d_cutoff=2.0, d_refine=2.5, d_up=1.5, d_qbx=0.4)
