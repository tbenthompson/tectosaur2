import numpy as np

from .integrate import Kernel


class SingleLayer(Kernel):
    name = "single_layer"
    src_dim = 1
    obs_dim = 1

    def direct(self, obs_pts, src):

        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        G = (1.0 / (4 * np.pi)) * np.log(r2)
        G[too_close] = 0

        return (G * src.jacobians * src.quad_wts[None, :])[:, None, :, None]


class DoubleLayer(Kernel):
    name = "double_layer"
    src_dim = 1
    obs_dim = 1

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

        return (integrand * src.jacobians * src.quad_wts[None, :])[:, None, :, None]


class AdjointDoubleLayer(Kernel):
    name = "adjoint_double_layer"
    src_dim = 1
    obs_dim = 2

    def direct(self, obs_pts, src):
        dx = obs_pts[:, None, 0] - src.pts[None, :, 0]
        dy = obs_pts[:, None, 1] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        out = np.empty((obs_pts.shape[0], 2, src.n_pts, 1))
        out[:, 0, :, 0] = dx
        out[:, 1, :, 0] = dy

        C = -1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :, None]


class Hypersingular(Kernel):
    name = "hypersingular"
    src_dim = 1
    obs_dim = 2

    def direct(self, obs_pts, src):
        dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        A = 2 * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1]) / r2
        C = 1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        out = np.empty((obs_pts.shape[0], 2, src.n_pts, 1))

        # The definition of the hypersingular kernel.
        # unscaled sigma_xz component
        out[:, 0, :, 0] = src.normals[None, :, 0] - A * dx
        # unscaled sigma_xz component
        out[:, 1, :, 0] = src.normals[None, :, 1] - A * dy

        # multiply by the scaling factor, jacobian and quadrature weights
        return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :, None]


single_layer = SingleLayer(d_cutoff=2.0, d_up=1.5, d_qbx=0.3, default_tol=1e-13)
double_layer = DoubleLayer(d_cutoff=4.0, d_up=2.0, d_qbx=0.4, default_tol=1e-13)
adjoint_double_layer = AdjointDoubleLayer(
    d_cutoff=4.0, d_up=2.0, d_qbx=0.4, default_tol=1e-13
)
hypersingular = Hypersingular(d_cutoff=5.0, d_up=2.5, d_qbx=0.5, default_tol=1e-12)
