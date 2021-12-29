import warnings

import numpy as np

from .integrate import Kernel

"""
Facts to note if you are comparing the formulas here with formulas in another
place:

1. The dx and dy terms are (observation minus source). Many sources use (source
minus observation). This can introduce sign errors.

2. The normal vectors are inward pointing normal vectors. Many sources use
outward pointing normal vectors. This can introduce sign errors.
"""


class SingleLayer(Kernel):
    name = "single_layer"
    src_dim = 1
    obs_dim = 1

    def kernel(self, obs_pts, src_pts, src_normals=None):
        dx = obs_pts[:, 0, None] - src_pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src_pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        G = -(1.0 / (4 * np.pi)) * np.log(r2)
        G[too_close] = 0
        return G[:, None, :, None]


class DoubleLayer(Kernel):
    name = "double_layer"
    src_dim = 1
    obs_dim = 1

    def kernel(self, obs_pts, src_pts, src_normals):
        """
        Compute the entries of the matrix that forms the double layer potential.
        """
        dx = obs_pts[:, 0, None] - src_pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src_pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        # The double layer potential
        integrand = (
            -1.0
            / (2 * np.pi * r2)
            * (dx * src_normals[None, :, 0] + dy * src_normals[None, :, 1])
        )
        integrand[too_close] = 0.0

        return integrand[:, None, :, None]


class AdjointDoubleLayer(Kernel):
    name = "adjoint_double_layer"
    src_dim = 1
    obs_dim = 2

    def kernel(self, obs_pts, src_pts, src_normals=None):
        dx = obs_pts[:, None, 0] - src_pts[None, :, 0]
        dy = obs_pts[:, None, 1] - src_pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        out = np.empty((obs_pts.shape[0], 2, src_pts.shape[0], 1))
        out[:, 0, :, 0] = dx
        out[:, 1, :, 0] = dy

        C = -1.0 / (2 * np.pi * r2)
        C[too_close] = 0

        return out * C[:, None, :, None]


class Hypersingular(Kernel):
    name = "hypersingular"
    src_dim = 1
    obs_dim = 2
    C = -1.0 / (2 * np.pi)

    def kernel(self, obs_pts, src_pts, src_normals):
        dx = obs_pts[:, 0, None] - src_pts[None, :, 0]
        dy = obs_pts[:, 1, None] - src_pts[None, :, 1]
        r2 = dx ** 2 + dy ** 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            invr2 = 1 / r2
        too_close = r2 <= 1e-16
        invr2[too_close] = 0

        A = 2 * (dx * src_normals[None, :, 0] + dy * src_normals[None, :, 1]) * invr2
        B = self.C * invr2

        out = np.empty((obs_pts.shape[0], 2, src_pts.shape[0], 1))

        # The definition of the hypersingular kernel.
        # unscaled sigma_xz component
        out[:, 0, :, 0] = src_normals[None, :, 0] - A * dx
        # unscaled sigma_xz component
        out[:, 1, :, 0] = src_normals[None, :, 1] - A * dy

        return out * B[:, None, :, None]


single_layer = SingleLayer(d_cutoff=2.0, d_up=1.5, d_qbx=0.3, default_tol=1e-13)
double_layer = DoubleLayer(d_cutoff=4.0, d_up=2.0, d_qbx=0.4, default_tol=1e-13)
adjoint_double_layer = AdjointDoubleLayer(
    d_cutoff=4.0, d_up=2.0, d_qbx=0.4, default_tol=1e-13
)
hypersingular = Hypersingular(d_cutoff=5.0, d_up=2.5, d_qbx=0.5, default_tol=1e-12)
