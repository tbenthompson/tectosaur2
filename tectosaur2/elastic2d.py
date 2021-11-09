import numpy as np

from .integrate import Kernel


class ElasticU(Kernel):
    name = "elastic_U"
    src_dim = 2
    obs_dim = 2

    def __init__(self, shear_modulus=1.0, poisson_ratio=0.25, **kwargs):
        self.disp_C1 = 1.0 / (8 * np.pi * shear_modulus * (1 - poisson_ratio))
        self.disp_C2 = 3 - 4 * poisson_ratio
        super().__init__(**kwargs)

    def direct(self, obs_pts, src):
        d = [
            obs_pts[:, 0, None] - src.pts[None, :, 0],
            obs_pts[:, 1, None] - src.pts[None, :, 1],
        ]
        r2 = d[0] * d[0] + d[1] * d[1]
        too_close = r2 <= 1e-16
        r2[too_close] = 1

        S = np.empty((obs_pts.shape[0], 2, src.n_pts, 2))
        for d_obs in range(2):
            for d_src in range(2):
                S[:, d_obs, :, d_src] = self.disp_C1 * (
                    (d_obs == d_src) * (-0.5 * self.disp_C2 * np.log(r2))
                    + d[d_obs] * d[d_src] / r2
                )
                S[:, d_obs, :, d_src][too_close] = 0

        S *= (src.jacobians * src.quad_wts)[None, None, :, None]
        return S


class ElasticT(Kernel):
    name = "elastic_T"
    src_dim = 2
    obs_dim = 2

    def __init__(self, shear_modulus, poisson_ratio):
        self.trac_C1 = 1.0 / (4 * np.pi * (1 - poisson_ratio))
        self.trac_C2 = 1 - 2.0 * poisson_ratio

    def direct(self, obs_pts, src):
        d = [
            obs_pts[:, 0, None] - src.pts[None, :, 0],
            obs_pts[:, 1, None] - src.pts[None, :, 1],
        ]
        r2 = d[0] * d[0] + d[1] * d[1]
        too_close = r2 <= 1e-16
        r2[too_close] = 1
        r = np.sqrt(r2)

        drdn = (d[0] * src.normals[None, :, 0] + d[1] * src.normals[None, :, 1]) / r

        T = np.empty((obs_pts.shape[0], 2, src.n_pts, 2))
        for d_obs in range(2):
            for d_src in range(2):
                t1 = self.trac_C2 * (d_obs == d_src) + 2 * d[d_obs] * d[d_src] / r2
                t2 = (
                    self.trac_C2
                    * (
                        src.normals[None, :, d_src] * d[d_obs]
                        - src.normals[None, :, d_obs] * d[d_src]
                    )
                    / r
                )
                T[:, d_obs, :, d_src] = -(self.trac_C1 / r) * (t1 * drdn - t2)
                T[:, d_obs, :, d_src][too_close] = 0
        T *= (src.jacobians * src.quad_wts)[None, None, :, None]
        return T
