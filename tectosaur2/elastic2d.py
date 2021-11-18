import numpy as np

from .integrate import Kernel


class ElasticU(Kernel):
    name = "elastic_U"
    src_dim = 2
    obs_dim = 2

    def __init__(self, shear_modulus=1.0, poisson_ratio=0.25, **kwargs):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
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

    def __init__(self, shear_modulus=1.0, poisson_ratio=0.25, **kwargs):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.trac_C1 = 1.0 / (4 * np.pi * (1 - poisson_ratio))
        self.trac_C2 = 1 - 2.0 * poisson_ratio
        super().__init__(**kwargs)

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


class ElasticA(Kernel):
    name = "elastic_A"
    src_dim = 2
    obs_dim = 3

    def __init__(self, shear_modulus=1.0, poisson_ratio=0.25, **kwargs):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.trac_C1 = 1.0 / (4 * np.pi * (1 - poisson_ratio))
        self.trac_C2 = 1 - 2.0 * poisson_ratio
        super().__init__(**kwargs)

    def direct(self, obs_pts, src):
        d = [
            obs_pts[:, 0, None] - src.pts[None, :, 0],
            obs_pts[:, 1, None] - src.pts[None, :, 1],
        ]
        r2 = d[0] * d[0] + d[1] * d[1]
        too_close = r2 <= 1e-16
        r2[too_close] = 1
        r = np.sqrt(r2)
        dr = [d[0] / r, d[1] / r]

        A = np.empty((obs_pts.shape[0], 3, src.n_pts, 2))

        voigt_lookup = [[0, 2], [2, 1]]

        C = self.trac_C1 / r

        for d_obs1 in range(2):
            for d_obs2 in range(2):
                if d_obs1 == 1 and d_obs2 == 0:
                    continue
                voigt = voigt_lookup[d_obs1][d_obs2]
                for d_src in range(2):
                    t1 = self.trac_C2 * (
                        (d_obs1 == d_obs2) * dr[d_src]
                        + (d_obs2 == d_src) * dr[d_obs1]
                        - (d_obs1 == d_src) * dr[d_obs2]
                    )
                    t2 = 2 * dr[d_obs1] * dr[d_obs2] * dr[d_src]
                    A[:, voigt, :, d_src] = C * (t1 + t2)
                    A[:, voigt, :, d_src][too_close] = 0

        A *= (src.jacobians * src.quad_wts)[None, None, :, None]
        return A


class ElasticH(Kernel):
    name = "elastic_H"
    src_dim = 2
    obs_dim = 3

    def __init__(self, shear_modulus=1.0, poisson_ratio=0.25, **kwargs):
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.trac_C2 = 1 - 2.0 * poisson_ratio
        super().__init__(**kwargs)

    def direct(self, obs_pts, src):
        d = [
            obs_pts[:, 0, None] - src.pts[None, :, 0],
            obs_pts[:, 1, None] - src.pts[None, :, 1],
        ]
        r2 = d[0] * d[0] + d[1] * d[1]
        too_close = r2 <= 1e-16
        r2[too_close] = 1
        r = np.sqrt(r2)

        srcn = [src.normals[None, :, 0], src.normals[None, :, 1]]
        dr = [d[0] / r, d[1] / r]
        drdn = (d[0] * srcn[0] + d[1] * srcn[1]) / r

        H = np.empty((obs_pts.shape[0], 3, src.n_pts, 2))

        d_stress_lookup = [[0, 2], [2, 1]]

        # TODO: it would be nice to simplify so that there aren't these
        # temporary normal vectors here and just calculate the stress directly.
        # also, this is horribly optimized.
        C = self.shear_modulus / (2 * np.pi * (1 - self.poisson_ratio))
        for nd in range(2):
            obsn = [float(nd == 0), float(nd == 1)]
            drdm = (d[0] * obsn[0] + d[1] * obsn[1]) / r
            dmdn = obsn[0] * srcn[0] + obsn[1] * srcn[1]
            for d_obs in range(2):
                if nd == 1 and d_obs == 0:
                    continue
                d_stress = d_stress_lookup[nd][d_obs]
                for d_src in range(2):
                    t1 = (
                        2
                        * drdn
                        * (
                            self.trac_C2 * obsn[d_obs] * dr[d_src]
                            + self.poisson_ratio
                            * (obsn[d_src] * dr[d_obs] + (d_obs == d_src) * drdm)
                            - 4 * dr[d_obs] * dr[d_src] * drdm
                        )
                    )
                    t2 = self.trac_C2 * (
                        2 * srcn[d_src] * dr[d_obs] * drdm
                        + (d_obs == d_src) * dmdn
                        + srcn[d_obs] * obsn[d_src]
                    )
                    t3 = (
                        2
                        * self.poisson_ratio
                        * (
                            srcn[d_obs] * dr[d_src] * drdm
                            + dmdn * dr[d_obs] * dr[d_src]
                        )
                    )
                    t4 = -(1 - 4 * self.poisson_ratio) * srcn[d_src] * obsn[d_obs]
                    H[:, d_stress, :, d_src] = (C / r2) * (t1 + t2 + t3 + t4)
                    H[:, d_stress, :, d_src][too_close] = 0

        H *= (src.jacobians * src.quad_wts)[None, None, :, None]
        return H
