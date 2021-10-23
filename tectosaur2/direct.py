import numpy as np

def single_layer_matrix(obs_pts, src):

    dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
    dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2
    r2[r2 == 0] = 1
    G = (1.0 / (4 * np.pi)) * np.log(r2)
    G[r2 == 0] = 0

    return (G * src.jacobians * src.quad_wts[None, :])[:, None, :]


def double_layer_matrix(obs_pts, src):
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

    return (integrand * src.jacobians * src.quad_wts[None, :])[:, None, :]


def adjoint_double_layer_matrix(obs_pts, src):
    dx = obs_pts[:, None, 0] - src.pts[None, :, 0]
    dy = obs_pts[:, None, 1] - src.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2
    r2[r2 == 0] = 1

    out = np.empty((obs_pts.shape[0], 2, src.n_pts))
    out[:, 0, :] = dx
    out[:, 1, :] = dy

    C = -1.0 / (2 * np.pi * r2)
    C[r2 == 0] = 0

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :]


def hypersingular_matrix(obs_pts, src):
    dx = obs_pts[:, 0, None] - src.pts[None, :, 0]
    dy = obs_pts[:, 1, None] - src.pts[None, :, 1]
    r2 = dx ** 2 + dy ** 2
    r2[r2 == 0] = 1

    A = 2 * (dx * src.normals[None, :, 0] + dy * src.normals[None, :, 1]) / r2
    C = 1.0 / (2 * np.pi * r2)
    C[r2==0] = 0
    out = np.empty((obs_pts.shape[0], 2, src.n_pts))

    # The definition of the hypersingular kernel.
    # unscaled sigma_xz component
    out[:, 0, :] = src.normals[None, :, 0] - A * dx
    # unscaled sigma_xz component
    out[:, 1, :] = src.normals[None, :, 1] - A * dy

    # multiply by the scaling factor, jacobian and quadrature weights
    return out * (C * (src.jacobians * src.quad_wts[None, :]))[:, None, :]
