import numpy as np

from .mesh import apply_interp_mat, upsample


def base_expand(exp_centers, src_pts, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    if m == 0:
        return np.log(w - z0) / (2 * np.pi)
    else:
        return -(r[:, None] ** m) / (m * (2 * np.pi) * (w - z0) ** m)


def deriv_expand(exp_centers, src_pts, src_normals, r, m):
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, 0, None] + exp_centers[:, 1, None] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j
    return nw * (r[:, None] ** m) / ((2 * np.pi) * (w - z0) ** (m + 1))


def base_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return (z - z0) ** m / (r ** m)


def deriv_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return -m * (z - z0) ** (m - 1) / (r ** m)


def global_qbx_self(src, p, direction=1, kappa=3):
    obs_pts = src.pts

    L = np.repeat(src.panel_length, src.panel_order)
    exp_centers = src.pts + direction * src.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    src_high, interp_mat_high = upsample(src, kappa)

    exp_terms = []
    for i in range(p):
        K = deriv_expand(exp_centers, src_high.pts, src_high.normals, exp_rs, i)
        integral = K * (src_high.quad_wts[None, :] * src_high.jacobians[None, :])
        exp_terms.append(integral)

    eval_terms = []
    for i in range(p):
        eval_terms.append(base_eval(obs_pts, exp_centers, exp_rs, i))

    kernel_ndim = 1  # exp_terms[0].shape[1]
    out = np.zeros((obs_pts.shape[0], kernel_ndim, src_high.n_pts), dtype=np.float64)
    for i in range(p):
        out[:, 0, :] += np.real(exp_terms[i] * eval_terms[i][:, None])

    return apply_interp_mat(out, interp_mat_high)
