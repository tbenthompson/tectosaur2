import numpy as np

from .mesh import apply_interp_mat, upsample


def base_expand(exp_centers, src_pts, src_normals, r, m):
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


def global_qbx_self(kernel, src, p, direction, kappa, obs_pt_normal_offset=0.0):
    """
    For testing purposes only!

    Points that are not on a surface can be tested by using an upsampled
    direct quadrature. Points that are on the surface cannot be tested that
    way. In order to test the more robust local QBX implementation, I've
    built this simple global QBX implementation.
    """
    exp_fnc = deriv_expand if kernel.exp_deriv else base_expand
    eval_fnc = deriv_eval if kernel.eval_deriv else base_eval

    obs_pts = src.pts + obs_pt_normal_offset * src.normals

    L = np.repeat(src.panel_length, src.panel_order)
    exp_centers = src.pts + direction * src.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    src_high, interp_mat_high = upsample(src, kappa)

    exp_terms = []
    for i in range(p):
        K = exp_fnc(exp_centers, src_high.pts, src_high.normals, exp_rs, i)
        integral = K * (src_high.quad_wts[None, :] * src_high.jacobians[None, :])
        exp_terms.append(integral)

    eval_terms = []
    for i in range(p):
        eval_terms.append(eval_fnc(obs_pts, exp_centers, exp_rs, i))

    out = np.zeros((obs_pts.shape[0], src_high.n_pts, kernel.ndim), dtype=np.float64)
    for i in range(p):
        out[:, :, 0] += np.real(exp_terms[i] * eval_terms[i][:, None])
        if kernel.ndim == 2:
            out[:, :, 1] -= np.imag(exp_terms[i] * eval_terms[i][:, None])

    return np.transpose(apply_interp_mat(out, interp_mat_high), (0, 2, 1))
