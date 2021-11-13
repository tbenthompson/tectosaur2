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
    return ((z - z0) ** m / (r ** m))[:, None]


def deriv_eval(obs_pts, exp_centers, r, m):
    z = obs_pts[:, 0] + obs_pts[:, 1] * 1j
    z0 = exp_centers[:, 0] + exp_centers[:, 1] * 1j
    return (-m * (z - z0) ** (m - 1) / (r ** m))[:, None]


def single_layer_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    return np.real(
        base_eval(obs_pts, exp_centers, r, m)
        * base_expand(exp_centers, src_pts, src_normals, r, m)
    )[:, None, :, None]


def double_layer_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    return np.real(
        base_eval(obs_pts, exp_centers, r, m)
        * deriv_expand(exp_centers, src_pts, src_normals, r, m)
    )[:, None, :, None]


def adjoint_double_layer_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    out = np.empty((obs_pts.shape[0], 2, src_pts.shape[0], 1))
    term = deriv_eval(obs_pts, exp_centers, r, m) * base_expand(
        exp_centers, src_pts, src_normals, r, m
    )
    out[:, 0, :, 0] = np.real(term)
    out[:, 1, :, 0] = -np.imag(term)
    return out


def hypersingular_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    out = np.empty((obs_pts.shape[0], 2, src_pts.shape[0], 1))
    term = deriv_eval(obs_pts, exp_centers, r, m) * deriv_expand(
        exp_centers, src_pts, src_normals, r, m
    )
    out[:, 0, :, 0] = np.real(term)
    out[:, 1, :, 0] = -np.imag(term)
    return out


def elastic_U_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):

    z = obs_pts[:, None, 0] + obs_pts[:, None, 1] * 1j
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, None, 0] + exp_centers[:, None, 1] * 1j

    # See equation 4.44 from Liu's FMM BEM book, page 95.
    term = np.empty((z0.shape[0], 2, src_pts.shape[0], 2))
    kappa = 3 - 4 * kernel.poisson_ratio
    C = 1.0 / (4 * np.pi * kernel.shear_modulus * (1 + kappa))
    ratio = (z - z0) / (w - z0)
    for d_src in range(2):
        tw = (d_src == 0) + (d_src == 1) * 1j
        G = -np.log(w - z0) if m == 0 else (1.0 / m) * (ratio ** m)
        Gp = -(ratio ** m) / (w - z0)
        t1 = kappa * (G * tw + np.conjugate(G) * tw)
        t2 = -(w - z) * np.conjugate(tw) * np.conjugate(Gp)
        V = C * (t1 + t2)
        if m == 0:
            V += kernel.disp_C1 * 0.5 * tw
        term[:, 0, :, d_src] = np.real(V)
        term[:, 1, :, d_src] = np.imag(V)
    return term


def elastic_T_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    z = obs_pts[:, None, 0] + obs_pts[:, None, 1] * 1j
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, None, 0] + exp_centers[:, None, 1] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j

    kappa = 3 - 4 * kernel.poisson_ratio
    C = -1.0 / (2 * np.pi * (1 + kappa))
    ratio = (z - z0) / (w - z0)
    term = np.empty((z0.shape[0], 2, src_pts.shape[0], 2))
    for d_src in range(2):
        uw = (d_src == 0) + (d_src == 1) * 1j
        Gp = -(ratio ** m) / (w - z0)
        Gpp = (m + 1) * (ratio ** m) / ((w - z0) ** 2)
        t1 = kappa * Gp * nw * uw
        t2 = -(w - z) * np.conjugate(Gpp) * np.conjugate(nw) * np.conjugate(uw)
        t3 = np.conjugate(Gp) * (nw * np.conjugate(uw) + np.conjugate(nw) * uw)
        V = C * (t1 + t2 + t3)
        term[:, 0, :, d_src] = np.real(V)
        term[:, 1, :, d_src] = np.imag(V)
    return term


def elastic_A_term(kernel, obs_pts, exp_centers, r, src_pts, src_normals, m):
    z = obs_pts[:, None, 0] + obs_pts[:, None, 1] * 1j
    w = src_pts[None, :, 0] + src_pts[None, :, 1] * 1j
    z0 = exp_centers[:, None, 0] + exp_centers[:, None, 1] * 1j
    nw = src_normals[None, :, 0] + src_normals[None, :, 1] * 1j

    kappa = 3 - 4 * kernel.poisson_ratio
    C = -1.0 / (2 * np.pi * (1 + kappa))
    ratio = (z - z0) / (w - z0)
    term = np.empty((z0.shape[0], 2, src_pts.shape[0], 2))
    for d_src in range(2):
        uw = (d_src == 0) + (d_src == 1) * 1j
        Gp = -(ratio ** m) / (w - z0)
        Gpp = (m + 1) * (ratio ** m) / ((w - z0) ** 2)
        t1 = kappa * Gp * nw * uw
        t2 = -(w - z) * np.conjugate(Gpp) * np.conjugate(nw) * np.conjugate(uw)
        t3 = np.conjugate(Gp) * (nw * np.conjugate(uw) + np.conjugate(nw) * uw)
        V = C * (t1 + t2 + t3)
        term[:, 0, :, d_src] = np.real(V)
        term[:, 1, :, d_src] = np.imag(V)
    return term


def global_qbx_self(kernel, src, p, direction, kappa, obs_pt_normal_offset=0.0):
    """
    For testing purposes only!

    Points that are not on a surface can be tested by using an upsampled
    direct quadrature. Points that are on the surface cannot be tested that
    way. In order to test the more robust local QBX implementation, I've
    built this simple global QBX implementation.
    """
    term_fnc = globals()[kernel.name + "_term"]
    obs_pts = src.pts + obs_pt_normal_offset * src.normals

    L = np.repeat(src.panel_length, src.panel_order)
    exp_centers = src.pts + direction * src.normals * L[:, None] * 0.5
    exp_rs = L * 0.5

    src_high, interp_mat_high = upsample(src, kappa)

    out = np.zeros(
        (obs_pts.shape[0], kernel.obs_dim, src_high.n_pts, kernel.src_dim),
        dtype=np.float64,
    )
    for m in range(p + 1):
        term = term_fnc(
            kernel, obs_pts, exp_centers, exp_rs, src_high.pts, src_high.normals, m
        )
        out += term * (src_high.quad_wts * src_high.jacobians)[None, None, :, None]

    return apply_interp_mat(out, interp_mat_high)
