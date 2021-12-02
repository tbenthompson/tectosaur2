import warnings

import numpy as np
import pytest
import sympy as sp

from tectosaur2 import gauss_rule, integrate_term, refine_surfaces, tensor_dot
from tectosaur2._ext import nearfield_integrals
from tectosaur2.elastic2d import ElasticA, ElasticH, ElasticT, ElasticU
from tectosaur2.global_qbx import global_qbx_self
from tectosaur2.laplace2d import (
    AdjointDoubleLayer,
    DoubleLayer,
    Hypersingular,
    SingleLayer,
    double_layer,
    hypersingular,
)
from tectosaur2.mesh import (
    apply_interp_mat,
    barycentric_deriv,
    build_interp_wts,
    unit_circle,
    upsample,
)


def elasticU(**kwargs):
    return ElasticU(0.3, **kwargs)


def elasticT(**kwargs):
    return ElasticT(0.2, **kwargs)


def elasticA(**kwargs):
    return ElasticA(0.2, **kwargs)


def elasticH(**kwargs):
    return ElasticH(0.35, **kwargs)


kernel_types = [
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    elasticU,
    elasticT,
    elasticA,
    elasticH,
]


def test_barycentric():
    qx, qw = gauss_rule(5)
    interp_wts = build_interp_wts(qx)
    f = lambda x: x ** 2
    eval_x = np.linspace(-1, 1, 100)
    deriv = barycentric_deriv(eval_x, qx, interp_wts, f(qx))
    correct = 2 * eval_x
    np.testing.assert_allclose(deriv, correct)


@pytest.mark.parametrize("K_type", kernel_types)
def test_nearfield_far(K_type):
    src = unit_circle(gauss_rule(12), max_curvature=100)

    obs_pts = 2 * src.pts[3:4]
    true = K_type().direct(obs_pts, src)

    pts_per_panel = [
        np.arange(obs_pts.shape[0], dtype=int) for i in range(src.n_panels)
    ]
    pts_starts = np.zeros(src.n_panels + 1, dtype=int)
    pts_starts[1:] = np.cumsum([p.shape[0] for p in pts_per_panel])
    pts_per_panel = np.concatenate(pts_per_panel)

    K = K_type()
    est_compact = np.zeros((obs_pts.shape[0], src.n_pts, K.obs_dim * K.src_dim))

    n_subsets = nearfield_integrals(
        K.name,
        K.parameters,
        est_compact,
        obs_pts,
        src,
        src.qx,
        src.qw,
        src.qw,
        pts_per_panel,
        pts_starts,
        1.0,
        3.0,
        adaptive=False,
    )
    est = np.transpose(
        est_compact.reshape((obs_pts.shape[0], src.n_pts, K.obs_dim, K.src_dim)),
        (0, 2, 1, 3),
    )

    assert n_subsets[0] == src.n_panels
    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_near(K_type):
    src = unit_circle(gauss_rule(12), control_points=np.array([[1, 0, 0.5, 0.05]]))
    obs_pts = 1.04 * src.pts[1:4]
    src_high, interp_mat = upsample(src, 7)
    true = apply_interp_mat(K_type().direct(obs_pts, src_high), interp_mat)

    mat, report = integrate_term(
        K_type(d_qbx=0.0), obs_pts, src, tol=1e-14, return_report=True
    )
    assert report["n_qbx"] == 0
    np.testing.assert_allclose(mat, true, rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize("K_type", kernel_types)
def test_global_qbx(K_type):
    src = unit_circle(gauss_rule(12))
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K_type().direct(obs_pts, src_high), interp_mat)
    density = np.stack((np.cos(src.pts[:, 0]), np.sin(src.pts[:, 0])), axis=1)[
        :, : K_type().src_dim
    ]
    true_v = tensor_dot(true, density)

    # p = 20, kappa = 5 are the minimal parameters for rtol=1e-13
    est = global_qbx_self(
        K_type(), src, p=20, direction=-1.0, kappa=5, obs_pt_normal_offset=-0.07
    )
    est_v = tensor_dot(est, density)
    np.testing.assert_allclose(est_v, true_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_can_do_global_qbx(K_type):
    # If we set d_cutoff very large, then integrate_term does a global QBX
    # integration. Except the order adaptive criterion fails. So, this
    # test only works for p<=3.
    src = unit_circle(gauss_rule(8))
    density = np.stack((np.cos(src.pts[:, 0]), np.sin(src.pts[:, 0])), axis=1)[
        :, : K_type().src_dim
    ]

    global_qbx, global_report = global_qbx_self(
        K_type(), src, p=3, direction=-1.0, kappa=10, return_report=True
    )
    global_v = tensor_dot(global_qbx, density)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_qbx, report = integrate_term(
            K_type(d_cutoff=100.0, max_p=3),
            src.pts,
            src,
            limit_direction=-1.0,
            return_report=True,
        )
    assert report["n_nearfield"] == 0
    local_v = tensor_dot(local_qbx, density)

    np.testing.assert_allclose(report["exp_rs"], global_report["exp_rs"], atol=1e-15)
    np.testing.assert_allclose(
        report["exp_centers"], global_report["exp_centers"], atol=1e-15
    )
    np.testing.assert_allclose(report["p"], 3)
    np.testing.assert_allclose(local_v, global_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_self(K_type):
    src = unit_circle(gauss_rule(12))
    density = np.stack((np.cos(src.pts[:, 0]), np.sin(src.pts[:, 0])), axis=1)[
        :, : K_type().src_dim
    ]

    global_qbx = global_qbx_self(K_type(), src, p=10, direction=1.0, kappa=3)
    global_v = tensor_dot(global_qbx, density)

    tol = 1e-13
    if K_type is Hypersingular:
        tol = 1e-12
    elif K_type is elasticH:
        tol = 1e-11
    local_qbx, report = integrate_term(
        K_type(d_cutoff=4.0), src.pts, src, tol=tol, return_report=True
    )
    local_v = tensor_dot(local_qbx, density)

    np.testing.assert_allclose(local_v, global_v, rtol=tol, atol=tol)


def test_control_points():
    C = unit_circle(gauss_rule(5))
    assert C.n_panels == 16
    C = unit_circle(gauss_rule(5), control_points=np.array([[1, 0, 0, 0.25]]))
    assert C.n_panels == 18
    C = unit_circle(gauss_rule(5), control_points=np.array([[1, 0, 0, 0.1]]))
    assert C.n_panels == 26

    C = unit_circle(gauss_rule(5), control_points=np.array([[1, 0, 3, 0.25]]))
    assert np.all(C.panel_length < 0.25)
    assert C.n_panels == 32

    C = unit_circle(
        gauss_rule(5), control_points=np.array([[1, 0, 3, 0.25], [1, 0, 0, 0.1]])
    )
    assert np.all(C.panel_length < 0.25)
    assert C.n_panels == 35


def test_safety_mode():
    # To test safety mode, we compute the stress resulting from a displacement
    # field with two linear segments with a kink between them. The sudden change
    # in derivative will cause the stress field to have a step function. This
    # will fail to integrate correctly without safety mode and will lead to
    # ringing in the stress component. With safety mode, there should be a
    # perfect step function in stress.
    t = sp.var("t")
    surf = refine_surfaces(
        [(t, t, 0 * t)], gauss_rule(12), control_points=[(0, 0, 0, 0.2)]
    )
    displacement = 1 - np.abs(surf.pts[:, 0])
    mat = integrate_term(
        hypersingular,
        surf.pts,
        surf,
        tol=1e-11,
        safety_mode=True,
        singularities=[(-1, 0), (1, 0)],
    )
    stress = -2 * tensor_dot(mat, displacement)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(surf.pts[:, 0], displacement)
    # plt.subplot(1, 2, 2)
    # plt.plot(surf.pts[:, 0], stress[:, 0])
    # plt.plot(surf.pts[:, 0], stress[:, 1])
    # plt.show()
    # plt.plot()
    # plt.plot(stress[:,0]+np.sign(surf.pts[:,0]) * 0.5)
    # plt.show()
    np.testing.assert_allclose(stress[:, 0], -np.sign(surf.pts[:, 0]))


def test_laplace_obs_derivatives():
    """
    The AdjointDoubleLayer kernel should be equal to the stress from the displacement from
    the SingleLayer kernel. The same should be true of the Hypersingula kernel with
    respect to the DoubleLayer kernel.
    """
    t = sp.var("t")
    line = refine_surfaces(
        [(t, 1 * t, 0.0 * t)],
        gauss_rule(10),
        control_points=np.array([[0, 0, 100, 0.5]]),
    )
    delta = 0.0001
    obs_pt = np.array([[0.5, 0.5]])
    obs_pts_fd = np.array(
        [
            [obs_pt[0, 0] - delta, obs_pt[0, 1]],
            [obs_pt[0, 0] + delta, obs_pt[0, 1]],
            [obs_pt[0, 0], obs_pt[0, 1] - delta],
            [obs_pt[0, 0], obs_pt[0, 1] + delta],
        ]
    )

    for K_base, K_deriv in [
        (SingleLayer, AdjointDoubleLayer),
        (DoubleLayer, Hypersingular),
    ]:
        Iv = integrate_term(K_base(), obs_pts_fd, line)
        vs = Iv[:, 0, :, 0].sum(axis=1)
        fd_deriv_vx = (vs[1] - vs[0]) / (2 * delta)
        fd_deriv_vy = (vs[3] - vs[2]) / (2 * delta)
        Id = integrate_term(K_deriv(), obs_pt, line)
        deriv_vs = Id[0, :, :, 0].sum(axis=1)
        np.testing.assert_allclose(deriv_vs, [fd_deriv_vx, fd_deriv_vy], atol=1e-3)


def test_elastic_obs_derivatives():
    """
    The ElasticA kernel should be equal to the stress from the displacement from
    the ElasticU kernel. The same should be true of the ElasticH kernel with
    respect to the ElasticT kernel.
    """
    t = sp.var("t")
    line = refine_surfaces(
        [(t, -1 * t, 0.0 * t)],
        gauss_rule(10),
        control_points=np.array([[0, 0, 100, 0.5]]),
    )
    delta = 0.0001
    obs_pt = np.array([[0.1, -1.5]])
    obs_pts_fd = np.array(
        [
            [obs_pt[0, 0] - delta, obs_pt[0, 1]],
            [obs_pt[0, 0] + delta, obs_pt[0, 1]],
            [obs_pt[0, 0], obs_pt[0, 1] - delta],
            [obs_pt[0, 0], obs_pt[0, 1] + delta],
        ]
    )

    nu = 0.25
    mu = 1.0
    lam = 2 * mu * nu / (1 - 2 * nu)
    for K_base, K_deriv in [(ElasticU, ElasticA), (ElasticT, ElasticH)]:
        Iv = integrate_term(K_base(nu), obs_pts_fd, line)
        vs = Iv[:, :, :, 0].sum(axis=2)

        dux_dx = (vs[1, 0] - vs[0, 0]) / (2 * delta)
        duy_dx = (vs[1, 1] - vs[0, 1]) / (2 * delta)
        dux_dy = (vs[3, 0] - vs[2, 0]) / (2 * delta)
        duy_dy = (vs[3, 1] - vs[2, 1]) / (2 * delta)

        exx = dux_dx
        eyy = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)

        trace_e = exx + eyy
        sxx = 2 * exx + lam * trace_e
        syy = 2 * eyy + lam * trace_e
        sxy = 2 * exy

        Id = integrate_term(K_deriv(nu), obs_pt, line)
        deriv_vs = Id[0, :, :, 0].sum(axis=1)
        np.testing.assert_allclose(deriv_vs, [sxx, syy, sxy], atol=1e-3)


def test_fault_surface():
    t = sp.var("t")
    L = 2000
    fault, free = refine_surfaces(
        [(t, t * 0, (t + 1) * -0.5), (t, -t * L, 0 * t)],
        gauss_rule(6),
        control_points=np.array([(0, 0, 2, 0.1), (0, 0, 20, 1.0)]),
    )
    singularities = np.array([(-L, 0), (L, 0), (0, 0), (0, -1)])
    (A, B) = integrate_term(
        double_layer, free.pts, free, fault, singularities=singularities
    )
    slip = np.ones(B.shape[2])
    lhs = np.eye(A.shape[0]) + A[:, 0, :, 0]
    surf_disp = np.linalg.inv(lhs).dot(-B[:, 0, :, 0].dot(slip))

    (C, D) = integrate_term(
        hypersingular, fault.pts, free, fault, tol=1e-10, singularities=singularities
    )
    fault_stress = -tensor_dot(C, surf_disp) - tensor_dot(D, slip)

    cmp_surf_disp = -np.arctan(-1 / free.pts[:, 0]) / np.pi

    obsx = fault.pts[:, 0]
    obsy = fault.pts[:, 1]
    rp = obsx ** 2 + (obsy + 1) ** 2
    ri = obsx ** 2 + (obsy - 1) ** 2
    analytical_sxz = -(1.0 / (2 * np.pi)) * (((obsy + 1) / rp) - ((obsy - 1) / ri))
    analytical_syz = (1.0 / (2 * np.pi)) * ((obsx / rp) - (obsx / ri))

    cmp_fault_stress = np.stack((analytical_sxz, analytical_syz), axis=1)
    np.testing.assert_allclose(surf_disp, cmp_surf_disp, atol=1e-10)
    np.testing.assert_allclose(fault_stress, cmp_fault_stress, atol=1e-7)

    from tectosaur2.mesh import pts_grid

    nobs = 50
    zoomx = [-1.5, 1.5]
    zoomy = [-3, 0]
    xs = np.linspace(*zoomx, nobs)
    ys = np.linspace(*zoomy, nobs)
    obs_pts = pts_grid(xs, ys)
    Ai, Bi = integrate_term(double_layer, obs_pts, free, fault)
    interior_disp = -Ai[:, 0, :, 0].dot(surf_disp) - Bi[:, 0, :, 0].dot(slip)
    obsx = obs_pts[:, 0]
    obsy = obs_pts[:, 1]
    analytical_disp = (
        1.0
        / (2 * np.pi)
        * (np.arctan((obsy + 1) / obsx) - np.arctan((obsy - 1) / obsx))
    )
    np.testing.assert_allclose(interior_disp, analytical_disp, atol=1e-10)

    # visual comparison
    # import matplotlib.pyplot as plt
    # plt.title('disp')
    # plt.plot(free.pts[:,0], surf_disp, 'r-')
    # plt.plot(free.pts[:,0], cmp_surf_disp, 'b-')
    # plt.figure()
    # plt.title('sxz')
    # plt.plot(fault.pts[:,1], fault_stress[:,0], 'r-')
    # plt.plot(fault.pts[:,1], cmp_fault_stress[:,0], 'b-')
    # plt.figure()
    # plt.title('syz')
    # plt.plot(fault.pts[:,1], fault_stress[:,1], 'r-')
    # plt.plot(fault.pts[:,1], cmp_fault_stress[:,1], 'b-')
    # plt.show()
