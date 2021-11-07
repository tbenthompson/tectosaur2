import numpy as np
import pytest
import sympy as sp

from tectosaur2.global_qbx import global_qbx_self
from tectosaur2.integrate import integrate_term
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
    gauss_rule,
    refine_surfaces,
    unit_circle,
    upsample,
)

# kernel_types = [SingleLayer]
# kernel_types = [DoubleLayer]
# kernel_types = [AdjointDoubleLayer]
# kernel_types = [Hypersingular]
kernel_types = [SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular]


@pytest.mark.parametrize("K_type", kernel_types)
def test_nearfield_far(K_type):
    K = K_type()

    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    obs_pts = 2 * src.pts[:1]
    true = K.direct(obs_pts, src)
    true_v = true.dot(density)

    pts_per_panel = [
        np.arange(obs_pts.shape[0], dtype=int) for i in range(src.n_panels)
    ]
    pts_starts = np.zeros(src.n_panels + 1, dtype=int)
    pts_starts[1:] = np.cumsum([p.shape[0] for p in pts_per_panel])
    pts_per_panel = np.concatenate(pts_per_panel)

    est = np.zeros((obs_pts.shape[0], src.n_pts, K.ndim))
    K.nearfield(est, obs_pts, src, pts_per_panel, pts_starts, 1.0, 3.0)
    est_v = np.transpose(est, (0, 2, 1)).dot(density)

    np.testing.assert_allclose(est_v, true_v, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_near(K_type):
    src = unit_circle(control_points=np.array([[1, 0, 0, 0.05]]))
    obs_pts = 1.04 * src.pts[1:4]
    src_high, interp_mat = upsample(src, 7)
    true = apply_interp_mat(K_type().direct(obs_pts, src_high), interp_mat)

    K = K_type(d_up=4.0, d_qbx=0.0)
    mats, report = integrate_term(K, obs_pts, src, tol=1e-14, return_report=True)
    assert report["n_qbx"] == 0

    np.testing.assert_allclose(mats[0], true, rtol=5e-14, atol=5e-14)


@pytest.mark.parametrize("K_type", kernel_types)
def test_global_qbx(K_type):
    src = unit_circle()
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K_type().direct(obs_pts, src_high), interp_mat)
    density = np.cos(src.pts[:, 0])
    true_v = true.dot(density)

    # p = 16, kappa = 4 are the minimal parameters for rtol=1e-13
    est = global_qbx_self(
        K_type(), src, p=16, direction=-1.0, kappa=4, obs_pt_normal_offset=-0.07
    )
    est_v = est.dot(density)
    np.testing.assert_allclose(est_v, true_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_can_do_global_qbx(K_type):
    # If we set d_cutoff very large, then integrate does a global QBX
    # integration. Except the order adaptive criterion fails. So, this
    # test only works for p<=3
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    global_qbx = global_qbx_self(K_type(), src, p=3, direction=-1.0, kappa=10)
    global_v = global_qbx.dot(density)

    local_qbx, report = integrate_term(
        K_type(d_cutoff=100.0, max_p=3),
        src.pts,
        src,
        limit_direction=-1.0,
        return_report=True,
    )
    assert report["n_nearfield"] == 0
    local_v = local_qbx[0].dot(density)

    np.testing.assert_allclose(local_v, global_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K_type", kernel_types)
def test_integrate_self(K_type):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    global_qbx = global_qbx_self(K_type(), src, p=10, direction=1.0, kappa=3)
    global_v = global_qbx.dot(density)

    tol = 1e-13
    if K_type is Hypersingular:
        tol = 1e-12
    local_qbx, report = integrate_term(
        K_type(d_cutoff=4.0), src.pts, src, tol=tol, return_report=True
    )
    local_v = local_qbx[0].dot(density)

    print(report["p"])
    print(report["integration_failed"])
    print(report["n_subsets"])
    print(local_v[:10], global_v[:10])
    np.testing.assert_allclose(local_v, global_v, rtol=tol, atol=tol)


def test_fault_surface():
    t = sp.var("t")
    fault, free = refine_surfaces(
        [(t, t * 0, (t + 1) * -0.5), (t, -t * 2, 0 * t)],
        gauss_rule(6),
        control_points=np.array([(0, 0, 0, 0.1)]),
    )
    (A, B) = integrate_term(double_layer, free.pts, free, fault)
    slip = np.ones(B.shape[2])
    lhs = np.eye(A.shape[0]) + A[:, 0, :]
    surf_disp = np.linalg.inv(lhs).dot(-B[:, 0, :].dot(slip))

    # from tectosaur2.mesh import pts_grid

    # nobs = 50
    # zoomx = [-1.5, 1.5]
    # zoomy = [-3, 0]
    # xs = np.linspace(*zoomx, nobs)
    # ys = np.linspace(*zoomy, nobs)
    # obs_pts = pts_grid(xs, ys)
    # TODO: add test for interior displacement and interior stress. OR ADD THE NOTEBOOK AS A TEST.w
    # Ai, Bi = integrate_term(double_layer, obs_pts, free, fault)
    # interior_disp = Ai[:, 0, :].dot(surf_disp) + Bi[:, 0, :].dot(slip)

    (C, D) = integrate_term(hypersingular, fault.pts, free, fault)
    fault_stress = C.dot(surf_disp) + D.dot(slip)

    # np.save('tests/test_fault_surface.npy', (surf_disp, fault_stress))

    cmp_surf_disp, cmp_fault_stress = np.load(
        "tests/test_fault_surface.npy", allow_pickle=True
    )
    np.testing.assert_allclose(surf_disp, cmp_surf_disp)
    np.testing.assert_allclose(fault_stress, cmp_fault_stress, atol=1e-10)

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
