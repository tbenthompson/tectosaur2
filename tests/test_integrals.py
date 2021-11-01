import numpy as np
import pytest
import sympy as sp

from tectosaur2.global_qbx import global_qbx_self
from tectosaur2.integrate import Integral, integrate
from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)
from tectosaur2.mesh import (
    apply_interp_mat,
    gauss_rule,
    stage1_refine,
    unit_circle,
    upsample,
)

# kernels = [single_layer]
# kernels = [double_layer]
# kernels=[adjoint_double_layer]
# kernels = [hypersingular]
kernels = [single_layer, double_layer, adjoint_double_layer, hypersingular]


@pytest.mark.parametrize("K", kernels)
def test_nearfield_far(K):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    obs_pts = 2 * src.pts[:1]
    true = K.direct(obs_pts, src)
    true_v = true.dot(density)

    for d_refine in [0.5, 3.0, 10.0]:

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


@pytest.mark.parametrize("K", kernels)
def test_integrate_near(K):
    src = unit_circle(control_points=np.array([[1, 0, 0, 0.05]]))
    obs_pts = 1.14 * src.pts[1:4]
    src_high, interp_mat = upsample(src, 5)
    true = apply_interp_mat(K.direct(obs_pts, src_high), interp_mat)

    term = Integral(K=K, src=src, d_refine=3.0, d_up=4.0, d_qbx=0.0)
    mats, report = integrate(obs_pts, term, tol=1e-14, return_reports=True)
    assert report[0]["n_qbx"] == 0

    np.testing.assert_allclose(mats[0], true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_global_qbx(K):
    src = unit_circle()
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K.direct(obs_pts, src_high), interp_mat)
    density = np.cos(src.pts[:, 0])
    true_v = true.dot(density)

    # p = 16, kappa = 4 are the minimal parameters for rtol=1e-13
    est = global_qbx_self(
        K, src, p=16, direction=-1.0, kappa=4, obs_pt_normal_offset=-0.07
    )
    est_v = est.dot(density)
    np.testing.assert_allclose(est_v, true_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K", kernels)
def test_integrate_can_do_global_qbx(K):
    # If we set d_cutoff very large, then integrate does a global QBX
    # integration. Except the order adaptive criterion fails. So, this
    # test only works for p<=3
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    global_qbx = global_qbx_self(K, src, p=3, direction=-1.0, kappa=10)
    global_v = global_qbx.dot(density)

    local_qbx, reports = integrate(
        src.pts,
        Integral(
            src=src,
            K=K,
            d_cutoff=100.0,
            max_p=3,
            on_src_direction=-1.0,
        ),
        return_reports=True,
    )
    # from tectosaur2.integrate import integrate_term
    # integrate_term(K, obs_pts, )
    local_v = local_qbx[0].dot(density)

    np.testing.assert_allclose(local_v, global_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K", kernels)
def test_integrate_self(K):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])
    print("HI")
    print("HI")
    print("HI")
    print("HI")
    print("HI")

    global_qbx = global_qbx_self(K, src, p=10, direction=1.0, kappa=3)
    global_v = global_qbx.dot(density)

    local_qbx, report = integrate(src.pts, (src, K), return_reports=True)
    local_v = local_qbx[0].dot(density)

    tol = 1e-13
    if K is hypersingular:
        tol = 1e-12
    np.testing.assert_allclose(local_v, global_v, rtol=tol, atol=tol)


def test_fault_surface():
    t = sp.var("t")
    fault, free = stage1_refine(
        [(t, t * 0, (t + 1) * -0.5), (t, -t, 0 * t)],
        gauss_rule(6),
        control_points=np.array([(0, 0, 0, 0.5)]),
    )
    (A, B) = integrate(free.pts, (free, double_layer), (fault, double_layer))
