import numpy as np
import pytest

from tectosaur2.global_qbx import global_qbx_self
from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)
from tectosaur2.mesh import apply_interp_mat, unit_circle, upsample

# kernels = [single_layer]
kernels = [double_layer]
# kernels=[adjoint_double_layer]
# kernels = [hypersingular]
kernels = [single_layer, double_layer, adjoint_double_layer, hypersingular]


@pytest.mark.parametrize("K", kernels)
def test_nearfield_far(K):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    obs_pts = 2 * src.pts[:1]
    true = K._direct(obs_pts, src)
    true_v = np.sum(true * density[None, :, None], axis=1)

    for d_refine in [0.5, 3.0, 10.0]:
        est = np.zeros_like(true)

        pts_per_panel = [
            np.arange(obs_pts.shape[0], dtype=int) for i in range(src.n_panels)
        ]
        pts_starts = np.zeros(src.n_panels + 1, dtype=int)
        pts_starts[1:] = np.cumsum([p.shape[0] for p in pts_per_panel])
        pts_per_panel = np.concatenate(pts_per_panel)

        K._nearfield(est, obs_pts, src, pts_per_panel, pts_starts, 1.0, 3.0)
        est_v = np.sum(est * density[None, :, None], axis=1)

        np.testing.assert_allclose(est_v, true_v, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_integrate_near(K):
    src = unit_circle(control_points=np.array([[1, 0, 0, 0.05]]))
    obs_pts = 1.14 * src.pts[1:4]
    src_high, interp_mat = upsample(src, 5)
    true = np.transpose(
        apply_interp_mat(K._direct(obs_pts, src_high), interp_mat), (0, 2, 1)
    )

    est, report = K.integrate(
        obs_pts, src, d_refine=3.0, d_up=4.0, d_qbx=0.0, tol=1e-14, return_report=True
    )
    assert report["n_qbx"] == 0

    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_global_qbx(K):
    src = unit_circle()
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K._direct(obs_pts, src_high), interp_mat)

    density = np.cos(src.pts[:, 0])
    true_v = np.transpose(true, (0, 2, 1)).dot(density)

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

    local_qbx, report = K.integrate(
        src.pts,
        src,
        d_cutoff=100.0,
        max_p=3,
        tol=1e-13,
        on_src_direction=-1.0,
        return_report=True,
    )
    local_v = local_qbx.dot(density)

    np.testing.assert_allclose(local_v, global_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K", kernels)
def test_integrate_self(K):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    global_qbx = global_qbx_self(K, src, p=10, direction=1.0, kappa=3)
    global_v = global_qbx.dot(density)

    local_qbx, report = K.integrate(src.pts, src, return_report=True)
    local_v = local_qbx.dot(density)

    tol = 1e-13
    np.testing.assert_allclose(local_v, global_v, rtol=tol, atol=tol)
