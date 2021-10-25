import numpy as np
import pytest
import scipy.spatial

from tectosaur2._ext import identify_nearfield_panels
from tectosaur2.global_qbx import global_qbx_self
from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)
from tectosaur2.mesh import apply_interp_mat, stage2_refine, unit_circle, upsample

kernels = [single_layer]
# kernels=[double_layer]
# kernels=[adjoint_double_layer]
# kernels = [hypersingular]
kernels = [single_layer, double_layer, adjoint_double_layer, hypersingular]


def all_panels(n_obs, n_src_panels):
    panels = [np.arange(n_src_panels, dtype=int) for i in range(n_obs)]
    panel_starts = np.zeros(n_obs + 1, dtype=int)
    panel_starts[1:] = np.cumsum([p.shape[0] for p in panels])
    panels = np.concatenate(panels)
    return panels, panel_starts


@pytest.mark.parametrize("K", kernels)
def test_nearfield_far(K):
    src = unit_circle()
    obs_pts = 2 * src.pts
    true = K._direct(obs_pts, src)

    est = np.zeros_like(true)
    panels, panel_starts = all_panels(obs_pts.shape[0], src.n_panels)
    K._nearfield(est, obs_pts, src, panels, panel_starts, 1.0)

    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_nearfield_near(K):
    src = unit_circle(control_points=np.array([[1, 0, 0, 0.05]]))
    obs_pts = 1.07 * src.pts[1:4]
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K._direct(obs_pts, src_high), interp_mat)

    src_local_refine, interp_mat, refinement_plan = stage2_refine(src, obs_pts, kappa=2)
    refinement_map = np.unique(refinement_plan[:, 0].astype(int), return_inverse=True)[
        1
    ]
    panels, panel_starts = identify_nearfield_panels(
        obs_pts,
        np.full(obs_pts.shape[0], 5.0),
        scipy.spatial.KDTree(src.pts),
        src.panel_order,
        refinement_map,
    )
    est = np.zeros((obs_pts.shape[0], src_local_refine.n_pts, K.ndim))
    K._nearfield(est, obs_pts.copy(), src_local_refine, panels, panel_starts, 1.0)
    est = apply_interp_mat(est, interp_mat)

    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_global_qbx(K):
    src = unit_circle()
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K._direct(obs_pts, src_high), interp_mat)

    density = np.cos(src.pts[:, 0])
    true_v = np.transpose(true, (0, 2, 1)).dot(density)

    # p = 16, kappa = 5 are the minimal parameters for rtol=1e-13
    est = global_qbx_self(
        K, src, p=16, direction=-1.0, kappa=4, obs_pt_normal_offset=-0.07
    )
    est_v = est.dot(density)
    np.testing.assert_allclose(est_v, true_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("K", kernels)
def test_integrate_vs_global_qbx(K):
    src = unit_circle()
    density = np.cos(src.pts[:, 0])

    global_qbx = global_qbx_self(K, src, 50, 1.0, 10)
    global_v = global_qbx.dot(density)

    local_qbx, report = K.integrate(src.pts, src, return_report=True)
    local_v = local_qbx.dot(density)
    # print(report['p'])
    # print(report['kappa_too_small'])

    np.testing.assert_allclose(local_v, global_v, rtol=1e-13, atol=1e-13)
