import numpy as np
import pytest

from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)
from tectosaur2.mesh import apply_interp_mat, unit_circle, upsample

kernels = [single_layer, double_layer, adjoint_double_layer, hypersingular]
kernels = [double_layer]


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
    refinement_map = np.arange(src.n_panels)
    K._nearfield(est, obs_pts, src, panels, panel_starts, refinement_map, 1.0)

    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("K", kernels)
def test_nearfield_near(K):
    src = unit_circle()
    obs_pts = 1.07 * src.pts
    src_high, interp_mat = upsample(src, 10)
    true = apply_interp_mat(K._direct(obs_pts, src_high), interp_mat)

    est = K.integrate(obs_pts, src)
    np.testing.assert_allclose(est, true, rtol=1e-14, atol=1e-14)
