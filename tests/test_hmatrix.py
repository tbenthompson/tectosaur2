import numpy as np
import pytest
import sympy as sp
from kernels import kernel_types

from tectosaur2 import gauss_rule, integrate_term, panelize_symbolic_surface, tensor_dot
from tectosaur2.elastic2d import elastic_h, elastic_u
from tectosaur2.hmatrix.hmatrix import (
    TempSurface,
    aca_blocks,
    build_tree,
    svd_blocks,
    toy_aca_blocks,
    traverse,
)
from tectosaur2.laplace2d import hypersingular, single_layer
from tectosaur2.mesh import unit_circle


@pytest.mark.parametrize(
    "K", [single_layer, hypersingular, elastic_u(0.25), elastic_h(0.25)]
)
def test_aca(K):
    t = sp.var("t")
    surf = panelize_symbolic_surface(t, 0 * t, 2.5 * t, gauss_rule(6), n_panels=100)
    exact_mat = K.direct(surf.pts, surf)[:, :, :, :].reshape(
        (K.obs_dim * surf.n_pts, K.src_dim * surf.n_pts)
    )
    obs_tree = build_tree(surf.pts, np.zeros(surf.n_pts), min_pts_per_box=30)
    src_tree = obs_tree

    direct, approx = traverse(obs_tree.root, src_tree.root)
    tree_surf = TempSurface(
        surf.pts[src_tree.ordered_idxs],
        surf.normals[src_tree.ordered_idxs],
        surf.quad_wts[src_tree.ordered_idxs],
        surf.jacobians[src_tree.ordered_idxs],
    )

    tol = 1e-6

    A = toy_aca_blocks(K, tree_surf.pts, tree_surf, approx, tol, deterministic=True)
    A2 = aca_blocks(
        K,
        tree_surf.pts,
        tree_surf,
        approx,
        np.array([tol] * len(approx)),
        deterministic=True,
    )

    n_svd = 3
    Asvd = svd_blocks(
        K, tree_surf.pts, tree_surf, approx[:n_svd], np.array([tol] * n_svd)
    )

    for i, (obs_node, src_node) in enumerate(approx):
        exact_block = exact_mat[
            K.obs_dim * obs_node.idx_start : K.obs_dim * obs_node.idx_end,
            K.src_dim * src_node.idx_start : K.src_dim * src_node.idx_end,
        ]

        if i < n_svd:
            # SVD is expensive so we'll only test the first three approx blocks.
            Usvd, Vsvd = Asvd[i]
            svd_err = np.sqrt(np.sum((exact_block - Usvd.dot(Vsvd)) ** 2))
            assert svd_err < tol

        U, V = A[i]
        U2, V2 = A2[i]

        toy_vs_real_aca = np.sqrt(np.sum((U2.dot(V2) - U.dot(V)) ** 2))
        assert toy_vs_real_aca < tol

        block_err = np.sqrt(np.sum((exact_block - U.dot(V)) ** 2))
        assert block_err < tol


@pytest.mark.parametrize("K_type", kernel_types)
# @profile
def test_hmatrix(K_type):
    K = K_type()
    src = unit_circle(gauss_rule(12), max_curvature=0.1)
    print(src.n_pts, src.n_panels)
    density = np.stack((np.cos(src.pts[:, 0]), np.sin(src.pts[:, 0])), axis=1)[
        :, : K.src_dim
    ]

    tol = 1e-6
    direct_op = integrate_term(K_type(), src.pts, src, tol=tol)
    direct_v = tensor_dot(direct_op, density)
    hmat_op = integrate_term(K_type(), src.pts, src, tol=tol, farfield="hmatrix")
    hmat_v = tensor_dot(hmat_op, density)

    np.testing.assert_allclose(direct_v, hmat_v, rtol=tol, atol=tol)


if __name__ == "__main__":
    from tectosaur2.laplace2d import Hypersingular

    test_hmatrix(Hypersingular)
