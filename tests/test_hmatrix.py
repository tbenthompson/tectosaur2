import numpy as np
import sympy as sp

from tectosaur2 import gauss_rule, integrate_term, panelize_symbolic_surface
from tectosaur2.hmatrix.hmatrix import (
    SVD_recompress,
    TempSurface,
    aca,
    aca_blocks,
    build_temp_surface,
    build_tree,
    direct_blocks,
    svd_blocks,
    toy_aca_blocks,
    traverse,
)
from tectosaur2.laplace2d import (
    adjoint_double_layer,
    double_layer,
    hypersingular,
    single_layer,
)


def test_aca():
    t = sp.var("t")
    surf = panelize_symbolic_surface(t, 0 * t, t, gauss_rule(6), n_panels=100)
    K = hypersingular
    exact_mat = K.direct(surf.pts, surf)[:, :, :, 0].reshape((-1, surf.n_pts))
    obs_tree = build_tree(surf.pts, np.zeros(surf.n_pts), min_pts_per_box=30)
    src_tree = obs_tree

    direct, approx = traverse(obs_tree.root, src_tree.root)
    tree_surf = TempSurface(
        surf.pts[src_tree.ordered_idxs],
        surf.normals[src_tree.ordered_idxs],
        surf.quad_wts[src_tree.ordered_idxs],
        surf.jacobians[src_tree.ordered_idxs],
    )

    tol = 1e-10

    A = toy_aca_blocks(K, tree_surf.pts, tree_surf, approx, tol, remove_randomness=True)
    A2 = aca_blocks(K, tree_surf.pts, tree_surf, approx, tol, remove_randomness=True)
    Asvd = svd_blocks(K, tree_surf.pts, tree_surf, approx[:3], tol)

    for i, (obs_node, src_node) in enumerate(approx):
        U, V = A[i]
        U2, V2 = A2[i]
        exact_block = exact_mat[
            K.obs_dim * obs_node.idx_start : K.obs_dim * obs_node.idx_end,
            K.src_dim * src_node.idx_start : K.src_dim * src_node.idx_end,
        ]

        block_err = np.sqrt(np.sum((exact_block - U.dot(V)) ** 2))
        assert block_err < tol

        toy_vs_real_aca = np.sqrt(np.sum((U2.dot(V2) - U.dot(V)) ** 2))
        assert toy_vs_real_aca < 4e-16

        if i < 3:
            # SVD is expensive so we'll only test the first three approx blocks.
            Usvd, Vsvd = Asvd[i]
            svd_err = np.sqrt(np.sum((exact_block - Usvd.dot(Vsvd)) ** 2))
            assert svd_err < tol


def test_hmatrix():
    pass
