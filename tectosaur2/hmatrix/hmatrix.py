import numpy as np

from .aca import aca
from .toy_aca import SVD_recompress
from .tree import TempSurface, build_temp_surface, build_tree, traverse


class HMatrix:
    def __init__(self, kernel, obs_pts, src, tol, remove_randomness=False):
        """
        It's important that the HMatrix represent the entire operator under
        consideration if we're going to do an approximate LU
        decompositon/inversion. Otherwise, we would be only inverting part of
        the matrix and that would produce bogus results.

        However, this means that we can't do a precorrection where we ignore the
        nearfield/qbx integrals and compute only point-to-point interactions. On
        the other hand, doing a precorrection would be okay if we only want to
        do matrix-vector products.

        It would be unpleasant to need to mix the nearfield matrices into the
        ACA implementation so I will impose the constraint that the nearfield
        matrix entries must be part of the direct Hmatrix blocks. Then, the
        nearfield entries will only be necessary for computing direct blocks.

        1. construct the direct blocks from the hmatrix pt-pt algorithm
        2. add nearfield matrix entries to the direct blocks. these are added
           because the nearfield matrix entries have been precorrected. this nicely
           decouples the two computation.
        3. any nearfield matrix entries that lie in approx block will lead to
           that block being converted into a direct block and a warning raised
           that a larger cluster separation parameter should be used next time.
        """
        self.tol = tol

        M = 30
        self.obs_tree = build_tree(src.pts, np.zeros(src.n_pts), min_pts_per_box=M)
        self.src_tree = build_tree(
            obs_pts, np.zeros(obs_pts.shape[0]), min_pts_per_box=M
        )

        self.direct_pairs, self.approx_pairs = traverse(
            self.obs_tree.root, self.src_tree.root
        )
        self.tree_surf = TempSurface(
            self.src.pts[self.src_tree.ordered_idxs],
            self.src.normals[self.src_tree.ordered_idxs],
            self.src.quad_wts[self.src_tree.ordered_idxs],
            self.src.jacobians[self.src_tree.ordered_idxs],
        )

        self.D = direct_blocks(
            kernel, self.obs_tree.pts, self.tree_surf, self.direct_pairs
        )
        self.A = aca_blocks(
            kernel, self.obs_tree.pts, self.tree_surf, self.approx_pairs, self.tol
        )


def svd_blocks(kernel, tree_obs, tree_src, node_pairs, tol):
    blocks = direct_blocks(kernel, tree_obs, tree_src, node_pairs)
    out = []
    for b in blocks:
        U, S, V = np.linalg.svd(b)
        # Reverse the list of singular values and sum them to compute the
        # error from each level of truncation.
        frob_K = np.sqrt(np.cumsum(S[::-1] ** 2))[::-1]

        appx_rank = np.argmax(frob_K < tol)

        Uappx = U[:, :appx_rank]
        Vappx = S[:appx_rank, None] * V[:appx_rank]
        out.append((Uappx, Vappx))
    return out


def direct_blocks(kernel, tree_obs, tree_src, node_pairs):
    out = []
    for obs_node, src_node in node_pairs:
        temp_src = build_temp_surface(tree_src, src_node.idx_start, src_node.idx_end)
        obs_pts = tree_obs[obs_node.idx_start : obs_node.idx_end]
        M = kernel.direct(obs_pts, temp_src)
        out.append(
            M.reshape(
                (
                    obs_pts.shape[0] * kernel.obs_dim,
                    (src_node.idx_end - src_node.idx_start) * kernel.src_dim,
                )
            )
        )
    return out


def aca_blocks(kernel, tree_obs, tree_surf, node_pairs, tol, remove_randomness=False):
    """
    The efficient C++ implementation of the ACA algorithm. See the
    toy_aca_blocks function for the Python/numpy version.
    """
    approx_obs_starts = np.array([b[0].idx_start for b in node_pairs])
    approx_obs_ends = np.array([b[0].idx_end for b in node_pairs])
    approx_src_starts = np.array([b[1].idx_start for b in node_pairs])
    approx_src_ends = np.array([b[1].idx_end for b in node_pairs])
    Iref0 = None
    Jref0 = None
    if remove_randomness:
        Iref0 = np.zeros(len(node_pairs), dtype=np.int32)
        Jref0 = np.zeros(len(node_pairs), dtype=np.int32)
    return [
        SVD_recompress(*uv, tol)
        for uv in aca(
            kernel,
            approx_obs_starts,
            approx_obs_ends,
            approx_src_starts,
            approx_src_ends,
            tree_obs,
            tree_surf,
            np.full(len(node_pairs), tol),
            np.full(len(node_pairs), 200, dtype=np.int32),
            Iref0=Iref0,
            Jref0=Jref0,
            verbose=False,
        )
    ]


def toy_aca_blocks(
    kernel, tree_obs, tree_src, node_pairs, tol, remove_randomness=False
):
    out = []
    for obs_node, src_node in node_pairs:
        from .toy_aca import ACA_plus

        Iref0 = None
        Jref0 = None
        if remove_randomness:
            Iref0 = 0
            Jref0 = 0

        s = src_node.idx_start
        e = src_node.idx_end
        node_surf = TempSurface(
            tree_src.pts[s:e],
            tree_src.normals[s:e],
            tree_src.quad_wts[s:e],
            tree_src.jacobians[s:e],
        )
        node_obs_pts = tree_obs[obs_node.idx_start : obs_node.idx_end]

        def calc_rows(Istart, Iend):
            obs_idx_start = Istart // kernel.obs_dim
            obs_idx_end = (Iend - 1) // kernel.obs_dim + 1
            rows = kernel.direct(node_obs_pts[obs_idx_start:obs_idx_end], node_surf)
            # Reshape the returned array and filter out the extra rows.
            n_rows_computed = kernel.obs_dim * (obs_idx_end - obs_idx_start)
            rows2d = rows.reshape((n_rows_computed, -1))
            local_start = Istart % kernel.obs_dim
            local_end = local_start + Iend - Istart
            filter_out_extra = rows2d[local_start:local_end, :]
            return filter_out_extra

        def calc_cols(Jstart, Jend):
            # temp_src = build_temp_surface(node_surf, s, e)
            # return kernel.direct(node_obs_pts, temp_src)[:, :, :, 0].reshape((-1, e - s))
            src_idx_start = Jstart // kernel.src_dim
            src_idx_end = (Jend - 1) // kernel.src_dim + 1
            cols = kernel.direct(
                node_obs_pts, build_temp_surface(node_surf, src_idx_start, src_idx_end)
            )
            # Reshape the returned array and filter out the extra rows.
            n_cols_computed = kernel.src_dim * (src_idx_end - src_idx_start)
            cols2d = cols.reshape((-1, n_cols_computed))
            local_start = Jstart % kernel.src_dim
            local_end = local_start + Jend - Jstart
            filter_out_extra = cols2d[:, local_start:local_end]
            return filter_out_extra

        U_ACA, V_ACA = ACA_plus(
            (obs_node.idx_end - obs_node.idx_start) * kernel.obs_dim,
            (e - s) * kernel.src_dim,
            calc_rows,
            calc_cols,
            tol,
            row_dim=kernel.obs_dim,
            col_dim=kernel.src_dim,
            Iref0=Iref0,
            Jref0=Jref0,
        )  # , verbose=True)
        U_SVD, V_SVD = SVD_recompress(U_ACA, V_ACA, tol)
        out.append((U_SVD, V_SVD))
    return out
