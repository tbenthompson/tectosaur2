import numpy as np

from .aca import aca
from .toy_aca import SVD_recompress
from .tree import TempSurface, build_temp_surface, build_tree, traverse


class HMatrix:
    def __init__(self, kernel, obs_pts, src, tol, nearfield_mat, deterministic=False):
        """
        One of the primary goals with this HMatrix implementation is to support
        H-matrix inversion or LU decomposition. This puts some important
        constraints on the implementation:

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
        self.obs_dim = kernel.obs_dim
        self.src_dim = kernel.src_dim
        self.tol = tol

        M = 30
        self.obs_tree = build_tree(src.pts, np.zeros(src.n_pts), min_pts_per_box=M)
        obs_tree_pts = obs_pts[self.obs_tree.ordered_idxs]
        self.src_tree = build_tree(
            obs_pts, np.zeros(obs_pts.shape[0]), min_pts_per_box=M
        )
        self.direct_pairs, self.approx_pairs = traverse(
            self.obs_tree.root, self.src_tree.root
        )
        self.tree_surf = TempSurface(
            src.pts[self.src_tree.ordered_idxs],
            src.normals[self.src_tree.ordered_idxs],
            src.quad_wts[self.src_tree.ordered_idxs],
            src.jacobians[self.src_tree.ordered_idxs],
        )

        block_tolerances = self.calc_block_tolerance(kernel, obs_pts, src)
        # block_tolerances2 = np.full(len(self.approx_pairs), tol)

        self.D = direct_blocks(
            kernel,
            obs_tree_pts,
            self.tree_surf,
            self.direct_pairs,
        )
        self.A = aca_blocks(
            kernel,
            obs_tree_pts,
            self.tree_surf,
            self.approx_pairs,
            block_tolerances,
            deterministic=deterministic,
        )
        self.shape = (obs_pts.shape[0], kernel.obs_dim, src.n_pts, kernel.src_dim)
        self.nearfield_mat = nearfield_mat

    def calc_block_tolerance(self, kernel, obs_pts, src):
        """
        If we use a fixed tolerance for each block, then the sum of the errors
        from each block can accumulate to much greater than the allowed
        tolerance. To resolve this problem, we scale the tolerance by the size
        of the block multiplied by the frobenius norm of the full matrix.
        However, the frobenius norm of the full matrix is not accessible without
        computing the full matrix. To work around this problem, we take a random
        sample of rows and then compute a jack-knife estimator of the full matrix
        frobenius norm. We use an underestimate by two standard deviations in order
        to return conservative error.

        This approach is explained in the Bradley 2014 paper.
        """
        n_samples = 20
        row_idxs = np.random.randint(0, obs_pts.shape[0], size=n_samples)
        sample = kernel.direct(obs_pts[row_idxs], src).flatten()
        n_full_entries = obs_pts.shape[0] * kernel.obs_dim * src.n_pts * kernel.src_dim
        frob2_ests = [
            np.sum(sample[i] ** 2) * (n_full_entries / sample[i].size)
            for i in range(n_samples)
        ]
        frob2_mean = np.mean(frob2_ests)

        jack_replicates = []
        for i in range(n_samples):
            jack_replicates.append(
                (n_samples * frob2_mean - frob2_ests[i]) / (n_samples - 1)
            )
        jack_mean = np.mean(jack_replicates)
        jack_stddev = np.sqrt(
            np.sum((jack_replicates - jack_mean) ** 2) * ((n_samples - 1) / n_samples)
        )

        print("|B|_F^2 =", frob2_mean)
        print("standard deviation", jack_stddev)
        safe_frob2_est = frob2_mean - jack_stddev * 2
        print(safe_frob2_est)
        n_entries_per_block = np.array(
            [
                (obs_node.idx_end - obs_node.idx_start)
                * kernel.obs_dim
                * (src_node.idx_end - src_node.idx_start)
                * kernel.src_dim
                for obs_node, src_node in self.approx_pairs
            ]
        )
        safe_frob_est = np.sqrt(safe_frob2_est)
        block_tolerances = (
            self.tol * np.sqrt(n_entries_per_block / n_full_entries) * safe_frob_est
        )
        return block_tolerances

    def tensor_dot(self, x):
        ytree = np.zeros(self.shape[:2])
        treex = x[self.src_tree.ordered_idxs]

        for i, (obs_node, src_node) in enumerate(self.direct_pairs):
            xchunk = treex[src_node.idx_start : src_node.idx_end]
            ytree[obs_node.idx_start : obs_node.idx_end] += (
                self.D[i].dot(xchunk.ravel()).reshape((-1, ytree.shape[1]))
            )

        for i, (obs_node, src_node) in enumerate(self.approx_pairs):
            xchunk = treex[src_node.idx_start : src_node.idx_end]
            U, V = self.A[i]
            ytree[obs_node.idx_start : obs_node.idx_end] += U.dot(
                V.dot(xchunk.ravel())
            ).reshape((-1, ytree.shape[1]))

        yh = np.zeros(self.shape[:2])
        yh[self.obs_tree.ordered_idxs] = ytree.reshape((-1, self.obs_dim))

        yh += np.tensordot(self.nearfield_mat, x, axes=2)
        return yh


def svd_blocks(kernel, tree_obs, tree_src, node_pairs, tol):
    blocks = direct_blocks(kernel, tree_obs, tree_src, node_pairs)
    out = []
    for i, b in enumerate(blocks):
        U, S, V = np.linalg.svd(b)
        # Reverse the list of singular values and sum them to compute the
        # error from each level of truncation.
        frob_K = np.sqrt(np.cumsum(S[::-1] ** 2))[::-1]

        appx_rank = np.argmax(frob_K < tol[i])

        Uappx = U[:, :appx_rank]
        Vappx = S[:appx_rank, None] * V[:appx_rank]
        out.append((Uappx, Vappx))
    return out


def direct_blocks(kernel, tree_obs, tree_src, node_pairs):
    out = []
    for obs_node, src_node in node_pairs:
        temp_src = build_temp_surface(tree_src, src_node.idx_start, src_node.idx_end)
        obs_pts = tree_obs[obs_node.idx_start : obs_node.idx_end]
        n_obs = (obs_node.idx_end - obs_node.idx_start) * kernel.obs_dim
        out.append(kernel.direct(obs_pts, temp_src).reshape((n_obs, -1)))
    return out


def aca_blocks(kernel, tree_obs, tree_surf, node_pairs, in_tol, deterministic=False):
    """
    The efficient C++ implementation of the ACA algorithm. See the
    toy_aca_blocks function for the Python/numpy version.
    """
    # Because of interactions between the ACA tolerance the SVD recompression
    # tolerance, the algorithmic tolerance is required to be half of the input
    # tolerance in order to achieve the required input tolerance. Essentially
    # both steps add a maximum of `tol` to the error. In the worst case, this
    # can be `2*tol`.
    tol = in_tol * 0.5
    approx_obs_starts = np.array([b[0].idx_start for b in node_pairs])
    approx_obs_ends = np.array([b[0].idx_end for b in node_pairs])
    approx_src_starts = np.array([b[1].idx_start for b in node_pairs])
    approx_src_ends = np.array([b[1].idx_end for b in node_pairs])
    Iref0 = None
    Jref0 = None
    if deterministic:
        Iref0 = np.zeros(len(node_pairs), dtype=np.int32)
        Jref0 = np.zeros(len(node_pairs), dtype=np.int32)
    return [
        SVD_recompress(*uv, tol[i])
        for i, uv in enumerate(
            aca(
                kernel,
                approx_obs_starts,
                approx_obs_ends,
                approx_src_starts,
                approx_src_ends,
                tree_obs,
                tree_surf,
                tol,
                np.full(len(node_pairs), 200, dtype=np.int32),
                Iref0=Iref0,
                Jref0=Jref0,
                verbose=False,
            )
        )
    ]


def toy_aca_blocks(kernel, tree_obs, tree_src, node_pairs, in_tol, deterministic=False):
    # Because of interactions between the ACA tolerance the SVD recompression
    # tolerance, the algorithmic tolerance is required to be half of the input
    # tolerance in order to achieve the required input tolerance. Essentially
    # both steps add a maximum of `tol` to the error. In the worst case, this
    # can be `2*tol`.
    tol = in_tol * 0.5
    out = []
    for obs_node, src_node in node_pairs:
        from .toy_aca import ACA_plus

        Iref0 = None
        Jref0 = None
        if deterministic:
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
