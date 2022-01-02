from math import ceil

import numpy as np

from .aca_ext import aca_integrals


def aca(
    kernel,
    obs_start,
    obs_end,
    src_start,
    src_end,
    obs_pts,
    src,
    tol,
    max_iter,
    Iref0=None,
    Jref0=None,
    verbose=False,
):
    default_chunk_size = 512
    n_blocks = obs_end.shape[0]
    float_type = np.float64

    n_chunks = int(ceil(n_blocks / default_chunk_size))
    appxs = []
    for i in range(n_chunks):
        chunk_start = i * default_chunk_size
        chunk_size = min(n_blocks - chunk_start, default_chunk_size)
        chunk_end = chunk_start + chunk_size

        n_obs_per_block = (
            obs_end[chunk_start:chunk_end] - obs_start[chunk_start:chunk_end]
        )
        n_src_per_block = (
            src_end[chunk_start:chunk_end] - src_start[chunk_start:chunk_end]
        )
        n_rows = n_obs_per_block * kernel.obs_dim
        n_cols = n_src_per_block * kernel.src_dim
        block_sizes = n_rows * n_cols
        # Storage for the U, V output matrices. These will be in a packed format.
        # We allocate the maximum possible necessary amount of memory. Since the
        # outermost loop over chunks means that we're dealing with at most 512
        # blocks here, then this won't be too much memory. The (mostly unused)
        # memory can be freed and recycled for the next chunk loop iteration.
        buffer = np.empty(block_sizes.sum(), float_type)

        # Storage for temporary rows and columns: RIref, RJref, RIstar, RJstar
        fworkspace_per_block = (
            n_cols + n_rows + kernel.src_dim * n_cols + kernel.obs_dim * n_rows
        ).astype(np.int32)
        fworkspace_ends = np.cumsum(fworkspace_per_block, dtype=np.int32)
        fworkspace_starts = fworkspace_ends - fworkspace_per_block
        fworkspace = np.empty(fworkspace_ends[-1], float_type)

        # uv_ptrs forms arrays that point to the start of each U/V vector pairs in
        # the main output buffer
        uv_ptrs_size = np.minimum(n_rows, n_cols, dtype=np.int32)
        uv_ptrs_ends = np.cumsum(uv_ptrs_size, dtype=np.int32)
        uv_ptrs_starts = uv_ptrs_ends - uv_ptrs_size
        uv_ptrs = np.empty(uv_ptrs_ends[-1], np.int32)
        iworkspace = np.empty(uv_ptrs_ends[-1], np.int32)

        # Output space for specifying the number of terms used for each
        # approximation.
        n_terms = np.empty(chunk_size, np.int32)

        # Storage space for a pointer to the next empty portion of the output
        # buffer.
        next_ptr = np.zeros(1, np.int32)

        # The index of the starting reference rows/cols.
        if Iref0 is None:
            chunk_Iref0 = np.random.randint(0, n_rows, size=chunk_size, dtype=np.int32)
        else:
            chunk_Iref0 = Iref0[chunk_start:chunk_end]
        if Jref0 is None:
            chunk_Jref0 = np.random.randint(0, n_cols, size=chunk_size, dtype=np.int32)
        else:
            chunk_Jref0 = Jref0[chunk_start:chunk_end]

        chunk_obs_start = obs_start[chunk_start:chunk_end]
        chunk_obs_end = obs_end[chunk_start:chunk_end]
        chunk_src_start = src_start[chunk_start:chunk_end]
        chunk_src_end = src_end[chunk_start:chunk_end]
        chunk_tol = tol[chunk_start:chunk_end]
        chunk_max_iter = max_iter[chunk_start:chunk_end]

        if verbose:
            print(f"buffer.shape = {buffer.shape}")
            print(f"uv_ptrs.shape = {uv_ptrs.shape}")
            print(f"n_terms.shape = {n_terms.shape}")
            print(f"next_ptr.shape = {next_ptr.shape}")
            print(f"fworkspace.shape = {fworkspace.shape}")
            print(f"iworkspace.shape = {iworkspace.shape}")
            print(f"uv_ptrs_starts.shape = {uv_ptrs_starts.shape}")
            print(f"Iref0.shape = {chunk_Iref0.shape}")
            print(f"Jref0.shape = {chunk_Jref0.shape}")
            print(f"obs_pts.shape = {obs_pts.shape}")
            print(f"obs_start.shape = {chunk_obs_start.shape}")
            print(f"obs_end.shape = {chunk_obs_end.shape}")
            print(f"src_start.shape = {chunk_src_start.shape}")
            print(f"src_end.shape = {chunk_src_end.shape}")

        aca_integrals(
            kernel,
            buffer,
            uv_ptrs,
            n_terms,
            next_ptr,
            fworkspace,
            iworkspace,
            chunk_size,
            chunk_obs_start,
            chunk_obs_end,
            chunk_src_start,
            chunk_src_end,
            uv_ptrs_starts,
            fworkspace_starts,
            chunk_Iref0,
            chunk_Jref0,
            obs_pts,
            src,
            chunk_tol,
            chunk_max_iter,
            kernel.parameters,
            verbose,
        )

        # post-process the buffer to collect the U, V vectors
        for i in range(chunk_size):
            us = []
            vs = []
            uv_ptr0 = uv_ptrs_starts[i]
            ptrs = uv_ptrs[uv_ptr0 + np.arange(n_terms[i])]
            us = buffer[ptrs[:, None] + np.arange(n_rows[i])[None, :]]
            vs = buffer[
                ptrs[:, None] + np.arange(n_rows[i], n_rows[i] + n_cols[i])[None, :]
            ]
            appxs.append((us.T, vs))
    return appxs
