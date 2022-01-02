#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from math import ceil

import numpy as np

cimport numpy as np
from libcpp cimport bool


cdef extern from 'aca_ext_impl.cpp':
    cdef struct ACAArgs:
        double* buffer
        int* uv_ptrs
        int* n_terms

        int* next_buffer_ptr
        double* fworkspace
        int* iworkspace

        int row_dim
        int col_dim

        int n_blocks

        long* obs_start
        long* obs_end
        long* src_start
        long* src_end

        int* uv_ptrs_starts

        int* fworkspace_starts

        int* Iref0
        int* Jref0

        double* obs_pts
        double* src_pts
        double* src_normals
        double* src_weights

        double* tol
        int* max_iter
        double* kernel_parameters
        bool verbose

    cdef void aca_single_layer(const ACAArgs&)
    cdef void aca_double_layer(const ACAArgs&)
    cdef void aca_adjoint_double_layer(const ACAArgs&)
    cdef void aca_hypersingular(const ACAArgs&)
    cdef void aca_elastic_U(const ACAArgs&)
    cdef void aca_elastic_T(const ACAArgs&)
    cdef void aca_elastic_A(const ACAArgs&)
    cdef void aca_elastic_H(const ACAArgs&)

def aca_integrals(kernel, double[::1] buffer, int[::1] uv_ptrs, int[::1] n_terms, int[::1] next_ptr,
        double[::1] fworkspace, int[::1] iworkspace, int n_blocks,
        long[::1] obs_start, long[::1] obs_end, long[::1] src_start, long[::1] src_end,
        int[::1] uv_ptrs_starts, int[::1] fworkspace_starts, int[::1] Iref0, int[::1] Jref0,
        double[:,::1] obs_pts, src,
        double[::1] tol, int[::1] max_iter, double[::1] kernel_parameters, bool verbose):
    """
    kernel_name, kernel_parameters:
        Kernel info!

    obs_start, obs_end, src_start, src_end:
        The index of the obs and src point start and end for each block. All
        blocks are contiguous in obs/src idx because the indices have been
        re-arranged during the tree construction phase.

    obs_pts, src:
        Info on the observation points and source surface.

    tol:
        absolute tolerance for the frobenius norm error of each block.

    max_iter:
        the maximum allowable rank for each block. The maximum allowable value
        is min(block_rows, block_cols) since anything larger would imply more
        than full rank.

    Iref0, Jref0:
        The only randomness in the ACA+ algorithm comes from the initial choice
        of "reference" row and column. By specifying these parameters, that
        randomness can be removed. This can be useful for debugging or for
        producing consistent results.
    """
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    # TODO: weights times jacobians, this should be fixed globally.
    cdef double[::1] src_weights = src.quad_wts * src.jacobians

    cdef ACAArgs args = ACAArgs(&buffer[0], &uv_ptrs[0], &n_terms[0], &next_ptr[0],
            &fworkspace[0], &iworkspace[0], kernel.obs_dim, kernel.src_dim, n_blocks,
            &obs_start[0], &obs_end[0], &src_start[0],
            &src_end[0], &uv_ptrs_starts[0], &fworkspace_starts[0],
            &Iref0[0], &Jref0[0], &obs_pts[0,0], &src_pts[0,0],
            &src_normals[0,0], &src_weights[0], &tol[0], &max_iter[0], &kernel_parameters[0], verbose)

    if kernel.name == "single_layer":
        aca_single_layer(args)
    elif kernel.name == "double_layer":
        aca_double_layer(args)
    elif kernel.name == "adjoint_double_layer":
        aca_adjoint_double_layer(args)
    elif kernel.name == "hypersingular":
        aca_hypersingular(args)
    elif kernel.name == "elastic_U":
        aca_elastic_U(args)
    elif kernel.name == "elastic_T":
        aca_elastic_T(args)
    elif kernel.name == "elastic_A":
        aca_elastic_A(args)
    elif kernel.name == "elastic_H":
        aca_elastic_H(args)
    else:
        raise Exception("Unknown kernel name.")
