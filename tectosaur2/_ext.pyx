#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "local_qbx.cpp":
    cdef struct LocalQBXArgs:
        double* mat
        int* p
        int* failed;
        int* n_subsets
        int n_obs
        int n_src
        double* obs_pts
        double* src_pts
        double* src_normals
        double* src_jacobians
        double* src_panel_lengths
        double* src_param_width
        int n_src_panels
        double* qx
        double* qw
        double* interp_wts
        int nq
        double* exp_centers
        double* exp_rs
        int max_p
        double tol
        long* panels
        long* panel_starts

    cdef void local_qbx_single_layer(const LocalQBXArgs&)
    cdef void local_qbx_double_layer(const LocalQBXArgs&)
    cdef void local_qbx_adjoint_double_layer(const LocalQBXArgs&)
    cdef void local_qbx_hypersingular(const LocalQBXArgs&)

def local_qbx_integrals(
    kernel_name,
    double[:,:,::1] mat, double[:,::1] obs_pts, src,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol, long[:] panels, long[:] panel_starts
):
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_panel_lengths = src.panel_length
    cdef double[::1] src_param_width = src.panel_parameter_width
    cdef double[::1] qx = src.qx
    cdef double[::1] qw = src.qw
    cdef double[::1] interp_wts = src.interp_wts

    p_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] p = p_np

    n_subsets_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] n_subsets = n_subsets_np

    failed_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] failed = failed_np

    cdef LocalQBXArgs args = LocalQBXArgs(
        &mat[0,0,0], &p[0], &failed[0], &n_subsets[0], obs_pts.shape[0], src.n_pts,
        &obs_pts[0,0], &src_pts[0,0], &src_normals[0,0], &src_jacobians[0],
        &src_panel_lengths[0], &src_param_width[0], src.n_panels, &qx[0],
        &qw[0], &interp_wts[0], qx.shape[0], &exp_centers[0,0], &exp_rs[0],
        max_p, tol, &panels[0], &panel_starts[0]
    )

    if kernel_name == "single_layer":
        local_qbx_single_layer(args)
    elif kernel_name == "double_layer":
        local_qbx_double_layer(args)
    elif kernel_name == "adjoint_double_layer":
        local_qbx_adjoint_double_layer(args)
    elif kernel_name == "hypersingular":
        local_qbx_hypersingular(args)
    else:
        raise Exception("Unknown kernel name.")

    return p_np, failed_np, n_subsets_np


cdef extern from "nearfield.cpp":
    cdef struct NearfieldArgs:
        double* mat
        int n_obs
        int n_src
        double* obs_pts
        double* src_pts
        double* src_normals
        double* src_jacobians
        double* src_panel_lengths
        double* src_param_width
        int src_n_panels
        double* qx
        double* qw
        double* interp_wts
        int nq
        long* panel_obs_pts
        long* panel_obs_pts_starts
        double mult
        double tol
        bool adaptive

    cdef void nearfield_single_layer(const NearfieldArgs&)
    cdef void nearfield_double_layer(const NearfieldArgs&)
    cdef void nearfield_adjoint_double_layer(const NearfieldArgs&)
    cdef void nearfield_hypersingular(const NearfieldArgs&)

def nearfield_integrals(
    kernel_name, double[:,:,::1] mat, double[:,::1] obs_pts, src,
    long[::1] panel_obs_pts, long[::1] panel_obs_pts_starts,
    double mult, double tol, bool adaptive
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_panel_lengths = src.panel_length
    cdef double[::1] src_param_width = src.panel_parameter_width
    cdef double[::1] qx = src.qx
    cdef double[::1] qw = src.qw
    cdef double[::1] interp_wts = src.interp_wts


    cdef NearfieldArgs args = NearfieldArgs(
        &mat[0,0,0], obs_pts.shape[0], src.n_pts, &obs_pts[0,0],
        &src_pts[0,0], &src_normals[0,0], &src_jacobians[0],
        &src_panel_lengths[0], &src_param_width[0], src.n_panels,
        &qx[0], &qw[0], &interp_wts[0], qx.shape[0],
        &panel_obs_pts[0], &panel_obs_pts_starts[0],
        mult, tol, adaptive
    )

    if kernel_name == "single_layer":
        nearfield_single_layer(args)
    elif kernel_name == "double_layer":
        nearfield_double_layer(args)
    elif kernel_name == "adjoint_double_layer":
        nearfield_adjoint_double_layer(args)
    elif kernel_name == "hypersingular":
        nearfield_hypersingular(args)
    else:
        raise Exception("Unknown kernel name.")


def identify_nearfield_panels(obs_pts, src_pts, int n_src_panels, int source_order):
    cdef int n_obs = obs_pts.shape[0]

    cdef long[:] src_pts_starts = np.zeros(n_obs + 1, dtype=int)
    cdef long sum = 0
    cdef int i, j
    for i in range(n_obs):
        sum += len(src_pts[i])
        src_pts_starts[1 + i] = sum
    cdef long[:] all_src_pts = np.concatenate(src_pts, dtype=int, casting='unsafe')

    cdef long start, end

    cdef vector[vector[int]] panel_obs_pts_vecs
    cdef vector[int] empty
    for i in range(n_src_panels):
        panel_obs_pts_vecs.push_back(empty)

    cdef vector[int] panels_vector
    panel_starts_np = np.empty(n_obs + 1, dtype=int)
    cdef long[:] panel_starts = panel_starts_np
    panel_starts[0] = 0

    cdef int panel, last_panel
    cdef int n_panels
    with nogil:
        for i in range(n_obs):
            start = src_pts_starts[i]
            end = src_pts_starts[i + 1]

            n_panels = 0
            last_panel = -1
            for j in range(start, end):
                panel = all_src_pts[j] // source_order
                if panel == last_panel:
                    continue
                panels_vector.push_back(panel)
                panel_obs_pts_vecs[panel].push_back(i)
                n_panels += 1
                last_panel = panel
            panel_starts[i + 1] = panel_starts[i] + n_panels

    panels_np = np.empty(panel_starts[n_obs], dtype=int)
    cdef long[:] panels = panels_np
    for i in range(panel_starts[n_obs]):
        panels[i] = panels_vector[i]

    panel_obs_pts_np = np.empty(panel_starts[n_obs], dtype=int)
    panel_obs_pt_starts_np = np.empty(n_src_panels + 1, dtype=int)
    cdef long[:] panel_obs_pts = panel_obs_pts_np
    cdef long[:] panel_obs_pt_starts = panel_obs_pt_starts_np
    panel_obs_pt_starts[0] = 0
    for i in range(n_src_panels):
        panel_obs_pt_starts[i + 1] = panel_obs_pt_starts[i]
        for j in range(panel_obs_pts_vecs[i].size()):
            panel_obs_pts[panel_obs_pt_starts[i + 1]] = panel_obs_pts_vecs[i][j]
            panel_obs_pt_starts[i + 1] += 1

    return panels_np, panel_starts_np, panel_obs_pts_np, panel_obs_pt_starts_np
