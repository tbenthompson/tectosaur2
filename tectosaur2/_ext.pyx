#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "local_qbx.cpp":

    # Look in local_qbx.cpp for details on the meaning of parameters.
    cdef struct LocalQBXArgs:
        double* entries
        long* rows
        long* cols
        int* p
        double* integration_error
        int* n_subsets

        double* test_density
        int n_obs
        int n_src

        int obs_dim
        int src_dim

        double* obs_pts

        double* src_pts
        double* src_normals
        double* src_jacobians
        double* src_param_width
        int n_src_panels

        double* interp_qx;
        double* interp_wts;
        int n_interp;

        double* kronrod_qx;
        double* kronrod_qw;
        double* kronrod_qw_gauss;
        int n_kronrod;

        double* exp_centers
        double* exp_rs

        int max_p
        double tol

        long* panels
        long* panel_starts

        double* kernel_parameters

    cdef void local_qbx_single_layer(const LocalQBXArgs&)
    cdef void local_qbx_double_layer(const LocalQBXArgs&)
    cdef void local_qbx_adjoint_double_layer(const LocalQBXArgs&)
    cdef void local_qbx_hypersingular(const LocalQBXArgs&)
    cdef void local_qbx_elastic_U(const LocalQBXArgs&)
    cdef void local_qbx_elastic_T(const LocalQBXArgs&)
    cdef void local_qbx_elastic_A(const LocalQBXArgs&)
    cdef void local_qbx_elastic_H(const LocalQBXArgs&)

    cdef void cpp_choose_expansion_circles(double*, double*, double*, int,
                                        double*, long*, double*, double*, int,
                                        int, long*, long*, double*,
                                        long*, long*, double, double)


def local_qbx_integrals(
    kernel,
    double[::1] entries,  long[::1] rows, long[::1] cols, double[:,::1] obs_pts, src, double[::1] test_density,
    double[::1] kronrod_qx, double[::1] kronrod_qw, double[::1] kronrod_qw_gauss,
    double[:,::1] exp_centers, double[::1] exp_rs,
    double tol, long[:] panels, long[:] panel_starts
):
    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_param_width = src.panel_parameter_width
    cdef double[::1] interp_qx = src.qx
    cdef double[::1] interp_wts = src.interp_wts

    p_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] p = p_np

    n_subsets_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] n_subsets = n_subsets_np

    integration_error_np = np.empty(obs_pts.shape[0], dtype=np.float64)
    cdef double[::1] integration_error = integration_error_np

    cdef double[::1] kernel_parameters = kernel.parameters

    cdef LocalQBXArgs args = LocalQBXArgs(
        &entries[0], &rows[0], &cols[0], &p[0], &integration_error[0],
        &n_subsets[0], &test_density[0], obs_pts.shape[0], src.n_pts,
        kernel.obs_dim, kernel.src_dim, &obs_pts[0,0], &src_pts[0,0],
        &src_normals[0,0], &src_jacobians[0], &src_param_width[0], src.n_panels,
        &interp_qx[0], &interp_wts[0], interp_qx.shape[0], &kronrod_qx[0],
        &kronrod_qw[0], &kronrod_qw_gauss[0], kronrod_qx.shape[0],
        &exp_centers[0,0], &exp_rs[0], kernel.max_p, tol, &panels[0],
        &panel_starts[0], &kernel_parameters[0]
    )

    if kernel.name == "single_layer":
        local_qbx_single_layer(args)
    elif kernel.name == "double_layer":
        local_qbx_double_layer(args)
    elif kernel.name == "adjoint_double_layer":
        local_qbx_adjoint_double_layer(args)
    elif kernel.name == "hypersingular":
        local_qbx_hypersingular(args)
    elif kernel.name == "elastic_U":
        local_qbx_elastic_U(args)
    elif kernel.name == "elastic_T":
        local_qbx_elastic_T(args)
    elif kernel.name == "elastic_A":
        local_qbx_elastic_A(args)
    elif kernel.name == "elastic_H":
        local_qbx_elastic_H(args)
    else:
        raise Exception("Unknown kernel name.")

    return p_np, integration_error_np, n_subsets_np


cdef extern from "nearfield.cpp":
    cdef struct NearfieldArgs:
        double* entries
        long* rows
        long* cols

        int* n_subsets
        double* integration_error
        int n_obs
        int n_src
        int obs_dim
        int src_dim

        double* obs_pts

        double* src_pts
        double* src_normals
        double* src_jacobians
        double* src_param_width
        int n_src_panels

        double* interp_qx;
        double* interp_wts;
        int n_interp;

        double* kronrod_qx;
        double* kronrod_qw;
        double* kronrod_qw_gauss;
        int n_kronrod;

        double mult
        double tol
        bool adaptive

        long* panel_obs_pts
        long* panel_obs_pts_starts

        double* kernel_parameters

    cdef void nearfield_single_layer(const NearfieldArgs&)
    cdef void nearfield_double_layer(const NearfieldArgs&)
    cdef void nearfield_adjoint_double_layer(const NearfieldArgs&)
    cdef void nearfield_hypersingular(const NearfieldArgs&)
    cdef void nearfield_elastic_U(const NearfieldArgs&)
    cdef void nearfield_elastic_T(const NearfieldArgs&)
    cdef void nearfield_elastic_A(const NearfieldArgs&)
    cdef void nearfield_elastic_H(const NearfieldArgs&)

def nearfield_integrals(
    kernel, double[::1] entries, long[::1] rows, long[::1] cols, double[:,::1]
    obs_pts, src, double[::1] kronrod_qx, double[::1] kronrod_qw, double[::1]
    kronrod_qw_gauss, long[::1] panel_obs_pts, long[::1] panel_obs_pts_starts,
    double mult, double tol, bool adaptive
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_param_width = src.panel_parameter_width
    cdef double[::1] interp_qx = src.qx
    cdef double[::1] interp_wts = src.interp_wts

    n_subsets_np = np.zeros(obs_pts.shape[0], dtype=np.int32)
    cdef int[::1] n_subsets = n_subsets_np

    integration_error_np = np.zeros(obs_pts.shape[0], dtype=np.float64)
    cdef double[::1] integration_error = integration_error_np

    cdef double[::1] kernel_parameters = kernel.parameters

    cdef NearfieldArgs args = NearfieldArgs(
        &entries[0], &rows[0], &cols[0], &n_subsets[0], &integration_error[0], obs_pts.shape[0],
        src.n_pts, kernel.obs_dim, kernel.src_dim, &obs_pts[0,0], &src_pts[0,0],
        &src_normals[0,0], &src_jacobians[0], &src_param_width[0], src.n_panels,
        &interp_qx[0], &interp_wts[0], interp_qx.shape[0], &kronrod_qx[0],
        &kronrod_qw[0], &kronrod_qw_gauss[0], kronrod_qx.shape[0], mult, tol,
        adaptive, &panel_obs_pts[0], &panel_obs_pts_starts[0],
        &kernel_parameters[0]
    )

    if kernel.name == "single_layer":
        nearfield_single_layer(args)
    elif kernel.name == "double_layer":
        nearfield_double_layer(args)
    elif kernel.name == "adjoint_double_layer":
        nearfield_adjoint_double_layer(args)
    elif kernel.name == "hypersingular":
        nearfield_hypersingular(args)
    elif kernel.name == "elastic_U":
        nearfield_elastic_U(args)
    elif kernel.name == "elastic_T":
        nearfield_elastic_T(args)
    elif kernel.name == "elastic_A":
        nearfield_elastic_A(args)
    elif kernel.name == "elastic_H":
        nearfield_elastic_H(args)
    else:
        raise Exception("Unknown kernel name.")

    return n_subsets_np, integration_error_np

def identify_nearfield_panels(int n_obs, src_pts, int n_src_panels, int source_order):
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

def choose_expansion_circles(
    double[:,::1] exp_centers,
    double[:] exp_rs,
    double[:,::1] obs_pts,
    double[:,::1] offset_vector,
    long[::1] owner_panel_idx,
    double[:,::1] src_pts,
    double[:,::1] interp_mat,
    long[::1] panels,
    long[::1] panel_starts,
    double[:,::1] singularities,
    long[::1] nearby_singularities,
    long[::1] nearby_singularity_starts,
    double nearby_safety_ratio,
    double singularity_safety_ratio
):

    cpp_choose_expansion_circles(
        &exp_centers[0,0],
        &exp_rs[0],
        &obs_pts[0,0],
        obs_pts.shape[0],
        &offset_vector[0,0],
        &owner_panel_idx[0],
        &src_pts[0,0],
        &interp_mat[0,0],
        interp_mat.shape[0],
        interp_mat.shape[1],
        &panels[0],
        &panel_starts[0],
        &singularities[0,0],
        &nearby_singularities[0],
        &nearby_singularity_starts[0],
        nearby_safety_ratio,
        singularity_safety_ratio
    )
