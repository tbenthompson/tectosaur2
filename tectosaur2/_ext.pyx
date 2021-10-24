#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport fabs, pi
from libcpp cimport bool
from libcpp.pair cimport pair


cdef extern from "<complex.h>" namespace "std" nogil:
    double real(double complex z)
    double complex log(double complex z)

cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef pair[int, bool] single_obs(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts,
    double[:,::1] src_pts, double[:,::1] src_normals,
    double[::1] src_jacobians, double[::1] src_quad_wts,
    int src_panel_order,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol,
    long[:] panels, int obs_pt_idx, int exp_idx
) nogil:
    if len(panels) == 0:
        return pair[int, bool](-1, False)

    cdef double complex z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    cdef double complex z0 = exp_centers[exp_idx,0] + (exp_centers[exp_idx, 1] * I)
    cdef double r = exp_rs[exp_idx]
    cdef double invr = 1.0/r
    cdef double complex zz0_div_r = (z - z0) * invr

    cdef int n_panels
    cdef int n_srcs
    cdef double complex[:] r_inv_wz0
    cdef double complex[:] exp_t
    cdef double[:] qbx_terms

    with gil:
        n_panels = panels.shape[0]
        n_srcs = n_panels * src_panel_order
        r_inv_wz0 = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        exp_t = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        qbx_terms = np.zeros(n_panels * src_panel_order)


    cdef double a0, a1, constant
    cdef double complex w, nw, inv_wz0, exp_m0, exp_m1

    cdef int pt_start, pt_end, pt_idx, panel_idx, src_pt_idx, j, m

    cdef double eval_m0 = (0.0 if eval_deriv else 1.0)
    cdef double complex eval_m1 = (invr if eval_deriv else zz0_div_r)
    cdef double complex eval_t = eval_m1
    cdef double complex eval_tm

    cdef double mag_a0 = 0.0
    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = panel_idx * src_panel_order + pt_idx
            w = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
            nw = src_normals[src_pt_idx,0] + src_normals[src_pt_idx,1] * I
            inv_wz0 = 1.0 / (w - z0)

            r_inv_wz0[j] = r * inv_wz0
            constant = C * src_quad_wts[src_pt_idx] * src_jacobians[src_pt_idx]
            if exp_deriv:
                exp_m0 = nw * inv_wz0 * constant
                exp_m1 = exp_m0 * r_inv_wz0[j]
            else:
                exp_m0 = log(w-z0) * constant
                exp_m1 = constant * r_inv_wz0[j]
            exp_t[j] = exp_m1

            a1 = real(exp_m1 * eval_m1)
            if eval_deriv:
                # eval_m0 is zero
                mag_a0 += fabs(a1)
                qbx_terms[j] = a1
            else:
                a0 = real(exp_m0 * eval_m0)
                mag_a0 += fabs(a0)
                qbx_terms[j] = a0 + a1

    cdef double am, am_sum
    cdef double am_sum_prev = mag_a0
    cdef int divergences = 0
    for m in range(2, max_p+1):
        eval_t *= zz0_div_r
        eval_tm = eval_t
        if (<int>exp_deriv) + (<int>eval_deriv) == 0:
            eval_tm /= m
        elif (<int>exp_deriv) + (<int>eval_deriv) == 2:
            eval_tm *= m

        am_sum = 0
        for j in range(n_srcs):
            exp_t[j] *= r_inv_wz0[j]
            am = real(exp_t[j] * eval_tm)
            am_sum += am
            qbx_terms[j] += am

        # We use the sum of the last two terms to avoid issues with
        # common sequences where every terms alternate in magnitude
        # See Klinteberg and Tornberg 2018 at the end of page 5
        if (fabs(am_sum) + fabs(am_sum_prev)) < 2 * tol * mag_a0:
            divergences = 0
            break

        if (fabs(am_sum) > fabs(am_sum_prev)):
            divergences += 1
        else:
            divergences = 0
        am_sum_prev = am_sum

    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = panel_idx * src_panel_order + pt_idx
            qbx_mat[obs_pt_idx, 0, src_pt_idx] += qbx_terms[j]

    return pair[int, bool](m, divergences > 1)

def local_qbx_integrals(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] qbx_mat, double[:,::1] obs_pts, src,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol,
    qbx_panels
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_quad_wts = src.quad_wts
    cdef int src_panel_order = src.panel_order

    cdef int i
    cdef np.ndarray p = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef np.ndarray kappa_too_small = np.empty(obs_pts.shape[0], dtype=np.bool_)

    cdef pair[int,bool] result
    for i in range(obs_pts.shape[0]):
        panel_set = qbx_panels[i]
        result = single_obs(
            exp_deriv, eval_deriv,
            qbx_mat, obs_pts,
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, max_p, tol, panel_set, i, i
        )
        p[i], kappa_too_small[i] = result
    return p, kappa_too_small

cdef extern from "nearfield.cpp":
    cdef struct NearfieldArgs:
        double* mat
        int n_obs
        int n_src
        double* obs_pts
        double* src_pts
        double* src_normals
        double* src_quad_wt_jac
        int src_panel_order
        long* panels
        long* panel_starts
        long* refinement_map
        double mult

    cdef void nearfield_single_layer(const NearfieldArgs&)
    cdef void nearfield_double_layer(const NearfieldArgs&)
    cdef void nearfield_adjoint_double_layer(const NearfieldArgs&)
    cdef void nearfield_hypersingular(const NearfieldArgs&)

def nearfield_integrals(
    kernel_name, double[:,:,::1] mat, double[:,::1] obs_pts, src,
    long[::1] panels, long[::1] panel_starts, long[::1] refinement_map,
    double mult
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_quad_wt_jac = src.quad_wt_jac
    cdef int src_panel_order = src.panel_order

    # kernel = double_layer
    cdef NearfieldArgs args = NearfieldArgs(
        &mat[0,0,0], obs_pts.shape[0], src.n_pts, &obs_pts[0,0],
        &src_pts[0,0], &src_normals[0,0], &src_quad_wt_jac[0],
        src_panel_order, &panels[0], &panel_starts[0], &refinement_map[0],
        mult
    )

    if kernel_name == "single_layer":
        kernel_fnc = nearfield_single_layer
    elif kernel_name == "double_layer":
        kernel_fnc = nearfield_double_layer
    elif kernel_name == "adjoint_double_layer":
        kernel_fnc = nearfield_adjoint_double_layer
    elif kernel_name == "hypersingular":
        kernel_fnc = nearfield_hypersingular
    else:
        raise Exception("Unknown kernel name.")
    kernel_fnc(args)
