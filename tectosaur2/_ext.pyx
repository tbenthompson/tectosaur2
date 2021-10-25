#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport fabs, pi
from libcpp cimport bool
from libcpp.pair cimport pair


cdef extern from "<complex.h>" namespace "std" nogil:
    double norm(double complex z)
    double real(double complex z)
    double imag(double complex z)
    double complex log(double complex z)

cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef pair[int, bool] single_obs(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] mat, double[:,::1] obs_pts,
    double[:,::1] src_pts, double[:,::1] src_normals,
    double[::1] src_jacobians, double[::1] src_quad_wts,
    int src_panel_order,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol,
    long[:] panels, long[:] panel_starts, int obs_pt_idx, int exp_idx
) nogil:
    cdef int panel_start = panel_starts[obs_pt_idx]
    cdef int panel_end = panel_starts[obs_pt_idx + 1]
    cdef int n_panels = panel_end - panel_start
    if n_panels == 0:
        return pair[int, bool](-1, False)

    cdef double complex z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    cdef double complex z0 = exp_centers[exp_idx,0] + (exp_centers[exp_idx, 1] * I)
    cdef double r = exp_rs[exp_idx]
    cdef double invr = 1.0/r
    cdef double complex zz0_div_r = (z - z0) * invr

    cdef double complex[:] r_inv_wz0
    cdef double complex[:] exp_t
    cdef double complex[:] qbx_terms

    cdef int n_srcs = n_panels * src_panel_order
    # cdef vector[complex double] r_inv_wz0 = vector[complex double](n_panels*src_panel_order)
    with gil:
        r_inv_wz0 = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        exp_t = np.empty(n_panels * src_panel_order, dtype=np.complex128)
        qbx_terms = np.zeros(n_panels * src_panel_order, dtype=np.complex128)


    cdef double constant
    cdef double complex a0, a1, w, nw, inv_wz0, exp_m0, exp_m1

    cdef int pt_start, pt_end, pt_idx, panel_idx, src_pt_idx, j, m

    cdef double eval_m0 = (0.0 if eval_deriv else 1.0)
    cdef double complex eval_m1 = (-invr if eval_deriv else zz0_div_r)
    cdef double complex eval_t = eval_m1
    cdef double complex eval_tm

    cdef double norm_a0 = 0.0
    for panel_idx in range(panel_start, panel_end):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = (panel_idx - panel_start) * src_panel_order + pt_idx
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
                exp_m1 = -constant * r_inv_wz0[j]
            exp_t[j] = exp_m1

            a1 = exp_m1 * eval_m1
            if eval_deriv:
                # eval_m0 is zero
                norm_a0 += norm(a1)
                qbx_terms[j] = a1
            else:
                a0 = exp_m0 * eval_m0
                norm_a0 += norm(a0)
                qbx_terms[j] = a0 + a1

    cdef double complex am, am_sum
    cdef double norm_am_sum
    cdef double norm_am_sum_prev = norm_a0
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
            am = exp_t[j] * eval_tm
            am_sum += am
            qbx_terms[j] += am
        norm_am_sum = norm(am_sum)

        # We use the sum of the last two terms to avoid issues with
        # common sequences where every terms alternate in magnitude
        # See Klinteberg and Tornberg 2018 at the end of page 5
        if (norm_am_sum + norm_am_sum_prev) < 2 * (tol ** 2) * norm_a0:
            divergences = 0
            break

        if (norm_am_sum > norm_am_sum_prev):
            divergences += 1
        else:
            divergences = 0
        norm_am_sum_prev = norm_am_sum

    cdef int kernel_dim = 2 if eval_deriv else 1
    for panel_idx in range(panel_start, panel_end):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = (panel_idx - panel_start) * src_panel_order + pt_idx
            mat[obs_pt_idx, src_pt_idx, 0] += real(qbx_terms[j])
            if eval_deriv:
                mat[obs_pt_idx, src_pt_idx, 1] -= imag(qbx_terms[j])


    return pair[int, bool](m, divergences > 1)

def local_qbx_integrals(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] mat, double[:,::1] obs_pts, src,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol,
    long[:] panels, long[:] panel_starts
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
        result = single_obs(
            exp_deriv, eval_deriv,
            mat, obs_pts,
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, max_p, tol, panels, panel_starts, i, i
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
        double mult

    cdef void nearfield_single_layer(const NearfieldArgs&)
    cdef void nearfield_double_layer(const NearfieldArgs&)
    cdef void nearfield_adjoint_double_layer(const NearfieldArgs&)
    cdef void nearfield_hypersingular(const NearfieldArgs&)

def nearfield_integrals(
    kernel_name, double[:,:,::1] mat, double[:,::1] obs_pts, src,
    long[::1] panels, long[::1] panel_starts,
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
        src_panel_order, &panels[0], &panel_starts[0],
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

from libcpp.algorithm cimport lower_bound
from libcpp.vector cimport vector


def identify_nearfield_panels(
    obs_pts, radii, src_tree, int source_order, long[:] refinement_map
):
    cdef int n_obs = obs_pts.shape[0]

    # TODO: use ckdtree directly to avoid python
    src_pts = src_tree.query_ball_point(
        obs_pts, radii, return_sorted=True
    )

    cdef long[:] src_pts_starts = np.zeros(n_obs + 1, dtype=int)
    cdef long sum = 0
    cdef int i
    for i in range(n_obs):
        sum += len(src_pts[i])
        src_pts_starts[1 + i] = sum
    cdef long[:] all_src_pts = np.concatenate(src_pts)

    cdef long start, end
    cdef vector[int] panels_vector

    panel_starts_np = np.empty(n_obs + 1, dtype=int)
    cdef long[:] panel_starts = panel_starts_np
    panel_starts[0] = 0

    cdef int refined_panel, unrefined_panel, last_unrefined_panel, n_panels
    with nogil:
        for i in range(n_obs):
            start = src_pts_starts[i]
            end = src_pts_starts[i + 1]

            n_panels = 0
            last_unrefined_panel = -1
            for j in range(start, end):
                unrefined_panel = all_src_pts[j] // source_order
                if unrefined_panel == last_unrefined_panel:
                    continue

                refined_panel = lower_bound(
                    &refinement_map[0],
                    &refinement_map[refinement_map.shape[0]],
                    unrefined_panel
                ) - &refinement_map[0]

                while refinement_map[refined_panel] == unrefined_panel:
                    panels_vector.push_back(refined_panel)
                    n_panels += 1
                    refined_panel += 1

                last_unrefined_panel = unrefined_panel
            panel_starts[i + 1] = panel_starts[i] + n_panels

    panels_np = np.empty(panel_starts[n_obs], dtype=int)
    cdef long[:] panels = panels_np
    for i in range(panel_starts[n_obs]):
        panels[i] = panels_vector[i]

    return panels_np, panel_starts_np
