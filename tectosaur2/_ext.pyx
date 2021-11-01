#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport fabs, pi
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector


cdef extern from "<complex.h>" namespace "std" nogil:
    double norm(double complex z)
    double real(double complex z)
    double imag(double complex z)
    double complex log(double complex z)

cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

ctypedef double complex dcomplex

cdef pair[int, bool] single_obs(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] mat, double[:,::1] obs_pts,
    double[:,::1] src_pts, double[:,::1] src_normals,
    double[::1] src_jacobians, double[::1] src_panel_lengths,
    double[::1] src_param_width,
    double[::1] qx, double[::1] qw, double[::1] interp_wts, int nq,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol, double d_refine,
    long[:] panels, long[:] panel_starts, int obs_pt_idx, int exp_idx
) nogil:

    cdef int panel_start = panel_starts[obs_pt_idx]
    cdef int panel_end = panel_starts[obs_pt_idx + 1]
    cdef int n_panels = panel_end - panel_start
    if n_panels == 0:
        return pair[int, bool](-1, False)

    # Step 1: prepare data on the observation point
    cdef double complex z = obs_pts[obs_pt_idx, 0] + (obs_pts[obs_pt_idx, 1] * I)
    cdef double complex z0 = exp_centers[exp_idx,0] + (exp_centers[exp_idx, 1] * I)
    cdef double r = exp_rs[exp_idx]
    cdef double invr = 1.0 / r
    cdef double complex zz0_div_r = (z - z0) * invr

    cdef int pt_start, pt_end, panel_list_idx, panel_idx, src_pt_idx, j

    # Step 2: Refine the source panels.
    cdef vector[double] empty_vector
    cdef vector[vector[double]] subsets
    for i in range(n_panels):
        subsets.push_back(empty_vector)
        subsets[i].push_back(-1)
        subsets[i].push_back(1)

    cdef vector[vector[double]] new_subsets;
    for i in range(n_panels):
        new_subsets.push_back(empty_vector)

    cdef bool any_refinement = True
    cdef int pi, subset_idx
    cdef double r2
    cdef double appx_panel_L, panel_L2
    cdef double d_refine2 = d_refine * d_refine
    cdef bool subset_refined
    cdef int depth = 0
    cdef int max_depth = 20
    cdef double complex w
    while any_refinement:
        depth += 1
        if depth > max_depth:
            break

        any_refinement = False
        for i in range(n_panels):
            new_subsets[i].clear()

        for panel_list_idx in range(panel_start, panel_end):
            panel_idx = panels[panel_list_idx]
            pt_start = panel_idx * nq
            pt_end = (panel_idx + 1) * nq
            pi = panel_list_idx - panel_start

            new_subsets[pi].push_back(subsets[pi][0])
            for subset_idx in range(subsets[pi].size() - 1):
                xhat_left = subsets[pi][subset_idx]
                xhat_right = subsets[pi][subset_idx + 1]
                appx_panel_L = (xhat_right - xhat_left) * 0.5 * src_panel_lengths[panel_idx]
                panel_L2 = appx_panel_L * appx_panel_L
                subset_refined = False
                for src_pt_idx in range(pt_start, pt_end):
                    w = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
                    r2 = norm(w - z0)
                    if r2 < d_refine2 * panel_L2:
                        midpt = (xhat_left + xhat_right) * 0.5
                        new_subsets[pi].push_back(midpt)
                        new_subsets[pi].push_back(xhat_right)
                        subset_refined = True
                        any_refinement = True
                        break
                if not subset_refined:
                    new_subsets[pi].push_back(xhat_right)

        for i in range(n_panels):
            subsets[i] = new_subsets[i]

    # Step 3: construct the terms for building the QBX power series.
    cdef int n_srcs = 0
    for pi in range(n_panels):
        n_srcs += subsets[pi].size() * nq

    cdef int kernel_dim = 2 if eval_deriv else 1
    cdef vector[double complex] r_inv_wz0 = vector[dcomplex](n_srcs)
    cdef vector[double complex] exp_t = vector[dcomplex](n_srcs)
    cdef vector[double] qbx_terms = vector[double](n_srcs * kernel_dim)

    cdef double constant, jac, qxj, qxk, denom, inv_denom
    cdef double complex w_src, nw_src
    cdef double jac_src, interp_K
    cdef double complex nw, inv_wz0, exp_m0, exp_m1, t0, a0, a1


    cdef double eval_m0 = (0.0 if eval_deriv else 1.0)
    cdef double complex eval_m1 = (-invr if eval_deriv else zz0_div_r)
    cdef double complex eval_t = eval_m1
    cdef double complex eval_tm

    cdef double mag_a0 = 0.0

    cdef int subset_start = 0
    cdef int k, quad_pt_idx
    for panel_list_idx in range(panel_start, panel_end):
        panel_idx = panels[panel_list_idx]
        pt_start = panel_idx * nq
        pt_end = (panel_idx + 1) * nq
        pi = panel_list_idx - panel_start
        for subset_idx in range(subsets[pi].size() - 1):
            xhat_left = subsets[pi][subset_idx]
            xhat_right = subsets[pi][subset_idx + 1]
            for quad_pt_idx in range(nq):
                qxj = xhat_left + (qx[quad_pt_idx] + 1) * 0.5 * (xhat_right - xhat_left);

                if subsets[pi].size() == 2:
                    src_pt_idx = pt_start + quad_pt_idx
                    w = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
                    nw = src_normals[src_pt_idx,0] + src_normals[src_pt_idx,1] * I
                    jac = src_jacobians[src_pt_idx]
                else:
                    w = 0
                    nw = 0
                    jac = 0
                    denom = 0
                    for src_pt_idx in range(pt_start, pt_end):
                        k = src_pt_idx - pt_start
                        qxk = qx[k]
                        w_src = src_pts[src_pt_idx,0] + src_pts[src_pt_idx,1] * I
                        nw_src = src_normals[src_pt_idx,0] + src_normals[src_pt_idx,1] * I
                        jac_src = src_jacobians[src_pt_idx]
                        interp_K = interp_wts[k] / (qxj - qxk);
                        w += w_src * interp_K
                        nw += nw_src * interp_K
                        jac += jac_src * interp_K
                        denom += interp_K

                    inv_denom = 1.0 / denom
                    w *= inv_denom
                    nw *= inv_denom
                    jac *= inv_denom

                j = subset_start + quad_pt_idx
                inv_wz0 = 1.0 / (w - z0)

                r_inv_wz0[j] = r * inv_wz0
                constant = C * qw[quad_pt_idx] * src_param_width[panel_idx] * jac * 0.5 * (xhat_right - xhat_left) * 0.5

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
                    mag_a0 += fabs(real(a1)) + fabs(imag(a1))
                    t0 = a1
                else:
                    a0 = exp_m0 * eval_m0
                    mag_a0 += fabs(real(a0)) + fabs(imag(a0))
                    t0 = a0 + a1
                qbx_terms[j * kernel_dim + 0] = real(t0)
                if eval_deriv:
                    qbx_terms[j * kernel_dim + 1] = -imag(t0)

            subset_start += nq

    # Step 4: construct the power series.
    cdef double complex am
    cdef double am_sum_real
    cdef double am_sum_imag
    cdef double mag_am_sum
    cdef double mag_am_sum_prev = mag_a0
    cdef int divergences = 0
    cdef int m = 1

    for m in range(2, max_p + 1):
        eval_t *= zz0_div_r
        eval_tm = eval_t
        if (<int>exp_deriv) + (<int>eval_deriv) == 0:
            eval_tm /= m
        elif (<int>exp_deriv) + (<int>eval_deriv) == 2:
            eval_tm *= m

        am_sum_real = 0
        am_sum_imag = 0
        for j in range(n_srcs):
            exp_t[j] *= r_inv_wz0[j]
            am = exp_t[j] * eval_tm
            am_sum_real += real(am)
            am_sum_imag += imag(am)
            qbx_terms[j * kernel_dim + 0] += real(am)
            if eval_deriv:
                qbx_terms[j * kernel_dim + 1] -= imag(am)
        mag_am_sum = fabs(am_sum_real) + fabs(am_sum_imag)

        # We use the sum of the last two terms to avoid issues with
        # common sequences where every terms alternate in magnitude
        # See Klinteberg and Tornberg 2018 at the end of page 5
        if mag_am_sum + mag_am_sum_prev < 2 * tol * mag_a0:
            divergences = 0
            break

        if mag_am_sum > mag_am_sum_prev:
            divergences += 1
        else:
            divergences = 0
        mag_am_sum_prev = mag_am_sum

    # Step 5: Insert the results into the output matrix.
    subset_start = 0
    cdef vector[double] entries = vector[double](kernel_dim * nq)
    for panel_list_idx in range(panel_start, panel_end):
        panel_idx = panels[panel_list_idx]
        pt_start = panel_idx * nq
        pt_end = (panel_idx + 1) * nq
        pi = panel_list_idx - panel_start
        for subset_idx in range(subsets[pi].size() - 1):
            xhat_left = subsets[pi][subset_idx]
            xhat_right = subsets[pi][subset_idx + 1]
            for quad_pt_idx in range(nq):
                j = subset_start + quad_pt_idx
                qxj = xhat_left + (qx[quad_pt_idx] + 1) * 0.5 * (xhat_right - xhat_left);

                if subsets[pi].size() == 2:
                    src_pt_idx = pt_start + quad_pt_idx
                    mat[obs_pt_idx, src_pt_idx, 0] += qbx_terms[j * kernel_dim + 0]
                    if eval_deriv:
                        mat[obs_pt_idx, src_pt_idx, 1] += qbx_terms[j * kernel_dim + 1]
                else:
                    denom = 0
                    for src_pt_idx in range(pt_start, pt_end):
                        k = src_pt_idx - pt_start
                        qxk = qx[k]
                        interp_K = interp_wts[k] / (qxj - qxk);
                        entries[kernel_dim*k] = interp_K * qbx_terms[j * kernel_dim + 0]
                        if eval_deriv:
                            entries[kernel_dim*k + 1] = interp_K * qbx_terms[j * kernel_dim + 1]
                        denom += interp_K
                    inv_denom = 1.0 / denom

                    for src_pt_idx in range(pt_start, pt_end):
                        k = src_pt_idx - pt_start
                        mat[obs_pt_idx, src_pt_idx, 0] += entries[k*kernel_dim] * inv_denom
                        if eval_deriv:
                            mat[obs_pt_idx, src_pt_idx, 1] += entries[k * kernel_dim + 1] * inv_denom
            subset_start += nq

    return pair[int, bool](m, divergences > 1)

def local_qbx_integrals(
    bool exp_deriv, bool eval_deriv,
    double[:,:,::1] mat, double[:,::1] obs_pts, src,
    double[:,::1] exp_centers, double[::1] exp_rs,
    int max_p, double tol, double d_refine,
    long[:] panels, long[:] panel_starts
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_panel_lengths = src.panel_length
    cdef double[::1] src_param_width = src.panel_parameter_width

    cdef double[::1] qx = src.qx
    cdef double[::1] qw = src.qw
    cdef double[::1] interp_wts = src.interp_wts
    cdef int nq = src.qx.shape[0]

    cdef int i
    cdef np.ndarray p_np = np.empty(obs_pts.shape[0], dtype=np.int32)
    cdef int[:] p = p_np
    cdef np.ndarray kappa_too_small_np = np.empty(obs_pts.shape[0], dtype=np.bool_)
    cdef bool[:] kappa_too_small = kappa_too_small_np

    cdef pair[int,bool] result
    for i in prange(obs_pts.shape[0], nogil=True):
        result = single_obs(
            exp_deriv, eval_deriv, mat, obs_pts,
            src_pts, src_normals, src_jacobians, src_panel_lengths, src_param_width,
            qx, qw, interp_wts, nq, exp_centers, exp_rs,
            max_p, tol, d_refine, panels, panel_starts, i, i
        )
        p[i] = result.first
        kappa_too_small[i] = result.second
    return p_np, kappa_too_small_np

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
        double d_refine

    cdef void nearfield_single_layer(const NearfieldArgs&)
    cdef void nearfield_double_layer(const NearfieldArgs&)
    cdef void nearfield_adjoint_double_layer(const NearfieldArgs&)
    cdef void nearfield_hypersingular(const NearfieldArgs&)

def nearfield_integrals(
    kernel_name, double[:,:,::1] mat, double[:,::1] obs_pts, src,
    long[::1] panel_obs_pts, long[::1] panel_obs_pts_starts,
    double mult, double d_refine
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_panel_lengths = src.panel_length
    cdef double[::1] src_param_width = src.panel_parameter_width
    cdef double[::1] qx = src.qx
    cdef double[::1] qw = src.qw
    cdef double[::1] interp_wts = src.interp_wts

    # kernel = double_layer
    cdef NearfieldArgs args = NearfieldArgs(
        &mat[0,0,0], obs_pts.shape[0], src.n_pts, &obs_pts[0,0],
        &src_pts[0,0], &src_normals[0,0], &src_jacobians[0],
        &src_panel_lengths[0], &src_param_width[0], src.n_panels,
        &qx[0], &qw[0], &interp_wts[0], qx.shape[0],
        &panel_obs_pts[0], &panel_obs_pts_starts[0],
        mult, d_refine
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


def identify_nearfield_panels(obs_pts, src_pts, int n_src_panels, int source_order):
    cdef int n_obs = obs_pts.shape[0]

    cdef long[:] src_pts_starts = np.zeros(n_obs + 1, dtype=int)
    cdef long sum = 0
    cdef int i
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
