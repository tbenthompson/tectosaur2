#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport pi, fabs
from libcpp cimport bool
from libcpp.pair cimport pair

cdef extern from "<complex.h>" namespace "std" nogil:
    double real(double complex z)

cdef double complex I = 1j
cdef double C = 1.0 / (2 * pi)

cdef pair[int, bool] single_obs(
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
    cdef double complex zz0_div_r = (z - z0) / r

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


    cdef double am, am_sum
    cdef double complex w, nw, inv_wz0

    cdef double mag_a0 = 0
    cdef double complex eval_t = 1.0

    cdef int pt_start, pt_end, pt_idx, panel_idx, src_pt_idx, j, m
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
            exp_t[j] = nw * inv_wz0 * C * src_quad_wts[src_pt_idx] * src_jacobians[src_pt_idx]
            qbx_terms[j] = 0.0
            mag_a0 += fabs(real(exp_t[j] * eval_t))

    cdef double am_sum_prev = 0.0
    cdef int divergences = 0
    for m in range(max_p+1):
        am_sum = 0
        for j in range(n_srcs):
            am = real(exp_t[j] * eval_t)
            am_sum += am
            qbx_terms[j] += am
            exp_t[j] *= r_inv_wz0[j]
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
        eval_t *= zz0_div_r

    for panel_idx in range(n_panels):
        pt_start = panels[panel_idx] * src_panel_order
        pt_end = (panels[panel_idx] + 1) * src_panel_order
        for pt_idx in range(pt_end - pt_start):
            src_pt_idx = pt_start + pt_idx
            j = panel_idx * src_panel_order + pt_idx
            qbx_mat[obs_pt_idx, 0, src_pt_idx] += qbx_terms[j]

    return pair[int, bool](m, divergences > 1)

def local_qbx_integrals(
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
            qbx_mat, obs_pts,
            src_pts, src_normals, src_jacobians, src_quad_wts, src_panel_order,
            exp_centers, exp_rs, max_p, tol, panel_set, i, i
        )
        p[i], kappa_too_small[i] = result
    return p, kappa_too_small

def nearfield_integrals(
    double[:,:,::1] mat, double[:,::1] obs_pts, src,
    nearfield_panels, double mult
):

    cdef double[:,::1] src_pts = src.pts
    cdef double[:,::1] src_normals = src.normals
    cdef double[::1] src_jacobians = src.jacobians
    cdef double[::1] src_quad_wts = src.quad_wts
    cdef int src_panel_order = src.panel_order

    cdef long[:] panels
    cdef int n_panels

    cdef int i, panel_idx, pt_idx, pt_start, pt_end, j, src_pt_idx
    cdef double obsx, obsy, r2, dx, dy, G, integral

    for i in range(obs_pts.shape[0]):
        with nogil:
            obsx = obs_pts[i,0]
            obsy = obs_pts[i,1]

            with gil:
                panels = nearfield_panels[i]

            if panels.shape[0] == 0:
                continue
            n_panels = panels.shape[0]
            n_srcs = n_panels * src_panel_order

            for panel_idx in range(n_panels):
                pt_start = panels[panel_idx] * src_panel_order
                pt_end = (panels[panel_idx] + 1) * src_panel_order
                for pt_idx in range(pt_end - pt_start):
                    src_pt_idx = pt_start + pt_idx
                    j = panel_idx * src_panel_order + pt_idx
                    dx = obsx - src_pts[src_pt_idx, 0]
                    dy = obsy - src_pts[src_pt_idx, 1]
                    r2 = dx*dx + dy*dy
                    G = -C * (dx * src_normals[src_pt_idx,0] + dy * src_normals[src_pt_idx,1]) / r2
                    if r2 == 0:
                        G = 0.0
                    integral = G * src_jacobians[src_pt_idx] * src_quad_wts[src_pt_idx]

                    mat[i, 0, src_pt_idx] += mult * integral
