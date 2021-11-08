#include <algorithm>
#include <array>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>

constexpr double C = 1.0 / (2 * M_PI);
constexpr double C2 = 1.0 / (4 * M_PI);
constexpr double too_close = 1e-16;

inline std::array<double, 1> single_layer(double obsx, double obsy, double srcx,
                                          double srcy, double srcnx, double srcny) {
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double G = C2 * log(r2);
    if (r2 <= too_close) {
        G = 0.0;
    }
    return {G};
}

inline std::array<double, 1> double_layer(double obsx, double obsy, double srcx,
                                          double srcy, double srcnx, double srcny) {
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }

    return {-C * (dx * srcnx + dy * srcny) * invr2};
}

inline std::array<double, 2> adjoint_double_layer(double obsx, double obsy, double srcx,
                                                  double srcy, double srcnx,
                                                  double srcny) {
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }
    double F = -C * invr2;

    return {F * dx, F * dy};
}

inline std::array<double, 2> hypersingular(double obsx, double obsy, double srcx,
                                           double srcy, double srcnx, double srcny) {
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }

    double A = 2 * (dx * srcnx + dy * srcny) * invr2;
    double B = C * invr2;
    return {B * (srcnx - A * dx), B * (srcny - A * dy)};
}

struct NearfieldArgs {
    double* mat;
    int* n_subsets;
    int n_obs;
    int n_src;
    double* obs_pts;
    double* src_pts;
    double* src_normals;
    double* src_jacobians;
    double* src_panel_lengths;
    double* src_param_width;
    int src_n_panels;
    double *qx;
    double *qw;
    double *interp_wts;
    int nq;
    long* panel_obs_pts;
    long* panel_obs_pts_starts;
    double mult;
    double tol;
    bool adaptive;
};

template <typename K>
void integrate_domain(double* out, K kernel_fnc, const NearfieldArgs& a, int panel_idx,
                      double obsx, double obsy, double xhat_left, double xhat_right) {
    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0))>::value;

    int pt_start = panel_idx * a.nq;

    if (xhat_left == -1 && xhat_right == 1) {
        for (int k = 0; k < a.nq; k++) {
            int src_pt_idx = pt_start + k;
            double srcx = a.src_pts[src_pt_idx * 2 + 0];
            double srcy = a.src_pts[src_pt_idx * 2 + 1];
            double srcnx = a.src_normals[src_pt_idx * 2 + 0];
            double srcny = a.src_normals[src_pt_idx * 2 + 1];
            double srcmult = a.mult * a.src_jacobians[src_pt_idx] * a.qw[k] *
                             a.src_param_width[panel_idx] * 0.5;

            auto kernel = kernel_fnc(obsx, obsy, srcx, srcy, srcnx, srcny);

            // We don't want to store the integration result in the output
            // matrix yet because we might abort this integration and refine
            // another level at any point. So, we store the integration result
            // in a temporary variable.
            for (size_t dim = 0; dim < ndim; dim++) {
                int entry = k * ndim + dim;
                out[entry] += kernel[dim] * srcmult;
            }
        }
    } else {
        for (int j = 0; j < a.nq; j++) {
            double qxj = xhat_left + (a.qx[j] + 1) * 0.5 * (xhat_right - xhat_left);

            double srcx = 0.0;
            double srcy = 0.0;
            double srcnx = 0.0;
            double srcny = 0.0;
            double srcjac = 0.0;
            double denom = 0.0;

                for (int k = 0; k < a.nq; k++) {
                int src_pt_idx = k + pt_start;
                    double interp_K = a.interp_wts[k] / (qxj - a.qx[k]);

                denom += interp_K;
                srcx += interp_K * a.src_pts[src_pt_idx * 2 + 0];
                srcy += interp_K * a.src_pts[src_pt_idx * 2 + 1];
                srcnx += interp_K * a.src_normals[src_pt_idx * 2 + 0];
                srcny += interp_K * a.src_normals[src_pt_idx * 2 + 1];
                srcjac += interp_K * a.src_jacobians[src_pt_idx];
            }

            double inv_denom = 1.0 / denom;
            srcx *= inv_denom;
            srcy *= inv_denom;
            srcnx *= inv_denom;
            srcny *= inv_denom;
            srcjac *= inv_denom;

            double srcmult = a.mult * srcjac * a.qw[j] * a.src_param_width[panel_idx] *
                             0.5 * (xhat_right - xhat_left) * 0.5;

            auto kernel = kernel_fnc(obsx, obsy, srcx, srcy, srcnx, srcny);
            for (size_t dim = 0; dim < ndim; dim++) {
                kernel[dim] *= srcmult;
            }

            for (int k = 0; k < a.nq; k++) {
                double interp_K = a.interp_wts[k] * inv_denom / (qxj - a.qx[k]);
                for (size_t dim = 0; dim < ndim; dim++) {
                    int entry = k * ndim + dim;
                    out[entry] += kernel[dim] * interp_K;
                }
            }
        }
    }
}

template <typename K>
int adaptive_integrate(double* out, double* baseline, K kernel_fnc,
                        const NearfieldArgs& a, int panel_idx, double obsx, double obsy,
                        double xhat_left, double xhat_right, int depth) {
    constexpr int max_depth = 10;

    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0))>::value;
    int Nv = a.nq * ndim;

    double midpt = (xhat_right + xhat_left) / 2.0;
    std::vector<double> est2_left(Nv, 0.0);
    integrate_domain(est2_left.data(), kernel_fnc, a, panel_idx, obsx, obsy, xhat_left,
                     midpt);
    std::vector<double> est2_right(Nv, 0.0);
    integrate_domain(est2_right.data(), kernel_fnc, a, panel_idx, obsx, obsy, midpt,
                     xhat_right);

    bool refine = false;
    double subset_tol = (xhat_right - xhat_left) * 0.5 * a.tol;
    for (int i = 0; i < Nv; i++) {
        double est2 = est2_left[i] + est2_right[i];
        double diff = est2 - baseline[i];
        double err = fabs(diff);
        if (err > subset_tol) {
            refine = true;
        }
    }

    if (!refine || depth >= max_depth) {
        for (int i = 0; i < Nv; i++) {
            double est2 = est2_left[i] + est2_right[i];
            out[i] += est2;
        }
        return 2;
    } else {
        return (adaptive_integrate(out, est2_left.data(), kernel_fnc, a, panel_idx,
                                   obsx, obsy, xhat_left, midpt, depth + 1) +
                adaptive_integrate(out, est2_right.data(), kernel_fnc, a, panel_idx,
                                   obsx, obsy, midpt, xhat_right, depth + 1));
    }
}

template <typename K> void _nearfield_integrals(K kernel_fnc, const NearfieldArgs& a) {

    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0))>::value;

#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        int n_integrals = a.panel_obs_pts_starts[a.src_n_panels];
        int n_integrals_per_thread = n_integrals / n_threads;
        int start = n_integrals_per_thread * thread_idx;
        int end = n_integrals_per_thread * (thread_idx + 1);
        if (thread_idx == n_threads - 1) {
            end = n_integrals;
        }

        int cur_panel = 0;
        for (int integral_idx = start; integral_idx < end; integral_idx++) {
            while (integral_idx >= a.panel_obs_pts_starts[cur_panel + 1]) {
                cur_panel += 1;
            }

            int pt_start = cur_panel * a.nq;
            int obs_i = a.panel_obs_pts[integral_idx];
            double* out_ptr = &a.mat[obs_i * a.n_src * ndim + pt_start * ndim];
            double obsx = a.obs_pts[obs_i * 2 + 0];
            double obsy = a.obs_pts[obs_i * 2 + 1];

            std::vector<double> baseline(a.nq * ndim);
            int n_subsets = 1;
            integrate_domain(baseline.data(), kernel_fnc, a, cur_panel, obsx, obsy, -1,
                             1);
            if (a.adaptive) {
                n_subsets += adaptive_integrate(out_ptr, baseline.data(), kernel_fnc, a,
                                                cur_panel, obsx, obsy, -1, 1, 0);
            } else {
                for (int i = 0; i < a.nq * ndim; i++) {
                    out_ptr[i] += baseline[i];
                }
            }
            #pragma omp atomic
            a.n_subsets[obs_i] += n_subsets;
        }
    }
}

void nearfield_single_layer(const NearfieldArgs& a) {
    _nearfield_integrals(single_layer, a);
}

void nearfield_double_layer(const NearfieldArgs& a) {
    _nearfield_integrals(double_layer, a);
}

void nearfield_adjoint_double_layer(const NearfieldArgs& a) {
    _nearfield_integrals(adjoint_double_layer, a);
}

void nearfield_hypersingular(const NearfieldArgs& a) {
    _nearfield_integrals(hypersingular, a);
}
