#include <algorithm>
#include <array>
#include <cassert>
#define _USE_MATH_DEFINES
#include "adaptive.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>

#include "direct_kernels.hpp"

struct NearfieldArgs {
    double* entries;
    long* rows;
    long* cols;

    int* n_subsets;
    double* integration_error;
    int n_obs;
    int n_src;

    int obs_dim;
    int src_dim;

    double* obs_pts;

    double* src_pts;
    double* src_normals;
    double* src_jacobians;
    double* src_param_width;
    int n_src_panels;

    double* interp_qx;
    double* interp_wts;
    int n_interp;

    double* kronrod_qx;
    double* kronrod_qw;
    double* kronrod_qw_gauss;
    int n_kronrod;

    double mult;
    double tol;
    bool adaptive;

    long* panel_obs_pts;
    long* panel_obs_pts_starts;

    double* kernel_parameters;
};

template <typename K> void _nearfield_integrals(K kernel_fnc, const NearfieldArgs& a) {

    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(DirectObsInfo{}, 0, 0, 0, 0))>::value;

    SourceData sd{a.src_pts,      a.src_normals, a.src_jacobians,    a.src_param_width,
                  a.n_src_panels, a.interp_qx,   a.interp_wts,       a.n_interp,
                  a.kronrod_qx,   a.kronrod_qw,  a.kronrod_qw_gauss, a.n_kronrod};

#pragma omp parallel
    {
        std::vector<double> memory_pool(
            (max_adaptive_integrals * 2 + 1) * a.n_interp * ndim * 2, 0.0);

        int thread_idx = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        int n_integrals = a.panel_obs_pts_starts[a.n_src_panels];
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

            int obs_i = a.panel_obs_pts[integral_idx];

            DirectObsInfo obs{a.obs_pts[obs_i * 2 + 0], a.obs_pts[obs_i * 2 + 1],
                              a.kernel_parameters};

            int n_subsets;
            double max_err;
            if (a.adaptive) {
                std::vector<double> integral(a.n_interp * ndim);
                auto result = adaptive_integrate(integral.data(), obs, kernel_fnc, sd,
                                                 cur_panel, a.tol, memory_pool.data());
                max_err = result.first;
                n_subsets = result.second;

                for (int pt_idx = 0; pt_idx < a.n_interp; pt_idx++) {
                    for (int d1 = 0; d1 < a.obs_dim; d1++) {
                        for (int d2 = 0; d2 < a.src_dim; d2++) {
                            int i = pt_idx * a.obs_dim * a.src_dim + d1 * a.src_dim + d2;
                            int start_offset = ((integral_idx) * a.obs_dim + d1) * a.n_interp * a.src_dim;
                            int entry_idx = start_offset + pt_idx * a.src_dim + d2;
                            a.entries[entry_idx] += a.mult * integral[i];
                            a.rows[entry_idx] = obs_i * a.obs_dim + d1;

                            int col = cur_panel * a.n_interp * a.src_dim + pt_idx * a.src_dim + d2;
                            a.cols[entry_idx] = col;
                        }
                    }
                }
            } else {
                std::vector<double> integral(2 * a.n_interp * ndim);
                integrate_domain(integral.data(), obs, kernel_fnc, sd, cur_panel, -1,
                                 1);
                n_subsets = 1;
                for (int pt_idx = 0; pt_idx < a.n_interp; pt_idx++) {
                    for (int d1 = 0; d1 < a.obs_dim; d1++) {
                        for (int d2 = 0; d2 < a.src_dim; d2++) {
                            int i = pt_idx * a.obs_dim * a.src_dim + d1 * a.src_dim + d2;
                            int start_offset = ((integral_idx) * a.obs_dim + d1) * a.n_interp * a.src_dim;
                            int entry_idx = start_offset + pt_idx * a.src_dim + d2;
                            a.entries[entry_idx] += a.mult * integral[2 * i];
                            a.rows[entry_idx] = obs_i * a.obs_dim + d1;

                            int col = cur_panel * a.n_interp * a.src_dim + pt_idx * a.src_dim + d2;
                            a.cols[entry_idx] = col;
                        }
                    }
                }
            }

            // Inserting matrix entries does not cause data races because each
            // matrix entry is handled by a single iteration of the loop.
            // However, summing the total number of subsets used per observation
            // point does require summing across threads, so we need to use an
            // atomic instruction.
#pragma omp atomic
            a.n_subsets[obs_i] += n_subsets;

            // TODO: critical sections sometimes suck. check the performance of this.
#pragma omp critical
            {
                a.integration_error[obs_i] =
                    std::max(a.integration_error[obs_i], max_err);
            }
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

void nearfield_elastic_U(const NearfieldArgs& a) { _nearfield_integrals(elastic_U, a); }
void nearfield_elastic_T(const NearfieldArgs& a) { _nearfield_integrals(elastic_T, a); }

void nearfield_elastic_A(const NearfieldArgs& a) { _nearfield_integrals(elastic_A, a); }

void nearfield_elastic_H(const NearfieldArgs& a) { _nearfield_integrals(elastic_H, a); }
