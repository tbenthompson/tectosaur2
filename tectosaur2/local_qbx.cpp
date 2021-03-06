#include "adaptive.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

struct LocalQBXArgs {
    // out parameters
    double* entries;
    long* rows;
    long* cols;

    int* p;
    double* integration_error;
    int* n_subsets;

    // input parameters
    double* test_density;

    // Number of observation points.
    int n_obs;
    // The number of source points.
    int n_src;

    // The number of rows in the tensor returned by the kernel function.
    int obs_dim;
    // The number of columns in the tensor returned by the kernel function.
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

    double* exp_centers;
    double* exp_rs;

    int max_p;
    double tol;

    long* panels;
    long* panel_starts;

    double* kernel_parameters;
};

struct QBXObsInfo {
    double x;
    double y;
    double expx;
    double expy;
    double expr;
    int p_start;
    int p_end;
    double* kernel_parameters;
};

// Assuming an adaptive p the order of loops has to look like:
// given an observation point... and a set of QBX source panels
// 1. loop over the expansion order until the terms are small enough to allow for
// stopping
// 2. loop over the panels
// 3. loop over the quadrature points in that panel. if the integral at this
//    level is insufficiently precise, do it again at a higher precision.
//
// In order to make this efficient, I need to store big temporary arrays of
// intermediate calculation values like 1.0 / (w - z0). The code also ends up being
// super ugly.
// It seems nicer to pull the p adaptivity way out into the outermost loop and
// then push the p integration into the innermost integral so that each obs pt
// and panel pair are mostly integrated in one shot. Conceptually, this means that I can
// deal with the QBX implementation as essentially a bunch of "regularized" kernels.

std::array<double, 2> single_layer_qbx(const QBXObsInfo& obs, double srcx, double srcy,
                                       double srcnx, double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::array<double, 2> result{};

    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> expand;
        if (m == 0) {
            expand = std::log(w - z0) / (2 * M_PI);
        } else {
            expand = -std::pow(obs.expr, m) / (m * (2 * M_PI) * std::pow(w - z0, m));
        }
        std::complex<double> eval = -std::pow((z - z0) / obs.expr, m);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
    }
    return result;
}

std::array<double, 2> double_layer_qbx(const QBXObsInfo& obs, double srcx, double srcy,
                                       double srcnx, double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 2> result{};

    constexpr double C = 1.0 / (2 * M_PI);
    auto invwz0 = 1.0 / (w - z0);
    auto ratio = (z - z0) * invwz0;

    auto term = nw * invwz0 * C;
    for (int m = 0; m < obs.p_start; m++) {
        term *= ratio;
    }

    for (int m = obs.p_start; m < obs.p_end; m++) {
        result[0] += result[1];
        result[1] = std::real(term);

        if (m < obs.p_end - 1) {
            term *= ratio;
        }
    }
    return result;
}

std::array<double, 4> adjoint_double_layer_qbx(const QBXObsInfo& obs, double srcx, double srcy,
                                               double srcnx, double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::array<double, 4> result{};

    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> expand;
        if (m == 0) {
            expand = -std::log(w - z0) / (2 * M_PI);
        } else {
            expand = std::pow(obs.expr, m) / (m * (2 * M_PI) * std::pow(w - z0, m));
        }

        std::complex<double> eval = (m / obs.expr) * std::pow((z - z0) / obs.expr, m - 1);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
        result[2] += result[3];
        result[3] = -std::imag(expand * eval);
    }
    return result;
}

std::array<double, 4> hypersingular_qbx(const QBXObsInfo& obs, double srcx, double srcy,
                                        double srcnx, double srcny) {
    constexpr double C = 1.0 / (2 * M_PI);

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    nw *= -C;

    std::array<double, 4> result{};

    auto invwz0 = 1.0 / (w - z0);
    std::complex<double> term = nw * invwz0 * invwz0;
    auto mult = (z - z0) * invwz0;

    // Skip m = 0 because that term is zero.
    int m_start = std::max(obs.p_start, 1);
    // Skip ahead to the first term that we want.
    term *= std::pow(mult, m_start - 1);

    for (int m = m_start; m < obs.p_end; m++) {
        // The summation here is a bit funny because we want to store the last
        // term separately from the rest of the series. This makes it possible
        // to check the magnitude of the last term to check for convergence.
        result[0] += result[1];
        result[1] = -std::real(term) * m;
        result[2] += result[3];
        result[3] = std::imag(term) * m;

        term *= mult;
    }
    return result;
}

std::array<double, 8> elastic_U_qbx(const QBXObsInfo& obs, double srcx, double srcy, double srcnx,
                                    double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double disp_C1 = 1.0 / (4 * M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::array<double, 8> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> G;
        if (m == 0) {
            G = -std::log(w - z0);
        } else {
            G = std::pow(ratio, m) / static_cast<double>(m);
        }
        std::complex<double> Gp = -std::pow(ratio, m) / (w - z0);
        auto t1 = kappa * (G + std::conj(G));
        auto t2 = -(w - z) * std::conj(Gp);
        result[0] += result[1];
        result[1] = std::real(disp_C1 * (t1 + t2));
        result[2] += result[3];
        result[3] = std::real(disp_C1 * (i * t1 - i * t2));
        result[4] += result[5];
        result[5] = std::imag(disp_C1 * (t1 + t2));
        result[6] += result[7];
        result[7] = std::imag(disp_C1 * (i * t1 - i * t2));
        if (m == 0) {
            result[1] += disp_C1;
            result[7] += disp_C1;
        }
    }
    return result;
}

std::array<double, 8> elastic_T_qbx(const QBXObsInfo& obs, double srcx, double srcy, double srcnx,
                                    double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double trac_C1 = -1.0 / (2 * M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 8> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> Gp = -std::pow(ratio, m) / (w - z0);
        std::complex<double> Gpp = (m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
        auto t1 = kappa * Gp * nw + std::conj(Gp * nw);
        auto t2 = nw * std::conj(Gp) - (w - z) * std::conj(Gpp * nw);
        result[0] += result[1];
        result[1] = std::real(trac_C1 * (t1 + t2));
        result[2] += result[3];
        result[3] = std::real(trac_C1 * (i * t1 - i * t2));
        result[4] += result[5];
        result[5] = std::imag(trac_C1 * (t1 + t2));
        result[6] += result[7];
        result[7] = std::imag(trac_C1 * (i * t1 - i * t2));
    }
    return result;
}

std::array<double, 12> elastic_A_qbx(const QBXObsInfo& obs, double srcx, double srcy, double srcnx,
                                     double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double trac_C1 = -1.0 / (2 * M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 12> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> Gp = std::pow(ratio, m) / (w - z0);
        std::complex<double> Gpp = -(m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
        for (int d_src = 0; d_src < 2; d_src++) {
            auto tw = static_cast<double>(d_src == 0) + static_cast<double>(d_src == 1) * i;
            // auto t1 = -kappa * std::conj(tw * Gp) - Gp * tw;
            // auto t2 = -std::conj(Gp) * tw + (w - z) * std::conj(Gpp * tw);
            auto t1 = -Gp * tw - std::conj(Gp * tw);
            auto t2 = -kappa * std::conj(Gp) * tw + (w - z) * std::conj(Gpp * tw);
            result[0 + d_src * 2 + 0] += result[0 + d_src * 2 + 1];
            result[0 + d_src * 2 + 1] = std::real(trac_C1 * (t1 + t2));
            result[4 + d_src * 2 + 0] += result[4 + d_src * 2 + 1];
            result[4 + d_src * 2 + 1] = std::real(trac_C1 * (t1 - t2));
            result[8 + d_src * 2 + 0] += result[8 + d_src * 2 + 1];
            result[8 + d_src * 2 + 1] = std::imag(trac_C1 * (-t1 + t2));
        }
    }
    return result;
}

std::array<double, 12> elastic_H_qbx(const QBXObsInfo& obs, double srcx, double srcy, double srcnx,
                                     double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double trac_C1 = -1.0 / (M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 12> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> Gpp = -(m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
        std::complex<double> Gppp =
            (m + 1.0) * (m + 2.0) * std::pow(ratio, m) / ((w - z0) * (w - z0) * (w - z0));
        for (int d_src = 0; d_src < 2; d_src++) {
            auto uw = static_cast<double>(d_src == 0) + static_cast<double>(d_src == 1) * i;
            auto t1 = Gpp * nw * uw + std::conj(Gpp * nw * uw);
            auto t2 = std::conj(Gpp) * (nw * std::conj(uw) + std::conj(nw) * uw) -
                      (w - z) * std::conj(Gppp * nw * uw);
            result[0 + d_src * 2 + 0] += result[0 + d_src * 2 + 1];
            result[0 + d_src * 2 + 1] = std::real(trac_C1 * (t1 + t2));
            result[4 + d_src * 2 + 0] += result[4 + d_src * 2 + 1];
            result[4 + d_src * 2 + 1] = std::real(trac_C1 * (t1 - t2));
            result[8 + d_src * 2 + 0] += result[8 + d_src * 2 + 1];
            result[8 + d_src * 2 + 1] = std::imag(trac_C1 * (-t1 + t2));
        }
    }
    return result;
}

template <typename K> void _local_qbx_integrals(K kernel_fnc, const LocalQBXArgs& a) {

    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(QBXObsInfo{}, 0, 0, 0, 0))>::value;

    // Critical: the kernel outputs twice as many values as the dimensionality
    // of the kernel because we want to use the magnitude of the last value in
    // order to estimate the error in the QBX expansion.
    constexpr size_t ndim = n_kernel_outputs / 2;

    double coefficient_tol = a.tol;
    double truncation_tol = a.tol;

    int Nv = a.n_interp * n_kernel_outputs;

    SourceData sd{a.src_pts,      a.src_normals, a.src_jacobians,    a.src_param_width,
                  a.n_src_panels, a.interp_qx,   a.interp_wts,       a.n_interp,
                  a.kronrod_qx,   a.kronrod_qw,  a.kronrod_qw_gauss, a.n_kronrod};

#pragma omp parallel
    {
        std::vector<double> memory_pool(
            (max_adaptive_integrals * 2 + 1) * a.n_interp * n_kernel_outputs * 2, 0.0);

#pragma omp for
        for (int obs_i = 0; obs_i < a.n_obs; obs_i++) {
            auto panel_start = a.panel_starts[obs_i];
            auto panel_end = a.panel_starts[obs_i + 1];
            auto n_panels = panel_end - panel_start;
            QBXObsInfo obs{a.obs_pts[obs_i * 2 + 0],
                           a.obs_pts[obs_i * 2 + 1],
                           a.exp_centers[obs_i * 2 + 0],
                           a.exp_centers[obs_i * 2 + 1],
                           a.exp_rs[obs_i],
                           0,
                           0,
                           a.kernel_parameters};

            int p_step = 15;
            bool converged = false;
            obs.p_start = 0;
            a.integration_error[obs_i] = 0;
            std::vector<double> temp_out(n_panels * Nv);

            while (!converged and obs.p_start <= a.max_p) {
                obs.p_end = std::min(obs.p_start + p_step, a.max_p + 1);
                p_step = 10;
                for (int i = 0; i < n_panels * Nv; i++) {
                    temp_out[i] = 0;
                }

                int n_subsets = 0;
                for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
                    auto panel_idx = a.panels[panel_offset + panel_start];
                    double* temp_out_ptr = &temp_out[panel_offset * Nv];
                    auto result = adaptive_integrate(temp_out_ptr, obs, kernel_fnc, sd, panel_idx,
                                                     coefficient_tol, memory_pool.data());
                    n_subsets += result.second;
                    double max_err = result.first;
                    a.integration_error[obs_i] = std::max(a.integration_error[obs_i], max_err);

                    // if (max_err > 1000 * coefficient_tol) {
                    // double srcx = a.src_pts[panel_idx * a.n_interp * 2 +
                    //                         (a.n_interp / 2) * 2 + 0];
                    // double srcy = a.src_pts[panel_idx * a.n_interp * 2 +
                    //                         (a.n_interp / 2) * 2 + 1];
                    // std::cout
                    //     << "Integration failed for observation point (" << obs.x
                    //     << ", " << obs.y << ") "
                    //     << ", source panel center at: (" << srcx << ", "
                    //     << srcy << "), panel_idx: " << panel_idx
                    //     << ", n_integrals: " << n_integrals << std::endl;
                    // std::cout
                    //     << "Expansion center: (" << obs.expx << ", " << obs.expy
                    //     << ") with expansion radius: " << obs.expr << std::endl;
                    // std::cout << "The maximum estimated coefficient error: "
                    //           << max_err
                    //           << " with tolerance: " << coefficient_tol
                    //           << std::endl;
                    // }
                }

                // Add the integral and calculate series convergence.
                std::array<double, ndim> p_end_integral{};

                for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
                    auto panel_idx = a.panels[panel_offset + panel_start];

                    for (int pt_idx = 0; pt_idx < a.n_interp; pt_idx++) {
                        for (int d1 = 0; d1 < a.obs_dim; d1++) {
                            for (int d2 = 0; d2 < a.src_dim; d2++) {
                                int d = d1 * a.src_dim + d2;
                                int k = panel_offset * a.n_interp * ndim + pt_idx * ndim + d;
                                double all_but_last_term = temp_out[2 * k];
                                double last_term = temp_out[2 * k + 1];

                                int start_offset = ((panel_start + panel_offset) * a.obs_dim + d1) * a.n_interp * a.src_dim;
                                int entry_idx = start_offset + pt_idx * a.src_dim + d2;
                                a.entries[entry_idx] += all_but_last_term + last_term;
                                a.rows[entry_idx] = obs_i * a.obs_dim + d1;

                                int col = panel_idx * a.n_interp * a.src_dim + pt_idx * a.src_dim + d2;
                                a.cols[entry_idx] = col;
                                p_end_integral[d] += last_term * a.test_density[col];
                            }
                        }
                    }
                }
                a.n_subsets[obs_i] = n_subsets;

                converged = true;
                for (int d = 0; d < ndim; d++) {
                    if (fabs(p_end_integral[d]) >= truncation_tol) {
                        converged = false;
                        break;
                    }
                }

                obs.p_start = obs.p_end;
            }

            a.p[obs_i] = obs.p_end - 1;
        }
    }
}

void local_qbx_single_layer(const LocalQBXArgs& a) { _local_qbx_integrals(single_layer_qbx, a); }

void local_qbx_double_layer(const LocalQBXArgs& a) { _local_qbx_integrals(double_layer_qbx, a); }

void local_qbx_adjoint_double_layer(const LocalQBXArgs& a) {
    _local_qbx_integrals(adjoint_double_layer_qbx, a);
}

void local_qbx_hypersingular(const LocalQBXArgs& a) { _local_qbx_integrals(hypersingular_qbx, a); }

void local_qbx_elastic_U(const LocalQBXArgs& a) { _local_qbx_integrals(elastic_U_qbx, a); }

void local_qbx_elastic_T(const LocalQBXArgs& a) { _local_qbx_integrals(elastic_T_qbx, a); }

void local_qbx_elastic_A(const LocalQBXArgs& a) { _local_qbx_integrals(elastic_A_qbx, a); }

void local_qbx_elastic_H(const LocalQBXArgs& a) { _local_qbx_integrals(elastic_H_qbx, a); }

void cpp_choose_expansion_circles(double* exp_centers, double* exp_rs, double* obs_pts, int n_obs,
                                  double* offset_vector, long* owner_panel_idx, double* src_pts,
                                  double* interp_mat, int n_interp, int nq, long* panels,
                                  long* panel_starts, double* singularities,
                                  long* nearby_singularities, long* nearby_singularity_starts,
                                  double nearby_safety_ratio, double singularity_safety_ratio) {

#pragma omp parallel for
    for (int i = 0; i < n_obs; i++) {
        double obsx = obs_pts[i * 2 + 0];
        double obsy = obs_pts[i * 2 + 1];
        double offx = offset_vector[i * 2 + 0];
        double offy = offset_vector[i * 2 + 1];

        double R = exp_rs[i];
        double expx = obsx + offx * R;
        double expy = obsy + offy * R;

        auto panel_start = panel_starts[i];
        auto panel_end = panel_starts[i + 1];

        auto sing_start = nearby_singularity_starts[i];
        auto sing_end = nearby_singularity_starts[i + 1];

        auto violation_fnc = [&](double dangerx, double dangery, double safety_ratio) {
            double dx = expx - dangerx;
            double dy = expy - dangery;
            double dist2 = dx * dx + dy * dy;
            if (dist2 < (safety_ratio * safety_ratio) * R * R) {
                return 0.6;
            }
            return 1.0;
        };

        // I put the loop body mostly inside this function in order to make the
        // control flow simpler. I used a lambda to keep the function conceptually
        // coherent with the surrounding code.
        auto find_violations = [&]() {
            for (int si = sing_start; si < sing_end; si++) {
                auto singularity_idx = nearby_singularities[si];
                double singx = singularities[singularity_idx * 2 + 0];
                double singy = singularities[singularity_idx * 2 + 1];
                auto violation = violation_fnc(singx, singy, singularity_safety_ratio);
                // std::cout << "sing: " << si << " " << singx << " " << singy <<
                // std::endl;
                if (violation != 1.0) {
                    return violation;
                }
            }

            for (int pi = panel_start; pi < panel_end; pi++) {
                auto panel_idx = panels[pi];
                if (panel_idx == owner_panel_idx[i]) {
                    continue;
                }

                for (int pt_idx = 0; pt_idx < n_interp; pt_idx++) {
                    double srcx = 0;
                    double srcy = 0;
                    for (int interp_idx = 0; interp_idx < nq; interp_idx++) {
                        auto src_pt_idx = panel_idx * nq + interp_idx;
                        srcx += interp_mat[pt_idx * nq + interp_idx] * src_pts[src_pt_idx * 2 + 0];
                        srcy += interp_mat[pt_idx * nq + interp_idx] * src_pts[src_pt_idx * 2 + 1];
                    }
                    auto violation = violation_fnc(srcx, srcy, nearby_safety_ratio);
                    if (violation != 1.0) {
                        return violation;
                    }
                }
            }
            return 1.0;
        };

        int max_iter = 30;
        for (int j = 0; j < max_iter; j++) {
            double violation = find_violations();

            if (violation == 1.0) {
                break;
            }

            R *= violation;
            expx = obsx + offx * R;
            expy = obsy + offy * R;
        }

        exp_rs[i] = R;
        exp_centers[i * 2 + 0] = expx;
        exp_centers[i * 2 + 1] = expy;
    }
}
