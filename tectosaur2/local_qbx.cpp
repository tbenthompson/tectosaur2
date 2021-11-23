#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

struct LocalQBXArgs {
    // out parameters
    double* mat;
    int* p;
    int* failed;
    int* n_subsets;

    // input parameters
    int n_obs;
    int n_src;
    double* obs_pts;
    double* src_pts;
    double* src_normals;
    double* src_jacobians;
    double* src_panel_lengths;
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
    bool safety_mode;

    long* panels;
    long* panel_starts;

    double* kernel_parameters;
};

struct ObsInfo {
    double x;
    double y;
    double expx;
    double expy;
    double expr;
    int p_start;
    int p_end;
    double* kernel_parameters;
};

// I could go with a fixed p. That would make the
//

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

std::array<double, 2> single_layer_qbx(const ObsInfo& obs, double srcx, double srcy,
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
        std::complex<double> eval = std::pow((z - z0) / obs.expr, m);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
    }
    return result;
}

std::array<double, 2> double_layer_qbx(const ObsInfo& obs, double srcx, double srcy,
                                       double srcnx, double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 2> result{};

    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> expand =
            nw * std::pow(obs.expr, m) / (2 * M_PI * std::pow(w - z0, m + 1));
        std::complex<double> eval = std::pow((z - z0) / obs.expr, m);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
    }
    return result;
}

std::array<double, 4> adjoint_double_layer_qbx(const ObsInfo& obs, double srcx,
                                               double srcy, double srcnx,
                                               double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::array<double, 4> result{};

    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> expand;
        if (m == 0) {
            expand = std::log(w - z0) / (2 * M_PI);
        } else {
            expand = -std::pow(obs.expr, m) / (m * (2 * M_PI) * std::pow(w - z0, m));
        }

        std::complex<double> eval =
            (-m / obs.expr) * std::pow((z - z0) / obs.expr, m - 1);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
        result[2] += result[3];
        result[3] = -std::imag(expand * eval);
    }
    return result;
}

std::array<double, 4> hypersingular_qbx(const ObsInfo& obs, double srcx, double srcy,
                                        double srcnx, double srcny) {
    constexpr double C = 1.0 / (2 * M_PI);

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    nw *= C;

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

std::array<double, 8> elastic_U_qbx(const ObsInfo& obs, double srcx, double srcy,
                                    double srcnx, double srcny) {
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

std::array<double, 8> elastic_T_qbx(const ObsInfo& obs, double srcx, double srcy,
                                    double srcnx, double srcny) {
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
        std::complex<double> Gpp =
            (m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
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

std::array<double, 12> elastic_A_qbx(const ObsInfo& obs, double srcx, double srcy,
                                     double srcnx, double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double trac_C1 = 1.0 / (2 * M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 12> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> Gp = std::pow(ratio, m) / (w - z0);
        std::complex<double> Gpp =
            -(m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
        for (int d_src = 0; d_src < 2; d_src++) {
            auto tw =
                static_cast<double>(d_src == 0) + static_cast<double>(d_src == 1) * i;
            auto t1 = -kappa * std::conj(tw * Gp) - Gp * tw;
            auto t2 = -std::conj(Gp) * tw + (w - z) * std::conj(Gpp * tw);
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

std::array<double, 12> elastic_H_qbx(const ObsInfo& obs, double srcx, double srcy,
                                     double srcnx, double srcny) {
    const std::complex<double> i(0.0, 1.0);
    double poisson_ratio = obs.kernel_parameters[0];
    double kappa = 3 - 4 * poisson_ratio;
    double trac_C1 = 1.0 / (M_PI * (1 + kappa));

    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::complex<double> nw = {srcnx, srcny};
    std::array<double, 12> result{};

    auto ratio = (z - z0) / (w - z0);
    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> Gpp =
            -(m + 1.0) * std::pow(ratio, m) / ((w - z0) * (w - z0));
        std::complex<double> Gppp = (m + 1.0) * (m + 2.0) * std::pow(ratio, m) /
                                    ((w - z0) * (w - z0) * (w - z0));
        for (int d_src = 0; d_src < 2; d_src++) {
            auto uw =
                static_cast<double>(d_src == 0) + static_cast<double>(d_src == 1) * i;
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

template <typename K>
void integrate_domain(double* out, K kernel_fnc, const LocalQBXArgs& a, int panel_idx,
                      const ObsInfo& obs, double xhat_left, double xhat_right) {

    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(ObsInfo{}, 0, 0, 0, 0))>::value;

    int pt_start = panel_idx * a.n_interp;

    for (int j = 0; j < a.n_kronrod; j++) {
        double qxj = xhat_left + (a.kronrod_qx[j] + 1) * 0.5 * (xhat_right - xhat_left);

        double srcx = 0.0;
        double srcy = 0.0;
        double srcnx = 0.0;
        double srcny = 0.0;
        double srcjac = 0.0;
        double denom = 0.0;

        for (int k = 0; k < a.n_interp; k++) {
            int src_pt_idx = k + pt_start;
            double interp_K = a.interp_wts[k] / (qxj - a.interp_qx[k]);

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

        auto kernel = kernel_fnc(obs, srcx, srcy, srcnx, srcny);

        double srcmult = srcjac * a.src_param_width[panel_idx] * 0.5 *
                         (xhat_right - xhat_left) * 0.5;
        for (size_t d = 0; d < n_kernel_outputs; d++) {
            kernel[d] *= srcmult;
        }

        for (int k = 0; k < a.n_interp; k++) {
            double interp_K = a.interp_wts[k] * inv_denom / (qxj - a.interp_qx[k]);
            for (size_t d = 0; d < n_kernel_outputs; d++) {
                int entry = k * n_kernel_outputs * 2 + d * 2;
                // todo: multiply by the two quadrature weights and add to the two
                // outputs.
                double value = kernel[d] * interp_K;
                double kronrod_value = value * a.kronrod_qw[j];
                out[entry] += kronrod_value;
                double gauss_value = 0;
                if (j % 2 == 1) {
                    gauss_value = value * a.kronrod_qw_gauss[j / 2];
                }
                // Error estimate from the nested Gauss-Kronrod quadrature rule.
                // note that this is a *difference* and not a error.
                out[entry + 1] += kronrod_value - gauss_value;
            }
        }
    }
}

struct EstimatedIntegral {
    double xhat_left;
    double xhat_right;
    double max_err;
    std::vector<double> value;
};

template <typename K>
std::pair<bool, int> adaptive_integrate(double* out, K kernel_fnc,
                                        const LocalQBXArgs& a, int panel_idx,
                                        const ObsInfo& obs, double tol) {
    constexpr int max_integrals = 100;

    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(ObsInfo{}, 0, 0, 0, 0))>::value;

    int Nv = a.n_interp * ndim;

    // We store twice as many values here because we're storing both an integral
    // and an error estimate.
    std::vector<double> integral(Nv * 2, 0.0);
    double max_err = 0;

    integrate_domain(integral.data(), kernel_fnc, a, panel_idx, obs, -1, 1);
    for (int i = 0; i < Nv; i++) {
        integral[2 * i + 1] = fabs(integral[2 * i + 1]);
        max_err = std::max(integral[2 * i + 1], max_err);
    }
    EstimatedIntegral initial_integral{-1, 1, max_err, integral};

    std::vector<EstimatedIntegral> next_integrals;
    auto heap_compare = [](auto& a, auto& b) { return a.max_err < b.max_err; };
    next_integrals.push_back(initial_integral);

    int integral_idx = 0;
    for (; integral_idx < max_integrals; integral_idx++) {
        // std::cout << integral_idx << " " << max_err << std::endl;
        auto& cur_integral = next_integrals.front();

        double midpt = (cur_integral.xhat_right + cur_integral.xhat_left) * 0.5;
        EstimatedIntegral left_child{cur_integral.xhat_left, midpt, 0};
        left_child.value.resize(Nv * 2);
        integrate_domain(left_child.value.data(), kernel_fnc, a, panel_idx, obs,
                         cur_integral.xhat_left, midpt);

        EstimatedIntegral right_child{midpt, cur_integral.xhat_right, 0};
        right_child.value.resize(Nv * 2);
        integrate_domain(right_child.value.data(), kernel_fnc, a, panel_idx, obs, midpt,
                         cur_integral.xhat_right);

        // Update the integral and its corresponding error estimate.
        max_err = 0;
        for (int i = 0; i < Nv; i++) {
            right_child.value[2*i+1] = fabs(right_child.value[2*i+1]);
            left_child.value[2*i+1] = fabs(left_child.value[2*i+1]);
            auto right_err = right_child.value[2*i+1];
            auto left_err = left_child.value[2*i+1];
            integral[2 * i] += (
                -cur_integral.value[2 * i] + left_child.value[2 * i] + right_child.value[2 * i]
            );
            integral[2 * i + 1] += -cur_integral.value[2 * i + 1] + left_err + right_err;
            left_child.max_err = std::max(left_child.max_err, left_err);
            right_child.max_err = std::max(right_child.max_err, right_err);
            max_err = std::max(integral[2 * i + 1], max_err);
        }

        // Update heap by removing the top entry that we just processed and
        // adding the two new children.
        std::pop_heap(next_integrals.begin(), next_integrals.end(), heap_compare);
        next_integrals.pop_back();
        next_integrals.push_back(std::move(left_child));
        std::push_heap(next_integrals.begin(), next_integrals.end(), heap_compare);
        next_integrals.push_back(std::move(right_child));
        std::push_heap(next_integrals.begin(), next_integrals.end(), heap_compare);

        if (max_err < tol) {
            break;
        }
    }

    bool failed = false;
    if (integral_idx == max_integrals) {
        if (max_err > 1000 * tol) {
            double srcx = a.src_pts[panel_idx * a.n_interp * 2 + (a.n_interp / 2) * 2 + 0];
            double srcy = a.src_pts[panel_idx * a.n_interp * 2 + (a.n_interp / 2) * 2 + 1];
            std::cout << "max fail! " << obs.x << " " << obs.y << " " << srcx << " "
                      << srcy << " " << panel_idx << " " << integral_idx << std::endl;
            std::cout << "exp: " << obs.expx << " " << obs.expy << " " << obs.expr
                      << std::endl;
            std::cout << "max err: " << max_err << "   tol: " << tol << std::endl;
            for (int i = 0; i < ndim; i++) {
                std::cout << integral[i] << std::endl;
            }
            for (int i = 0; i < std::min(8, (int)next_integrals.size()); i++) {
                std::cout << "option " << i << " " << next_integrals[i].max_err
                          << std::endl;
            }
        }
        if (max_err > 10 * tol) {
            failed = true;
        }
    }
    std::cout << max_err << std::endl;

    for (int i = 0; i < Nv; i++) {
        out[i] += integral[2*i];
    }

    //TODO: Could return the actual error here!
    return std::make_pair(failed, integral_idx);
}

template <typename K> void _local_qbx_integrals(K kernel_fnc, const LocalQBXArgs& a) {

    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(ObsInfo{}, 0, 0, 0, 0))>::value;

    // Critical: the kernel outputs twice as many values as the dimensionality
    // of the kernel because we want to use the magnitude of the last value in
    // order to estimate the error in the QBX expansion.
    constexpr size_t ndim = n_kernel_outputs / 2;

    double coefficient_tol = a.tol;
    double truncation_tol = a.tol;

#pragma omp parallel for
    for (int obs_i = 0; obs_i < a.n_obs; obs_i++) {
        auto panel_start = a.panel_starts[obs_i];
        auto panel_end = a.panel_starts[obs_i + 1];
        auto n_panels = panel_end - panel_start;
        ObsInfo obs{a.obs_pts[obs_i * 2 + 0],
                    a.obs_pts[obs_i * 2 + 1],
                    a.exp_centers[obs_i * 2 + 0],
                    a.exp_centers[obs_i * 2 + 1],
                    a.exp_rs[obs_i],
                    0,
                    0,
                    a.kernel_parameters};

        bool converged = false;
        obs.p_start = 0;
        std::vector<double> integral(n_panels * a.n_interp * ndim, 0.0);

        int p_step = 4;
        bool failed = false;
        while (!converged and obs.p_start <= a.max_p) {
            obs.p_end = std::min(obs.p_start + p_step, a.max_p + 1);

            std::vector<double> temp_out(n_panels * a.n_interp * ndim * 2, 0.0);
            int n_subsets = 0;
            for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
                auto panel_idx = a.panels[panel_offset + panel_start];
                double* temp_out_ptr = &temp_out[panel_offset * a.n_interp * ndim * 2];
                auto result = adaptive_integrate(temp_out_ptr, kernel_fnc, a, panel_idx,
                                                 obs, coefficient_tol);
                failed = failed || result.first;
                n_subsets += result.second;
            }

            // Add the integral and calculate series convergence.
            std::array<double, ndim> p_end_integral{};
            for (int pt_idx = 0; pt_idx < n_panels * a.n_interp; pt_idx++) {
                for (int d = 0; d < ndim; d++) {
                    int k = pt_idx * ndim + d;
                    double all_but_last_term = temp_out[2 * k];
                    double last_term = temp_out[2 * k + 1];
                    integral[k] += all_but_last_term + last_term;
                    if (a.safety_mode) {
                        p_end_integral[d] += fabs(last_term);
                    } else {
                        p_end_integral[d] += last_term;
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

        a.failed[obs_i] = failed;
        a.p[obs_i] = obs.p_end - 1;

        for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
            auto panel_idx = a.panels[panel_offset + panel_start];
            double* integral_ptr = &integral[panel_offset * a.n_interp * ndim];
            double* out_ptr = &a.mat[obs_i * a.n_src * ndim + panel_idx * a.n_interp * ndim];
            for (int k = 0; k < a.n_interp * ndim; k++) {
                out_ptr[k] += integral_ptr[k];
            }
        }
    }
}

void local_qbx_single_layer(const LocalQBXArgs& a) {
    _local_qbx_integrals(single_layer_qbx, a);
}

void local_qbx_double_layer(const LocalQBXArgs& a) {
    _local_qbx_integrals(double_layer_qbx, a);
}

void local_qbx_adjoint_double_layer(const LocalQBXArgs& a) {
    _local_qbx_integrals(adjoint_double_layer_qbx, a);
}

void local_qbx_hypersingular(const LocalQBXArgs& a) {
    _local_qbx_integrals(hypersingular_qbx, a);
}

void local_qbx_elastic_U(const LocalQBXArgs& a) {
    _local_qbx_integrals(elastic_U_qbx, a);
}

void local_qbx_elastic_T(const LocalQBXArgs& a) {
    _local_qbx_integrals(elastic_T_qbx, a);
}

void local_qbx_elastic_A(const LocalQBXArgs& a) {
    _local_qbx_integrals(elastic_A_qbx, a);
}

void local_qbx_elastic_H(const LocalQBXArgs& a) {
    _local_qbx_integrals(elastic_H_qbx, a);
}
