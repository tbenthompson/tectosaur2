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
    double* qx;
    double* qw;
    double* interp_wts;
    int nq;
    double* exp_centers;
    double* exp_rs;
    long* panels;
    long* panel_starts;
    int max_p;
    double tol;
};

struct ObsInfo {
    double x;
    double y;
    double expx;
    double expy;
    double expr;
    int p_start;
    int p_end;
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

std::array<double, 8> plane_U_qbx(const ObsInfo& obs, double srcx, double srcy, double srcnx, double srcny) {
    std::complex<double> w = {srcx, srcy};
    std::complex<double> z0 = {obs.expx, obs.expy};
    std::complex<double> z = {obs.x, obs.y};
    std::array<double, 2> result{};

    for (int m = obs.p_start; m < obs.p_end; m++) {
        std::complex<double> expand;
        if (m == 0) {
            expand = std::log(w - z0);
        } else {
            expand = -1.0 / (static_cast<double>(m) * std::pow(w - z0, m));
        }
        std::complex<double> eval = std::pow(z - z0, m);

        result[0] += result[1];
        result[1] = std::real(expand * eval);
    }
}

template <typename K>
void integrate_domain(double* out, K kernel_fnc, const LocalQBXArgs& a, int panel_idx,
                      const ObsInfo& obs, double xhat_left, double xhat_right) {

    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(ObsInfo{}, 0, 0, 0, 0))>::value;

    int pt_start = panel_idx * a.nq;

    if (xhat_left == -1 && xhat_right == 1) {
        for (int k = 0; k < a.nq; k++) {
            int src_pt_idx = pt_start + k;
            double srcx = a.src_pts[src_pt_idx * 2 + 0];
            double srcy = a.src_pts[src_pt_idx * 2 + 1];
            double srcnx = a.src_normals[src_pt_idx * 2 + 0];
            double srcny = a.src_normals[src_pt_idx * 2 + 1];
            double srcmult = a.src_jacobians[src_pt_idx] * a.qw[k] *
                             a.src_param_width[panel_idx] * 0.5;

            auto kernel = kernel_fnc(obs, srcx, srcy, srcnx, srcny);

            // We don't want to store the integration result in the output
            // matrix yet because we might abort this integration and refine
            // another level at any point. So, we store the integration result
            // in a temporary variable.
            for (size_t d = 0; d < n_kernel_outputs; d++) {
                int entry = k * n_kernel_outputs + d;
                out[entry] += kernel[d] * srcmult;
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

            double srcmult = srcjac * a.qw[j] * a.src_param_width[panel_idx] * 0.5 *
                             (xhat_right - xhat_left) * 0.5;

            auto kernel = kernel_fnc(obs, srcx, srcy, srcnx, srcny);

            for (size_t d = 0; d < n_kernel_outputs; d++) {
                kernel[d] *= srcmult;
            }

            for (int k = 0; k < a.nq; k++) {
                double interp_K = a.interp_wts[k] * inv_denom / (qxj - a.qx[k]);
                for (size_t d = 0; d < n_kernel_outputs; d++) {
                    int entry = k * n_kernel_outputs + d;
                    out[entry] += kernel[d] * interp_K;
                }
            }
        }
    }
}

struct EstimatedIntegral {
    double xhat_left;
    double xhat_right;
    double max_err;
    std::vector<double> err;
    std::vector<double> value;
    std::vector<double> value_left;
    std::vector<double> value_right;
};

template <typename K>
std::pair<bool, int> adaptive_integrate(double* out, K kernel_fnc,
                                        const LocalQBXArgs& a, int panel_idx,
                                        const ObsInfo& obs, double tol) {
    constexpr int max_integrals = 200;

    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(ObsInfo{}, 0, 0, 0, 0))>::value;

    int Nv = a.nq * ndim;

    std::vector<double> integral(a.nq * ndim, 0.0);
    std::vector<double> error(a.nq * ndim, 0.0);
    double max_err;
    integrate_domain(integral.data(), kernel_fnc, a, panel_idx, obs, -1, 1);

    auto process_integral = [&](double xhat_left, double xhat_right,
                                const std::vector<double>& baseline) {
        EstimatedIntegral cur_integral{xhat_left, xhat_right};
        cur_integral.value_left.resize(Nv);
        cur_integral.value_right.resize(Nv);
        cur_integral.value = std::move(baseline);
        cur_integral.err.resize(Nv);

        double midpt = (xhat_right + xhat_left) * 0.5;
        integrate_domain(cur_integral.value_left.data(), kernel_fnc, a, panel_idx, obs,
                         xhat_left, midpt);
        integrate_domain(cur_integral.value_right.data(), kernel_fnc, a, panel_idx, obs,
                         midpt, xhat_right);

        cur_integral.max_err = 0;
        max_err = 0;
        for (int i = 0; i < Nv; i++) {
            double est2 = cur_integral.value_left[i] + cur_integral.value_right[i];
            double diff = est2 - baseline[i];
            double err = fabs(diff);
            cur_integral.err[i] = err;
            cur_integral.max_err = std::max(cur_integral.max_err, err);
            integral[i] += diff;
            error[i] += err;
            max_err = std::max(error[i], max_err);
        }
        return cur_integral;
    };

    std::vector<EstimatedIntegral> next_integral;
    auto heap_compare = [](auto& a, auto& b) { return a.max_err < b.max_err; };
    next_integral.push_back(process_integral(-1, 1, integral));

    int integral_idx = 0;
    double min_max_err = std::numeric_limits<double>::max();
    int divergences = 0;
    bool failed = false;
    for (; integral_idx < max_integrals; integral_idx++) {
        auto& cur_integral = next_integral.front();
        if (max_err > min_max_err) {
            divergences++;
        } else {
            divergences = 0;
        }
        min_max_err = std::min(min_max_err, max_err);

        for (int i = 0; i < Nv; i++) {
            error[i] -= cur_integral.err[i];
        }

        double midpt = (cur_integral.xhat_right + cur_integral.xhat_left) * 0.5;
        auto left_child =
            process_integral(cur_integral.xhat_left, midpt, cur_integral.value_left);
        auto right_child =
            process_integral(midpt, cur_integral.xhat_right, cur_integral.value_right);

        std::pop_heap(next_integral.begin(), next_integral.end(), heap_compare);
        next_integral.pop_back();

        next_integral.push_back(std::move(left_child));
        std::push_heap(next_integral.begin(), next_integral.end(), heap_compare);
        next_integral.push_back(std::move(right_child));
        std::push_heap(next_integral.begin(), next_integral.end(), heap_compare);

        if (max_err < tol) {
            break;
        }
    }

    // This is currently commented out because I'm not sure if this divergence
    // "failure" criterion makes any sense.
    //
    // Pro: integrals that grow in error as the adaptivity proceeds are much
    // more likely to be poorly behaved.
    //
    // Con: the criterion is too aggressive and is labeling some
    // correctly-computed integrals as failures
    //
    // if (divergences >= 2) {
    //     std::cout << "divergence fail!" << std::endl;
    //     failed = true;
    // }

    if (integral_idx == max_integrals) {
        double srcx = a.src_pts[panel_idx * a.nq + (a.nq / 2) * 2 + 0];
        double srcy = a.src_pts[panel_idx * a.nq + (a.nq / 2) * 2 + 1];
        // std::cout << "max fail! " << obs.x << " " << obs.y << " " << srcx << " " << srcy << " " << panel_idx << " " << integral_idx << std::endl;
        std::cout << "max err: " << max_err <<  "    tol: " << tol << std::endl;
        // for (int i = 0; i < next_integral.size(); i++) {
        //     std::cout << "option " << i << " " << next_integral[i].max_err << std::endl;
        // }
        if (max_err > tol * 10) {
            failed = true;
        }
    }

    for (int i = 0; i < Nv; i++) {
        out[i] += integral[i];
    }

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
        ObsInfo obs{a.obs_pts[obs_i * 2 + 0], a.obs_pts[obs_i * 2 + 1],
                    a.exp_centers[obs_i * 2 + 0], a.exp_centers[obs_i * 2 + 1],
                    a.exp_rs[obs_i]};

        bool converged = false;
        obs.p_start = 0;
        std::vector<double> integral(n_panels * a.nq * ndim, 0.0);

        int p_step = 5;
        bool failed = false;
        while (!converged and obs.p_start <= a.max_p) {
            obs.p_end = std::min(obs.p_start + p_step, a.max_p + 1);

            std::vector<double> temp_out(n_panels * a.nq * ndim * 2, 0.0);
            int n_subsets = 0;
            for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
                auto panel_idx = a.panels[panel_offset + panel_start];
                double* temp_out_ptr = &temp_out[panel_offset * a.nq * ndim * 2];
                auto result = adaptive_integrate(temp_out_ptr, kernel_fnc, a, panel_idx,
                                                 obs, coefficient_tol);
                failed = failed || result.first;
                n_subsets += result.second;
            }

            // Check series convergence.
            std::array<double, ndim> p_end_integral{};
            for (int pt_idx = 0; pt_idx < n_panels * a.nq; pt_idx++) {
                for (int d = 0; d < ndim; d++) {
                    int k = pt_idx * ndim + d;
                    double last_term = temp_out[2 * k + 1];
                    p_end_integral[d] += last_term;
                }
            }

            converged = true;
            for (int d = 0; d < ndim; d++) {
                p_end_integral[d] = fabs(p_end_integral[d]);
                if (p_end_integral[d] >= truncation_tol) {
                    converged = false;
                    break;
                }
            }

            // Add the integral
            for (int k = 0; k < integral.size(); k++) {
                double all_but_last_term = temp_out[2 * k];
                double last_term = temp_out[2 * k + 1];
                integral[k] += all_but_last_term + last_term;
            }

            obs.p_start = obs.p_end;
            a.n_subsets[obs_i] = n_subsets;
        }

        a.failed[obs_i] = failed;
        a.p[obs_i] = obs.p_end - 1;

        for (auto panel_offset = 0; panel_offset < n_panels; panel_offset++) {
            auto panel_idx = a.panels[panel_offset + panel_start];
            double* integral_ptr = &integral[panel_offset * a.nq * ndim];
            double* out_ptr = &a.mat[obs_i * a.n_src * ndim + panel_idx * a.nq * ndim];
            for (int k = 0; k < a.nq * ndim; k++) {
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
