#pragma once
#include <iostream>
#include <queue>
#include <vector>

struct SourceData {
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
};

template <typename K, typename T>
void integrate_domain(double* out, const T& obs, const K& kernel_fnc,
                      const SourceData& a, int panel_idx, double xhat_left,
                      double xhat_right) {
    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(T{}, 0, 0, 0, 0))>::value;

    int pt_start = panel_idx * a.n_interp;

    for (int j = 0; j < a.n_kronrod; j++) {
        double qxj = xhat_left + (a.kronrod_qx[j] + 1) * 0.5 * (xhat_right - xhat_left);

        // Step 1: check for matching
        // If the quadrature point exactly matches one of the interpolation points,
        // then we can simplify and just use the values for that interpolation pt.
        int matching_k = -1;
        for (int k = 0; k < a.n_interp; k++) {
            if (fabs(qxj - a.interp_qx[k]) < 2e-16) {
                matching_k = k;
            }
        }

        // Step 2: simplified integral if matching
        if (matching_k != -1) {
            int src_pt_idx = matching_k + pt_start;

            double srcx = a.src_pts[src_pt_idx * 2 + 0];
            double srcy = a.src_pts[src_pt_idx * 2 + 1];
            double srcnx = a.src_normals[src_pt_idx * 2 + 0];
            double srcny = a.src_normals[src_pt_idx * 2 + 1];
            double srcjac = a.src_jacobians[src_pt_idx];
            auto kernel = kernel_fnc(obs, srcx, srcy, srcnx, srcny);
            double srcmult =
                srcjac * a.src_param_width[panel_idx] * (xhat_right - xhat_left) * 0.25;

            for (size_t d = 0; d < n_kernel_outputs; d++) {
                int entry = matching_k * n_kernel_outputs * 2 + d * 2;

                double value = kernel[d] * srcmult;
                double kronrod_value = value * a.kronrod_qw[j];
                out[entry] += kronrod_value;
                double gauss_value = 0;

                // The baseline Gauss rule uses only the odd half of the
                // function evaluations.
                if (j % 2 == 1) {
                    gauss_value = value * a.kronrod_qw_gauss[j / 2];
                }
                // difference estimate from the nested Gauss-Kronrod quadrature rule.
                // note that this is a *difference* and not a error. The absolute value
                // will ned to be taken later.
                out[entry + 1] += kronrod_value - gauss_value;
            }
            continue;
        }

        // Step 3: interpolate the source values.
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

        // Step 4: call the kernel and integrate with the interpolated source
        // location.
        auto kernel = kernel_fnc(obs, srcx, srcy, srcnx, srcny);
        double srcmult =
            srcjac * a.src_param_width[panel_idx] * (xhat_right - xhat_left) * 0.25;
        for (size_t d = 0; d < n_kernel_outputs; d++) {
            kernel[d] *= srcmult;
        }

        // Step 5: interpolate the results back onto the source basis to compute
        // matrix coefficients.
        for (int k = 0; k < a.n_interp; k++) {
            double interp_K = a.interp_wts[k] * inv_denom / (qxj - a.interp_qx[k]);
            for (size_t d = 0; d < n_kernel_outputs; d++) {
                int entry = k * n_kernel_outputs * 2 + d * 2;

                double value = kernel[d] * interp_K;
                double kronrod_value = value * a.kronrod_qw[j];
                out[entry] += kronrod_value;
                double gauss_value = 0;

                // The baseline Gauss rule uses only the odd half of the
                // function evaluations.
                if (j % 2 == 1) {
                    gauss_value = value * a.kronrod_qw_gauss[j / 2];
                }
                // difference estimate from the nested Gauss-Kronrod quadrature rule.
                // note that this is a *difference* and not a error. The absolute value
                // will ned to be taken later.
                out[entry + 1] += kronrod_value - gauss_value;
            }
        }
    }
}

struct EstimatedIntegral {
    double xhat_left;
    double xhat_right;
    double max_err;
    double* value_ptr;
};

constexpr int max_adaptive_integrals = 100;

template <typename K, typename T>
std::pair<double, int>
adaptive_integrate(double* out, const T& obs, const K& kernel_fnc, const SourceData& a,
                   int panel_idx, double tol, double* memory_pool) {

    constexpr size_t n_kernel_outputs =
        std::tuple_size<decltype(kernel_fnc(T{}, 0, 0, 0, 0))>::value;

    int Nv = a.n_interp * n_kernel_outputs;

    // We store twice as many values here because we're storing both an integral
    // and an error estimate.
    double* integral_ptr = &memory_pool[0];
    for (int i = 0; i < Nv * 2; i++) {
        integral_ptr[i] = 0;
    }
    double max_err = 0;

    integrate_domain(integral_ptr, obs, kernel_fnc, a, panel_idx, -1, 1);
    for (int i = 0; i < Nv; i++) {
        integral_ptr[2 * i + 1] = fabs(integral_ptr[2 * i + 1]);
        max_err = std::max(integral_ptr[2 * i + 1], max_err);
    }

    auto cmp = [](auto& x, auto& y) { return x.max_err < y.max_err; };
    std::priority_queue<EstimatedIntegral, std::vector<EstimatedIntegral>,
                        decltype(cmp)>
        next_integrals(cmp);
    next_integrals.emplace(EstimatedIntegral{-1, 1, max_err, integral_ptr});

    int integral_idx = 1;
    for (; integral_idx < max_adaptive_integrals; integral_idx++) {
        if (max_err < tol) {
            break;
        }

        // OPTIMIZATION POTENTIAL: The memory used for cur_integral could be
        // recycled during the next iteration of the loop.
        auto& cur_integral = next_integrals.top();

        double midpt = (cur_integral.xhat_right + cur_integral.xhat_left) * 0.5;
        EstimatedIntegral left_child{cur_integral.xhat_left, midpt, 0,
                                     &memory_pool[Nv * 2 * (1 + integral_idx * 2)]};
        for (int i = 0; i < Nv * 2; i++) {
            left_child.value_ptr[i] = 0;
        }
        integrate_domain(left_child.value_ptr, obs, kernel_fnc, a, panel_idx,
                         cur_integral.xhat_left, midpt);

        EstimatedIntegral right_child{midpt, cur_integral.xhat_right, 0,
                                      &memory_pool[Nv * 2 * (2 + integral_idx * 2)]};
        for (int i = 0; i < Nv * 2; i++) {
            right_child.value_ptr[i] = 0;
        }
        integrate_domain(right_child.value_ptr, obs, kernel_fnc, a, panel_idx, midpt,
                         cur_integral.xhat_right);


        // Update the integral and its corresponding error estimate.
        max_err = 0;
        for (int i = 0; i < Nv; i++) {
            right_child.value_ptr[2 * i + 1] = fabs(right_child.value_ptr[2 * i + 1]);
            left_child.value_ptr[2 * i + 1] = fabs(left_child.value_ptr[2 * i + 1]);
            auto right_err = right_child.value_ptr[2 * i + 1];
            auto left_err = left_child.value_ptr[2 * i + 1];
            integral_ptr[2 * i] +=
                (-cur_integral.value_ptr[2 * i] + left_child.value_ptr[2 * i] +
                 right_child.value_ptr[2 * i]);
            integral_ptr[2 * i + 1] +=
                -cur_integral.value_ptr[2 * i + 1] + left_err + right_err;
            left_child.max_err = std::max(left_child.max_err, left_err);
            right_child.max_err = std::max(right_child.max_err, right_err);
            max_err = std::max(integral_ptr[2 * i + 1], max_err);
            // if (i == 0) {
            //     std::cout.precision(17);
            //     std::cout << integral_idx << " " << integral_ptr[2*i] << " " << integral_ptr[2*i+1] << " " << left_child.value_ptr[2 * i] +
            //      right_child.value_ptr[2 * i] << " " <<
            //     cur_integral.value_ptr[2 * i + 1] << " " <<
            //     -cur_integral.value_ptr[2 * i] + left_child.value_ptr[2 * i] +
            //                     right_child.value_ptr[2 * i]
            //     <<  std::endl;
            //     std::cout << left_child.value_ptr[2 * i] << " " << right_child.value_ptr[2 * i] << std::endl;
            // }
        }

        if (max_err < tol) {
            break;
        }

        // Update heap by removing the top entry that we just processed and
        // adding the two new children.
        next_integrals.pop();
        next_integrals.push(std::move(left_child));
        next_integrals.push(std::move(right_child));
    }

    for (int i = 0; i < Nv; i++) {
        out[i] += integral_ptr[2 * i];
    }

    return std::make_pair(max_err, integral_idx);
}
