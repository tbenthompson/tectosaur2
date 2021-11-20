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
                                          double srcy, double srcnx, double srcny, double* parameters) {
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
                                          double srcy, double srcnx, double srcny, double* parameters) {
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
                                                  double srcny, double* parameters) {
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
                                           double srcy, double srcnx, double srcny, double* parameters) {
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

inline std::array<double, 4> elastic_U(double obsx, double obsy, double srcx,
                                       double srcy, double srcnx, double srcny, double* parameters) {
    double poisson_ratio = parameters[0];
    double disp_C1 = 1.0 / (8 * M_PI * (1 - poisson_ratio));
    double disp_C2 = 3 - 4 * poisson_ratio;

    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double G = -0.5 * disp_C2 * log(r2);
    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
        G = 0.0;
    }

    std::array<double, 4> out{};
    out[0] = disp_C1 * (G + dx * dx * invr2);
    out[1] = disp_C1 * (dx * dy * invr2);
    out[2] = out[1];
    out[3] = disp_C1 * (G + dy * dy * invr2);
    return out;
}

inline std::array<double, 4> elastic_T(double obsx, double obsy, double srcx,
                                       double srcy, double srcnx, double srcny, double* parameters) {
    double poisson_ratio = parameters[0];
    double trac_C1 = 1.0 / (4 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    double invr = sqrt(invr2);
    if (r2 <= too_close) {
        invr2 = 0.0;
        invr = 0.0;
    }

    double drdn = (dx * srcnx + dy * srcny) * invr;
    double nCd = srcny * dx - srcnx * dy;

    std::array<double, 4> out{};
    out[0] = -trac_C1 * invr * (trac_C2 + 2 * dx * dx * invr2) * drdn;
    out[1] = -trac_C1 * invr * (
        2 * dx * dy * invr2 * drdn -
        trac_C2 * nCd * invr
    );
    out[2] = -trac_C1 * invr * (
        2 * dx * dy * invr2 * drdn -
        trac_C2 * (-nCd) * invr
    );
    out[3] = -trac_C1 * invr * (trac_C2 + 2 * dy * dy * invr2) * drdn;

    return out;
}

inline std::array<double, 6> elastic_A(double obsx, double obsy, double srcx,
                                       double srcy, double srcnx, double srcny, double* parameters) {
    double poisson_ratio = parameters[0];
    double trac_C1 = 1.0 / (4 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    double invr = sqrt(invr2);
    if (r2 <= too_close) {
        invr2 = 0.0;
        invr = 0.0;
    }

    // For relating with formulae that specify the kernel with the dot product
    // with traction already done...
    std::array<double, 6> out{};
    // idx 0 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 0
    out[0] = trac_C1 * invr * (
        (trac_C2 + 2 * dx * dx * invr2) * dx * invr
    );
    // idx 1 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 1
    out[1] = trac_C1 * invr * (
        2 * dx * dy * invr2 * dx * invr -
        trac_C2 * invr * (-dy)
    );
    // idx 2 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 0
    out[2] = trac_C1 * invr * (
        2 * dy * dx * invr2 * dy * invr -
        trac_C2 * invr * (-dx)
    );
    // idx 3 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 1
    out[3] = trac_C1 * invr * (
        (trac_C2 + 2 * dy * dy * invr2) * dy * invr
    );
    // idx 4 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 0
    out[4] = trac_C1 * invr * (
        2 * dy * dx * invr2 * dx * invr -
        trac_C2 * invr * dy
    );
    // idx 5 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 1
    out[5] = trac_C1 * invr * (
        (trac_C2 + 2 * dy * dy * invr2) * dx * invr
    );

    return out;
}

inline std::array<double, 6> elastic_H(double obsx, double obsy, double srcx,
                                       double srcy, double srcnx, double srcny, double* parameters) {
    double poisson_ratio = parameters[0];
    double HC = 1.0 / (2 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    double invr = sqrt(invr2);
    if (r2 <= too_close) {
        invr2 = 0.0;
        invr = 0.0;
    }
    double rx = dx * invr;
    double ry = dy * invr;

    double drdn = (dx * srcnx + dy * srcny) * invr;

    // For relating with formulae that specify the kernel with the dot product
    // with traction already done...
    std::array<double, 6> out{};
    // idx 0 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 0
    out[0] = (HC * invr2) * (
        (2 * drdn * (trac_C2 * rx + poisson_ratio * (rx + rx) - 4 * rx * rx * rx))
        + trac_C2 * (2 * srcnx * rx * rx + srcnx + srcnx)
        + (2 * poisson_ratio * (srcnx * rx * rx + srcnx * rx * rx))
        - (1 - 4 * poisson_ratio) * srcnx
    );
    // idx 1 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 1
    out[1] = (HC * invr2) * (
        (2 * drdn * (trac_C2 * ry - 4 * rx * ry * rx))
        + trac_C2 * (2 * srcny * rx * rx)
        + (2 * poisson_ratio * (srcnx * ry * rx + srcnx * rx * ry))
        - (1 - 4 * poisson_ratio) * srcny
    );
    // idx 2 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 0
    out[2] = (HC * invr2) * (
        (2 * drdn * (trac_C2 * rx - 4 * ry * rx * ry))
        + trac_C2 * (2 * srcnx * ry * ry)
        + (2 * poisson_ratio * (srcny * rx * ry + srcny * ry * rx))
        - (1 - 4 * poisson_ratio) * srcnx
    );
    // idx 3 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 1
    out[3] = (HC * invr2) * (
        (2 * drdn * (trac_C2 * ry + poisson_ratio * (ry + ry) - 4 * ry * ry * ry))
        + trac_C2 * (2 * srcny * ry * ry + srcny + srcny)
        + (2 * poisson_ratio * (srcny * ry * ry + srcny * ry * ry))
        - (1 - 4 * poisson_ratio) * srcny
    );
    // idx 4 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 0
    out[4] = (HC * invr2) * (
        (2 * drdn * (poisson_ratio * ry - 4 * ry * rx * rx))
        + trac_C2 * (2 * srcnx * ry * rx + srcny)
        + (2 * poisson_ratio * (srcny * rx * rx + srcnx * ry * rx))
    );
    // idx 5 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 1
    out[5] = (HC * invr2) * (
        (2 * drdn * (poisson_ratio * rx - 4 * ry * ry * rx))
        + trac_C2 * (2 * srcny * ry * rx + srcnx)
        + (2 * poisson_ratio * (srcny * ry * rx + srcnx * ry * ry))
    );

    return out;
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
    double* kernel_parameters;
};

template <typename K>
void integrate_domain(double* out, K kernel_fnc, const NearfieldArgs& a, int panel_idx,
                      double obsx, double obsy, double xhat_left, double xhat_right) {
    constexpr size_t ndim =
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0, a.kernel_parameters))>::value;

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

            auto kernel = kernel_fnc(obsx, obsy, srcx, srcy, srcnx, srcny, a.kernel_parameters);

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

            auto kernel = kernel_fnc(obsx, obsy, srcx, srcy, srcnx, srcny, a.kernel_parameters);
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
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0, a.kernel_parameters))>::value;
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
        std::tuple_size<decltype(kernel_fnc(0, 0, 0, 0, 0, 0, a.kernel_parameters))>::value;

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

void nearfield_elastic_U(const NearfieldArgs& a) {
    _nearfield_integrals(elastic_U, a);
}
void nearfield_elastic_T(const NearfieldArgs& a) {
    _nearfield_integrals(elastic_T, a);
}

void nearfield_elastic_A(const NearfieldArgs& a) {
    _nearfield_integrals(elastic_A, a);
}

void nearfield_elastic_H(const NearfieldArgs& a) {
    _nearfield_integrals(elastic_H, a);
}
