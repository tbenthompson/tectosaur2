#include <array>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

constexpr double C = 1.0 / (2 * M_PI);
constexpr double C2 = 1.0 / (4 * M_PI);

inline std::array<double,1> single_layer(
    double obsx, double obsy,
    double srcx, double srcy, double srcnx, double srcny)
{
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx*dx + dy*dy;

    double G = C2 * log(r2);
    if (r2 == 0.0) {
        G = 0.0;
    }
    return {G};
}

inline std::array<double,1> double_layer(
    double obsx, double obsy,
    double srcx, double srcy, double srcnx, double srcny)
{
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx*dx + dy*dy;

    double invr2 = 1.0 / r2;
    if (r2 == 0.0) {
        invr2 = 0.0;
    }

    return {-C * (dx * srcnx + dy * srcny) * invr2};
}

inline std::array<double,2> adjoint_double_layer(
    double obsx, double obsy,
    double srcx, double srcy, double srcnx, double srcny)
{
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx*dx + dy*dy;

    double invr2 = 1.0 / r2;
    if (r2 == 0.0) {
        invr2 = 0.0;
    }
    double F = -C * invr2;

    return {F * dx, F * dy};
}

inline std::array<double,2> hypersingular(
    double obsx, double obsy,
    double srcx, double srcy, double srcnx, double srcny)
{
    double dx = obsx - srcx;
    double dy = obsy - srcy;
    double r2 = dx*dx + dy*dy;

    double invr2 = 1.0 / r2;
    if (r2 == 0.0) {
        invr2 = 0.0;
    }

    double A = 2 * (dx * srcnx + dy * srcny) * invr2;
    double B = C * invr2;
    return {B * (srcnx - A * dx), B * (srcny - A * dy)};
}

struct NearfieldArgs {
    double* mat; int n_obs; int n_src; double* obs_pts;
    double* src_pts; double* src_normals; double* src_quad_wt_jac;
    int src_panel_order; long* panels; long* panel_starts; long* refinement_map;
    double mult;
};

template <typename K>
void _nearfield_integrals(K kernel_fnc, const NearfieldArgs& a) {
    int n_src_panels = a.n_src / a.src_panel_order;

    for (int i = 0; i < a.n_obs; i++){
        long panel_start = a.panel_starts[i];
        long panel_end = a.panel_starts[i + 1];
        long n_panels = panel_end - panel_start;

        if (n_panels == 0) {
            continue;
        }

        double obsx = a.obs_pts[i * 2 + 0];
        double obsy = a.obs_pts[i * 2 + 1];

        auto panel_idx = std::lower_bound(
            a.refinement_map, &a.refinement_map[n_src_panels], a.panels[panel_start]
        ) - a.refinement_map;
        while (panel_idx < n_src_panels && a.refinement_map[panel_idx] < panel_end) {
            int pt_start = a.panels[panel_idx] * a.src_panel_order;
            int pt_end = (a.panels[panel_idx] + 1) * a.src_panel_order;
            for (int src_pt_idx = pt_start; src_pt_idx < pt_end; src_pt_idx++) {
                double srcx = a.src_pts[src_pt_idx * 2 + 0];
                double srcy = a.src_pts[src_pt_idx * 2 + 1];
                double srcnx = a.src_normals[src_pt_idx * 2 + 0];
                double srcny = a.src_normals[src_pt_idx * 2 + 1];
                double srcmult = a.mult * a.src_quad_wt_jac[src_pt_idx];

                auto kernel = kernel_fnc(obsx, obsy, srcx, srcy, srcnx, srcny);

                size_t ndim = kernel.size();
                for (size_t dim = 0; dim < ndim; dim++) {
                    double I = kernel[dim] * srcmult;
                    a.mat[i * a.n_src * ndim + src_pt_idx * ndim + dim] += I;
                }
            }

            panel_idx += 1;
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
