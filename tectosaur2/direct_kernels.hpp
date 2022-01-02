#include <array>

struct DirectObsInfo {
    double x;
    double y;
    double* parameters;
};

constexpr double C = 1.0 / (2 * M_PI);
constexpr double C2 = 1.0 / (4 * M_PI);
constexpr double too_close = 1e-16;

inline std::array<double, 1> single_layer(const DirectObsInfo& obs, double srcx,
                                          double srcy, double srcnx, double srcny) {
    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
    double r2 = dx * dx + dy * dy;

    double G = -C2 * log(r2);
    if (r2 <= too_close) {
        G = 0.0;
    }
    return {G};
}

inline std::array<double, 1> double_layer(const DirectObsInfo& obs, double srcx,
                                          double srcy, double srcnx, double srcny) {
    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }

    return {-C * (dx * srcnx + dy * srcny) * invr2};
}

inline std::array<double, 2> adjoint_double_layer(const DirectObsInfo& obs,
                                                  double srcx, double srcy,
                                                  double srcnx, double srcny) {
    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }
    double F = -C * invr2;

    return {F * dx, F * dy};
}

constexpr double HC = -1.0 / (2 * M_PI);
inline std::array<double, 2> hypersingular(const DirectObsInfo& obs, double srcx,
                                           double srcy, double srcnx, double srcny) {
    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
    double r2 = dx * dx + dy * dy;

    double invr2 = 1.0 / r2;
    if (r2 <= too_close) {
        invr2 = 0.0;
    }

    double A = 2 * (dx * srcnx + dy * srcny) * invr2;
    double B = HC * invr2;
    return {B * (srcnx - A * dx), B * (srcny - A * dy)};
}

inline std::array<double, 4> elastic_U(const DirectObsInfo& obs, double srcx,
                                       double srcy, double srcnx, double srcny) {
    double poisson_ratio = obs.parameters[0];
    double disp_C1 = 1.0 / (8 * M_PI * (1 - poisson_ratio));
    double disp_C2 = 3 - 4 * poisson_ratio;

    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
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

inline std::array<double, 4> elastic_T(const DirectObsInfo& obs, double srcx,
                                       double srcy, double srcnx, double srcny) {
    double poisson_ratio = obs.parameters[0];
    double trac_C1 = 1.0 / (4 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
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
    out[1] = -trac_C1 * invr * (2 * dx * dy * invr2 * drdn - trac_C2 * nCd * invr);
    out[2] = -trac_C1 * invr * (2 * dx * dy * invr2 * drdn - trac_C2 * (-nCd) * invr);
    out[3] = -trac_C1 * invr * (trac_C2 + 2 * dy * dy * invr2) * drdn;

    return out;
}

inline std::array<double, 6> elastic_A(const DirectObsInfo& obs, double srcx,
                                       double srcy, double srcnx, double srcny) {
    double poisson_ratio = obs.parameters[0];
    double trac_C1 = -1.0 / (4 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
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
    out[0] = trac_C1 * invr * (trac_C2 * dx * invr + 2 * dx * dx * invr2 * dx * invr);
    // idx 1 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 1
    out[1] =
        trac_C1 * invr * (2 * dx * dy * invr2 * dx * invr - trac_C2 * invr * dy);
    // idx 2 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 0
    out[2] =
        trac_C1 * invr * (2 * dy * dx * invr2 * dy * invr - trac_C2 * invr * dx);
    // idx 3 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 1
    out[3] = trac_C1 * invr * (trac_C2 * dy * invr + 2 * dy * dy * invr2 * dy * invr);
    // idx 4 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 0
    out[4] = trac_C1 * invr * (2 * dy * dx * invr2 * dx * invr + trac_C2 * invr * dy);
    // idx 5 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 1
    out[5] = trac_C1 * invr * (trac_C2 * dx * invr + 2 * dy * dy * invr2 * dx * invr);

    return out;
}

inline std::array<double, 6> elastic_H(const DirectObsInfo& obs, double srcx,
                                       double srcy, double srcnx, double srcny) {
    double poisson_ratio = obs.parameters[0];
    double HC = -1.0 / (2 * M_PI * (1 - poisson_ratio));
    double trac_C2 = 1 - 2.0 * poisson_ratio;

    double dx = obs.x - srcx;
    double dy = obs.y - srcy;
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
    out[0] =
        (HC * invr2) *
        ((2 * drdn * (trac_C2 * rx + poisson_ratio * (rx + rx) - 4 * rx * rx * rx)) +
         trac_C2 * (2 * srcnx * rx * rx + srcnx + srcnx) +
         (2 * poisson_ratio * (srcnx * rx * rx + srcnx * rx * rx)) -
         (1 - 4 * poisson_ratio) * srcnx);
    // idx 1 = s_xx from t_x --> t_x from n = (1, 0), nd = 0, d_obs = 0, d_src = 1
    out[1] = (HC * invr2) * ((2 * drdn * (trac_C2 * ry - 4 * rx * ry * rx)) +
                             trac_C2 * (2 * srcny * rx * rx) +
                             (2 * poisson_ratio * (srcnx * ry * rx + srcnx * rx * ry)) -
                             (1 - 4 * poisson_ratio) * srcny);
    // idx 2 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 0
    out[2] = (HC * invr2) * ((2 * drdn * (trac_C2 * rx - 4 * ry * rx * ry)) +
                             trac_C2 * (2 * srcnx * ry * ry) +
                             (2 * poisson_ratio * (srcny * rx * ry + srcny * ry * rx)) -
                             (1 - 4 * poisson_ratio) * srcnx);
    // idx 3 = s_yy from t_x --> t_y from n = (0, 1), nd = 1, d_obs = 1, d_src = 1
    out[3] =
        (HC * invr2) *
        ((2 * drdn * (trac_C2 * ry + poisson_ratio * (ry + ry) - 4 * ry * ry * ry)) +
         trac_C2 * (2 * srcny * ry * ry + srcny + srcny) +
         (2 * poisson_ratio * (srcny * ry * ry + srcny * ry * ry)) -
         (1 - 4 * poisson_ratio) * srcny);
    // idx 4 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 0
    out[4] = (HC * invr2) * ((2 * drdn * (poisson_ratio * ry - 4 * ry * rx * rx)) +
                             trac_C2 * (2 * srcnx * ry * rx + srcny) +
                             (2 * poisson_ratio * (srcny * rx * rx + srcnx * ry * rx)));
    // idx 5 = s_xy from t_x --> t_y from n = (1, 0), nd = 0, d_obs = 1, d_src = 1
    out[5] = (HC * invr2) * ((2 * drdn * (poisson_ratio * rx - 4 * ry * ry * rx)) +
                             trac_C2 * (2 * srcny * ry * rx + srcnx) +
                             (2 * poisson_ratio * (srcny * ry * rx + srcnx * ry * ry)));

    return out;
}
