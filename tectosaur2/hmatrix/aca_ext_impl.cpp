#include <algorithm>
#include <cstdio>
#include <math.h>

#include "../direct_kernels.hpp"

#define Real double

struct ACAArgs {

    // out parameters here
    Real* buffer;
    int* uv_ptrs;
    int* n_terms;

    // mutable workspace parameters
    int* next_buffer_ptr;
    Real* fworkspace;
    int* iworkspace;

    // immutable parameters below here
    /*
    There are three relevant dimensions to handle a wide range of kernels:
    - row_dim and col_dim represent the tensor dimensions (row_dim x col_dim)
      for the input and output of the kernel.
    - space_dim represents the underlying dimension of the space of the
      points/entities. Typically, this is 2D or 3D.

    For example, the displacement -> stress kernel in 3D elasticity has:
    - row_dim = 6, col_dim = 3, space_dim = 3
    - The row_dim is 6 because the stress tensor output has 6 independent
      components (sxx, syy, szz, sxy, sxz, syz)
    - The col_dim is 3 because the displacement input has 3 components (x, y, z)
    - The space_dim is 3 because the observation and source points are 3D.

    And, for the 2D hypersingular kernel for the Laplace equation:
    - row_dim = 2, col_dim = 1, space_dim = 2
    - The row_dim is 2 because the potential gradient has two components
      representing the x and y derivatives of the potential.
    */
    int row_dim;
    int col_dim;

    // The number of blocks to approximate.
    int n_blocks;

    // The index of the obs and src point start and end for each block. All
    // blocks are contiguous in obs/src idx because the indices have been
    // re-arranged during the tree construction phase.
    long* obs_start;
    long* obs_end;
    long* src_start;
    long* src_end;

    // uv_ptrs holds a set of indices into the buffer array that point to the
    // location of each row/col in the output U/V matrix. A sufficiently large
    // array of integers has been allocated in the calling code. Here,
    // uv_ptrs_starts indicates the first index in the uv_ptrs array to use for
    // each block.
    int* uv_ptrs_starts;

    // Which chunk of the float workspace to use for each block? This might be
    // removable.
    int* fworkspace_starts;

    // For each block, Iref0 and Jref0 provide the index of the starting
    // reference row/col. This was added historically because generating random
    // numbers in CUDA is harder than doing it in the calling code.
    int* Iref0;
    int* Jref0;

    // Data on the observation points and the source mesh.
    Real* obs_pts;
    Real* src_pts;
    Real* src_normals;
    Real* src_weights;

    Real* tol;
    int* max_iter;
    Real* kernel_parameters;
    bool verbose;
};

struct MatrixIndex {
    int row;
    int col;
};

int buffer_alloc(int* next_ptr, int n_values) {
    int out;
#pragma omp critical
    {
        out = *next_ptr;
        *next_ptr += n_values;
    }
    return out;
}

bool in(int target, int* arr, int n_arr) {
    // Could be faster by keeping arr sorted and doing binary search.
    // but that is probably premature optimization.
    for (int i = 0; i < n_arr; i++) {
        if (target == arr[i]) {
            return true;
        }
    }
    return false;
}

struct MatrixIndex argmax_abs_not_in_list(Real* data, int n_data_rows, int n_data_cols,
                                          int* prev, int n_prev, bool rows_or_cols) {
    struct MatrixIndex max_idx;
    Real max_val = -1;
    for (int i = 0; i < n_data_rows; i++) {
        for (int j = 0; j < n_data_cols; j++) {
            Real v = fabs(data[i * n_data_cols + j]);
            int relevant_idx;
            if (rows_or_cols) {
                relevant_idx = i;
            } else {
                relevant_idx = j;
            }
            if (v > max_val && !in(relevant_idx, prev, n_prev)) {
                max_idx.row = i;
                max_idx.col = j;
                max_val = v;
            }
        }
    }
    return max_idx;
}

void sub_residual(Real* output, const ACAArgs& a, int n_rows, int n_cols,
                  int rowcol_start, int rowcol_end, int n_terms, bool rows_or_cols,
                  int uv_ptr0) {

    for (int sr_idx = 0; sr_idx < n_terms; sr_idx++) {
        int buffer_ptr = a.uv_ptrs[uv_ptr0 + sr_idx];

        Real* U_term = &a.buffer[buffer_ptr];
        Real* V_term = &a.buffer[buffer_ptr + n_rows];
        int n_rowcol = rowcol_end - rowcol_start;

        if (rows_or_cols) {
            for (int i = 0; i < n_rowcol; i++) {
                Real uv = U_term[i + rowcol_start];
                for (int j = 0; j < n_cols; j++) {
                    Real vv = V_term[j];
                    output[i * n_cols + j] -= uv * vv;
                }
            }
        } else {
            for (int i = 0; i < n_rows; i++) {
                Real uv = U_term[i];
                for (int j = 0; j < n_rowcol; j++) {
                    Real vv = V_term[j + rowcol_start];
                    output[i * n_rowcol + j] -= uv * vv;
                }
            }
        }
    }
}

template <typename K>
void calc(const K& kf, Real* output, const ACAArgs& a, int ss, int se, int os, int oe,
          int rowcol_start, int rowcol_end, bool row_or_col) {

    int i_start;
    int i_end;
    int j_start;
    int j_end;
    int obs_dim_start;
    int obs_dim_end;
    int src_dim_start;
    int src_dim_end;
    if (row_or_col) {
        int offset = floor(((float)rowcol_start) / a.row_dim);
        i_start = os + offset;
        i_end = i_start + 1;
        j_start = ss;
        j_end = se;
        obs_dim_start = rowcol_start - offset * a.row_dim;
        obs_dim_end = rowcol_end - offset * a.row_dim;
        src_dim_start = 0;
        src_dim_end = a.col_dim;
    } else {
        int offset = floor(((float)rowcol_start) / a.col_dim);
        i_start = os;
        i_end = oe;
        j_start = ss + offset;
        j_end = j_start + 1;
        obs_dim_start = 0;
        obs_dim_end = a.row_dim;
        src_dim_start = rowcol_start - offset * a.col_dim;
        src_dim_end = rowcol_end - offset * a.col_dim;
    }

    if (a.verbose) {
        printf("calc i_start=%i i_end=%i j_start=%i j_end=%i obs_dim_start=%i "
            "obs_dim_end=%i src_dim_start=%i src_dim_end=%i\n",
            i_start, i_end, j_start, j_end, obs_dim_start, obs_dim_end, src_dim_start,
            src_dim_end);
    }

    int n_output_src = j_end - j_start;
    for (int i = i_start; i < i_end; i++) {
        int obs_idx = i - i_start;
        DirectObsInfo obs{a.obs_pts[i * 2 + 0], a.obs_pts[i * 2 + 1],
                            a.kernel_parameters};

        for (int j = j_start; j < j_end; j++) {
            int src_idx = j - j_start;

            Real srcx = a.src_pts[j * 2 + 0];
            Real srcy = a.src_pts[j * 2 + 1];
            Real srcnx = a.src_normals[j * 2 + 0];
            Real srcny = a.src_normals[j * 2 + 1];
            Real srcwt = a.src_weights[j];

            auto kernel = kf(obs, srcx, srcy, srcnx, srcny);

            for (int d_obs = obs_dim_start; d_obs < obs_dim_end; d_obs++) {
                for (int d_src = src_dim_start; d_src < src_dim_end; d_src++) {
                    int idx = ((obs_idx * a.row_dim + (d_obs - obs_dim_start)) *
                                   n_output_src +
                               src_idx) *
                                  (src_dim_end - src_dim_start) +
                              (d_src - src_dim_start);
                    output[idx] = kernel[d_obs * a.col_dim + d_src] * srcwt;
                }
            }
        }
    }
}

template <typename K> void _aca_integrals(K kf, const ACAArgs& a) {

#pragma omp parallel for
    for (long block_idx = 0; block_idx < a.n_blocks; block_idx++) {
        long os = a.obs_start[block_idx];
        long oe = a.obs_end[block_idx];
        long ss = a.src_start[block_idx];
        long se = a.src_end[block_idx];
        long n_obs = oe - os;
        long n_src = se - ss;
        long n_rows = n_obs * a.row_dim;
        long n_cols = n_src * a.col_dim;

        int uv_ptr0 = a.uv_ptrs_starts[block_idx];
        int* block_iworkspace = &a.iworkspace[uv_ptr0];

        int* prevIstar = block_iworkspace;
        int* prevJstar = &block_iworkspace[std::min(n_cols, n_rows) / 2];

        Real* block_fworkspace = &a.fworkspace[a.fworkspace_starts[block_idx]];
        Real* RIstar = block_fworkspace;
        Real* RJstar = &block_fworkspace[n_cols];

        Real* RIref = &block_fworkspace[n_cols + n_rows];
        Real* RJref = &block_fworkspace[n_cols + n_rows + a.row_dim * n_cols];

        int Iref = a.Iref0[block_idx];
        Iref -= Iref % a.row_dim;
        int Jref = a.Jref0[block_idx];
        Jref -= Jref % a.col_dim;

        calc(kf, RIref, a, ss, se, os, oe, Iref, Iref + a.row_dim, true);
        calc(kf, RJref, a, ss, se, os, oe, Jref, Jref + a.col_dim, false);

        // TODO: this is bad because it limits the number of vectors to half of
        // what might be needed
        int max_iter =
            std::min(a.max_iter[block_idx], (int)std::min(n_rows / 2, n_cols / 2));
        Real tol = a.tol[block_idx];
        if (a.verbose) {
            printf("(row_dim, col_dim) = (%i, %i)\n", a.row_dim, a.col_dim);
            printf("max_iter = %i\n", max_iter);
            printf("tol = %i\n", max_iter);
        }

        Real frob_est = 0;
        int k = 0;
        for (; k < max_iter; k++) {
            if (a.verbose) {
                printf("\n\nstart iteration %i with Iref=%i Jref=%i\n", k, Iref, Jref);
                for (int i = 0; i < 5; i++) {
                    printf("RIref[%i] = %f\n", i, RIref[i]);
                }
                for (int j = 0; j < 5; j++) {
                    printf("RJref[%i] = %f\n", j, RJref[j]);
                }
            }

            MatrixIndex Istar_entry =
                argmax_abs_not_in_list(RJref, n_rows, a.col_dim, prevIstar, k, true);
            MatrixIndex Jstar_entry =
                argmax_abs_not_in_list(RIref, a.row_dim, n_cols, prevJstar, k, false);
            int Istar = Istar_entry.row;
            int Jstar = Jstar_entry.col;

            Real Istar_val = fabs(RJref[Istar_entry.row * a.col_dim + Istar_entry.col]);
            Real Jstar_val = fabs(RIref[Jstar_entry.row * n_cols + Jstar_entry.col]);

            if (a.verbose) {
                printf("pivot guess %i %i %e %e \n", Istar, Jstar, Istar_val,
                       Jstar_val);
            }
            if (Istar_val > Jstar_val) {
                calc(kf, RIstar, a, ss, se, os, oe, Istar, Istar + 1, true);
                sub_residual(RIstar, a, n_rows, n_cols, Istar, Istar + 1, k, true,
                             uv_ptr0);

                Jstar_entry =
                    argmax_abs_not_in_list(RIstar, 1, n_cols, prevJstar, k, false);
                Jstar = Jstar_entry.col;

                calc(kf, RJstar, a, ss, se, os, oe, Jstar, Jstar + 1, false);
                sub_residual(RJstar, a, n_rows, n_cols, Jstar, Jstar + 1, k, false,
                             uv_ptr0);
            } else {
                calc(kf, RJstar, a, ss, se, os, oe, Jstar, Jstar + 1, false);
                sub_residual(RJstar, a, n_rows, n_cols, Jstar, Jstar + 1, k, false,
                             uv_ptr0);

                Istar_entry =
                    argmax_abs_not_in_list(RJstar, n_rows, 1, prevIstar, k, true);
                Istar = Istar_entry.row;

                calc(kf, RIstar, a, ss, se, os, oe, Istar, Istar + 1, true);
                sub_residual(RIstar, a, n_rows, n_cols, Istar, Istar + 1, k, true,
                             uv_ptr0);
            }

            bool done = false;

            prevIstar[k] = Istar;
            prevJstar[k] = Jstar;

            // claim a block of space for the first U and first V vectors and collect
            // the corresponding Real* pointers
            int next_buffer_u_ptr = buffer_alloc(a.next_buffer_ptr, n_rows + n_cols);
            int next_buffer_v_ptr = next_buffer_u_ptr + n_rows;
            Real* next_buffer_u = &a.buffer[next_buffer_u_ptr];
            Real* next_buffer_v = &a.buffer[next_buffer_v_ptr];

            // Assign our uv_ptr to point to the u,v buffer location.
            a.uv_ptrs[uv_ptr0 + k] = next_buffer_u_ptr;

            Real v2 = 0;
            for (int i = 0; i < n_cols; i++) {
                next_buffer_v[i] = RIstar[i] / RIstar[Jstar];
                v2 += next_buffer_v[i] * next_buffer_v[i];
            }

            Real u2 = 0;
            for (int j = 0; j < n_rows; j++) {
                next_buffer_u[j] = RJstar[j];
                u2 += next_buffer_u[j] * next_buffer_u[j];
            }

            if (a.verbose) {
                printf("true pivot: %i %i \n", Istar, Jstar);
                printf("diagonal %f \n", RIstar[Jstar]);
                for (int i = 0; i < 5; i++) {
                    printf("u[%i] = %f\n", i, next_buffer_u[i]);
                }
                for (int j = 0; j < 5; j++) {
                    printf("v[%i] = %f\n", j, next_buffer_v[j]);
                }
            }

            Real step_size = sqrt(u2 * v2);

            frob_est += step_size;
            if (a.verbose) {
                printf("step_size %f \n", step_size);
                printf("frob_est: %f \n", frob_est);
            }
            if (step_size < tol) {
                done = true;
            }

            if (k == max_iter - 1) {
                done = true;
            }

            if (done) {
                break;
            }

            if (Iref <= Istar && Istar < Iref + a.row_dim) {
                while (true) {
                    Iref = (Iref + a.row_dim) % n_rows;
                    Iref -= Iref % a.row_dim;
                    if (!in(Iref, prevIstar, k + 1)) {
                        if (a.verbose) {
                            printf("new Iref: %i \n", Iref);
                        }
                        break;
                    }
                }
                calc(kf, RIref, a, ss, se, os, oe, Iref, Iref + a.row_dim, true);
                sub_residual(RIref, a, n_rows, n_cols, Iref, Iref + a.row_dim, k + 1,
                             true, uv_ptr0);
            } else {
                Real* next_buffer_u = &a.buffer[a.uv_ptrs[uv_ptr0 + k]];
                Real* next_buffer_v = &a.buffer[a.uv_ptrs[uv_ptr0 + k] + n_rows];
                for (int i = 0; i < a.row_dim; i++) {
                    for (int j = 0; j < n_cols; j++) {
                        RIref[i * n_cols + j] -=
                            next_buffer_u[i + Iref] * next_buffer_v[j];
                    }
                }
            }

            if (Jref <= Jstar && Jstar < Jref + a.col_dim) {
                while (true) {
                    Jref = (Jref + a.col_dim) % n_cols;
                    Jref -= Jref % a.col_dim;
                    if (!in(Jref, prevJstar, k + 1)) {
                        if (a.verbose) {
                            printf("new Jref: %i \n", Jref);
                        }
                        break;
                    }
                }
                calc(kf, RJref, a, ss, se, os, oe, Jref, Jref + a.col_dim, false);
                sub_residual(RJref, a, n_rows, n_cols, Jref, Jref + a.col_dim, k + 1,
                             false, uv_ptr0);
            } else {
                Real* next_buffer_u = &a.buffer[a.uv_ptrs[uv_ptr0 + k]];
                Real* next_buffer_v = &a.buffer[a.uv_ptrs[uv_ptr0 + k] + n_rows];
                for (int i = 0; i < n_rows; i++) {
                    for (int j = 0; j < a.col_dim; j++) {
                        RJref[i * a.col_dim + j] -=
                            next_buffer_u[i] * next_buffer_v[j + Jref];
                    }
                }
            }
        }

        a.n_terms[block_idx] = k + 1;
    }
}

void aca_single_layer(const ACAArgs& a) { _aca_integrals(single_layer, a); }

void aca_double_layer(const ACAArgs& a) { _aca_integrals(double_layer, a); }

void aca_adjoint_double_layer(const ACAArgs& a) {
    _aca_integrals(adjoint_double_layer, a);
}

void aca_hypersingular(const ACAArgs& a) { _aca_integrals(hypersingular, a); }

void aca_elastic_U(const ACAArgs& a) { _aca_integrals(elastic_U, a); }
void aca_elastic_T(const ACAArgs& a) { _aca_integrals(elastic_T, a); }

void aca_elastic_A(const ACAArgs& a) { _aca_integrals(elastic_A, a); }

void aca_elastic_H(const ACAArgs& a) { _aca_integrals(elastic_H, a); }
