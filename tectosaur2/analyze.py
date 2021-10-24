import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from .global_qbx import global_qbx_self
from .mesh import apply_interp_mat, gauss_rule, panelize_symbolic_surface, upsample


def find_dcutoff_kappa(kernel, src, tol):
    # prep step 1: find d_cutoff and kappa
    # The goal is to estimate the error due to the QBX local patch
    # The local surface will have singularities at the tips where it is cut off
    # These singularities will cause error in the QBX expansion. We want to make
    # the local patch large enough that these singularities are irrelevant.
    # To isolate the QBX patch cutoff error, we will use a very high upsampling.
    # We'll also choose p to be the minimum allowed value since that will result in
    # the largest cutoff error. Increasing p will reduce the cutoff error guaranteeing that
    # we never need to worry about cutoff error.
    density = np.ones_like(src.pts[:, 0])  # np.cos(src.pts[:,0] * src.pts[:,1])
    plt.figure(figsize=(9, 13))

    params = []
    d_cutoffs = [1.1, 1.3, 1.6, 2.0]
    ps = np.arange(1, 55, 3)
    for di, direction in enumerate([-1.0, 1.0]):
        baseline = global_qbx_self(src, p=15, kappa=10, direction=direction)
        baseline_v = baseline[:, 0, :].dot(density)

        # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
        d_cutoff = 100.0
        local_baseline = kernel.integrate(
            src.pts,
            src,
            d_cutoff=100.0,
            tol=tol,
            max_p=10,
            kappa=10,
            on_src_direction=direction,
        )
        local_baseline_v = local_baseline.dot(density)
        assert np.max(np.abs(baseline_v - local_baseline_v)) < 5e-14

        n_qbx_panels = []
        kappa_optimal = []
        p_for_full_accuracy = []
        plt.subplot(3, 2, 1 + di)
        for i_d, d_cutoff in enumerate(d_cutoffs):
            errs = []
            for i_p, p in enumerate(ps):
                # print(p, d_cutoff)
                kappa_temp = 8
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test, report = kernel.integrate(
                        src.pts,
                        src,
                        d_cutoff=d_cutoff,
                        tol=tol,
                        max_p=p,
                        on_src_direction=direction,
                        kappa=kappa_temp,
                        return_report=True,
                    )
                    testv = test[:, 0, :].dot(density)
                    err = np.max(np.abs(baseline_v - testv))
                    errs.append(err)
                    if err < tol:
                        for kappa_decrease in range(1, kappa_temp + 1):
                            kappa_test, kappa_report = kernel.integrate(
                                src.pts,
                                src,
                                d_cutoff=d_cutoff,
                                tol=tol * 0.8,  # Increase tol to have a safety margin.
                                max_p=p
                                + 20,  # Increase p here to have a kappa safety margin
                                on_src_direction=direction,
                                kappa=kappa_decrease,
                                return_report=True,
                            )
                            kappa_testv = kappa_test[:, 0, :].dot(density)
                            kappa_err = np.max(np.abs(baseline_v - kappa_testv))
                            if kappa_err < tol:
                                kappa_optimal.append(kappa_decrease)
                                n_qbx_panels.append(kappa_report["n_qbx_panels"])
                                p_for_full_accuracy.append(p)
                                break
                        if len(n_qbx_panels) <= i_d:
                            print(f"Failed to find parameters for {d_cutoff}")
                            kappa_optimal.append(1000)
                            n_qbx_panels.append(1e6)
                            p_for_full_accuracy.append(1e3)
                        break
            print(d_cutoff, errs)
            plt.plot(ps[: i_p + 1], np.log10(errs), label=str(d_cutoff))

        params.append((direction, n_qbx_panels, kappa_optimal, p_for_full_accuracy))

        plt.legend()
        plt.title("interior" if direction > 0 else "exterior")
        plt.xlabel(r"$p_{\textrm{max}}$")
        if di == 0:
            plt.ylabel(r"$\log_{10}(\textrm{error})$")
        plt.yticks(-np.arange(0, 16, 3))
        plt.xticks(np.arange(0, 61, 10))
        plt.ylim([-15, 0])

        plt.subplot(3, 2, 3 + di)
        plt.plot(d_cutoffs, np.array(n_qbx_panels) / src.n_pts, "k-*")
        plt.xlabel(r"$d_{\textrm{cutoff}}$")
        plt.ylim([0, 8])
        if di == 0:
            plt.ylabel("QBX panels per point")

        plt.subplot(3, 2, 5 + di)
        plt.plot(d_cutoffs, np.array(kappa_optimal), "k-*")
        plt.xlabel(r"$d_{\textrm{cutoff}}$")
        plt.ylim([0, 6])
        if di == 0:
            plt.ylabel(r"$\kappa_{\textrm{optimal}}$")
    plt.tight_layout()
    plt.show()

    total_cost = 0
    for i in [0, 1]:
        direction, n_qbx_panels, kappa_optimal, p_for_full_accuracy = params[i]
        appx_cost = (
            np.array(p_for_full_accuracy)
            * np.array(n_qbx_panels)
            * np.array(kappa_optimal)
        )
        print(direction, appx_cost)
        total_cost += appx_cost
    plt.plot(d_cutoffs, total_cost, "k-o")
    plt.show()

    best_idx = np.argmin(total_cost)
    d_cutoff = d_cutoffs[best_idx]
    kappa_qbx = kappa_optimal[best_idx]
    return d_cutoff, kappa_qbx


# prep step 2: find the minimum distance at which integrals are computed
# to the required tolerance for each kappa in [1, kappa_qbx]
def find_safe_direct_distance(kernel, nq, max_curvature, start_d, tol, kappa):
    t = sp.var("t")

    n_panels = 2
    while True:
        panel_edges = np.linspace(-1, 1, n_panels + 1)
        panel_bounds = np.stack((panel_edges[:-1], panel_edges[1:]), axis=1)
        circle = panelize_symbolic_surface(
            t, sp.cos(sp.pi * t), sp.sin(sp.pi * t), panel_bounds, *gauss_rule(nq)
        )
        n_panels_new = np.max(circle.panel_length / max_curvature * circle.panel_radius)
        if n_panels_new <= n_panels:
            break
        n_panels = np.ceil(n_panels_new).astype(int)
    # print(f"\nusing {n_panels} panels with max_curvature={max_curvature}")

    L = np.repeat(circle.panel_length, circle.panel_order)

    circle_high, interp_mat_high = upsample(circle, kappa)
    circle_higher, interp_mat_higher = upsample(circle, 8)
    # test_density = np.cos(circle.pts[:,0] * circle.pts[:,1])
    test_density = np.ones_like(circle.pts[:, 0])
    d = start_d
    for i in range(50):
        dist = L * d
        # In actuality, we only need to test interior points because the curvature
        # of the surface ensures that more source panels are near the observation
        # points and, as a result, the error will be higher for any given value of d.
        test_pts = np.concatenate(
            (
                circle.pts + circle.normals * dist[:, None],
                circle.pts - circle.normals * dist[:, None],
            )
        )

        # Check to make sure that the closest distance to a source point is truly `dist`.
        # This check might fail if the interior test_pts are crossing over into the other half of the circle.
        min_src_dist = np.min(
            np.linalg.norm((test_pts[:, None] - circle.pts[None, :]), axis=2), axis=1
        )
        if not np.allclose(min_src_dist, np.concatenate((dist, dist))):
            return False, d

        higher_mat = apply_interp_mat(
            kernel._direct(test_pts, circle_higher), interp_mat_higher
        )
        high_mat = apply_interp_mat(
            kernel._direct(test_pts, circle_high), interp_mat_high
        )

        # Use the absolute value of the matrix coefficients in order to compute an upper bound on the error
        err = np.max(np.abs(higher_mat - high_mat).dot(test_density))
        if err < tol:
            return True, d
        d *= 1.2


def find_d_up(kernel, tol, nq, max_curvature, kappa_qbx):
    d_up = np.zeros(kappa_qbx)
    for k in range(kappa_qbx, 0, -1):
        max_iter = 20
        d_up[k - 1] = d_up[k] if k < kappa_qbx else 0.05
        for i in range(max_iter):
            result = find_safe_direct_distance(
                kernel, nq, max_curvature * (0.8) ** i, d_up[k - 1], tol, k
            )
            d_up[k - 1] = result[1]
            if result[0]:
                print("done", k, d_up[k - 1])
                break
    return d_up


def final_check(kernel, src):
    density = np.ones_like(src.pts[:, 0])  # np.cos(source.pts[:,0] * src.pts[:,1])
    baseline = global_qbx_self(src, p=50, kappa=10, direction=1.0)
    baseline_v = baseline.dot(density)
    tols = 10.0 ** np.arange(0, -15, -1)
    errs = []
    runtimes = []
    for tol in tols:
        runs = []
        for i in range(10):
            start = time.time()
            local_baseline, report = kernel.integrate(
                src.pts,
                src,
                tol=tol,
                on_src_direction=1.0,
                return_report=True,
            )
            runs.append(time.time() - start)
        runtimes.append(np.min(runs))
        local_baseline_v = local_baseline.dot(density)
        errs.append(np.max(np.abs(baseline_v - local_baseline_v)))
        # print(tol, errs[-1], runtime)
        # assert(np.max(np.abs(baseline_v-local_baseline_v)) < 5e-14)

    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(-np.log10(tols), np.log10(errs))
    plt.subplot(1, 2, 2)
    plt.plot(-np.log10(tols), runtimes)
    plt.tight_layout()
    plt.show()
