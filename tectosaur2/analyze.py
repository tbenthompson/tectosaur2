import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from .global_qbx import global_qbx_self
from .mesh import apply_interp_mat, gauss_rule, panelize_symbolic_surface, upsample


def find_dcutoff_refine(kernel, src, tol, plot=False):
    # prep step 1: find d_cutoff and d_refine
    # The goal is to estimate the error due to the QBX local patch
    # The local surface will have singularities at the tips where it is cut off
    # These singularities will cause error in the QBX expansion. We want to make
    # the local patch large enough that these singularities are irrelevant.
    # To isolate the QBX patch cutoff error, we will use a very high upsampling.
    # We'll also choose p to be the minimum allowed value since that will result in
    # the largest cutoff error. Increasing p will reduce the cutoff error guaranteeing that
    # we never need to worry about cutoff error.
    density = np.ones_like(src.pts[:, 0])  # np.cos(src.pts[:,0] * src.pts[:,1])
    if plot:
        plt.figure(figsize=(9, 13))

    params = []
    d_cutoffs = [1.1, 1.3, 1.6, 2.0]
    ps = np.arange(1, 55, 3)
    for di, direction in enumerate([-1.0, 1.0]):
        baseline = global_qbx_self(kernel, src, p=30, kappa=8, direction=direction)
        baseline_v = baseline.dot(density)

        # Check that the local qbx method matches the simple global qbx approach when d_cutoff is very large
        d_refine_high = 8.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local_baseline = kernel.integrate(
                src.pts,
                src,
                d_cutoff=3.0,
                tol=1e-20,
                max_p=50,
                d_refine=d_refine_high,
                on_src_direction=direction,
            )
        local_baseline_v = local_baseline.dot(density)
        err = np.max(np.abs(baseline_v - local_baseline_v))
        print(err)
        assert err < tol / 2

        n_qbx_panels = []
        drefine_optimal = []
        p_for_full_accuracy = []
        if plot:
            plt.subplot(3, 2, 1 + di)
        for i_d, d_cutoff in enumerate(d_cutoffs):
            errs = []
            for i_p, p in enumerate(ps):
                # print(p, d_cutoff)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test, report = kernel.integrate(
                        src.pts,
                        src,
                        d_cutoff=d_cutoff,
                        tol=1e-15,
                        max_p=p,
                        on_src_direction=direction,
                        d_refine=d_refine_high,
                        return_report=True,
                    )
                    testv = test.dot(density)
                    err = np.max(np.abs(baseline_v - testv))
                    errs.append(err)
                    # print(p, err)
                    if err < tol:
                        for d_refine_decrease in np.arange(1.0, d_refine_high, 0.25):
                            refine_test, refine_report = kernel.integrate(
                                src.pts,
                                src,
                                d_cutoff=d_cutoff,
                                tol=1e-15,
                                max_p=p
                                + 10,  # Increase p here to have a refinement safety margin
                                on_src_direction=direction,
                                d_refine=d_refine_decrease,
                                return_report=True,
                            )
                            refine_testv = refine_test.dot(density)
                            refine_err = np.max(np.abs(baseline_v - refine_testv))
                            if refine_err < tol:
                                drefine_optimal.append(d_refine_decrease)
                                n_qbx_panels.append(refine_report["n_qbx_panels"])
                                p_for_full_accuracy.append(p)
                                break
                        if len(n_qbx_panels) <= i_d:
                            print(f"Failed to find parameters for {d_cutoff}")
                            drefine_optimal.append(1000)
                            n_qbx_panels.append(1e6)
                            p_for_full_accuracy.append(1e3)
                        break
            if plot:
                print(d_cutoff, errs)
                plt.plot(ps[: i_p + 1], np.log10(errs), label=str(d_cutoff))

        params.append((direction, n_qbx_panels, drefine_optimal, p_for_full_accuracy))

        if plot:
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
            plt.plot(d_cutoffs, np.array(drefine_optimal), "k-*")
            plt.xlabel(r"$d_{\textrm{cutoff}}$")
            plt.ylim([0, 6])
            if di == 0:
                plt.ylabel(r"$d_{\textrm{refine}}$")
    if plot:
        plt.tight_layout()
        plt.show()

    total_cost = 0
    for i in [0, 1]:
        direction, n_qbx_panels, drefine_optimal, p_for_full_accuracy = params[i]
        appx_cost = (
            np.array(p_for_full_accuracy)
            * np.array(n_qbx_panels)
            * np.array(drefine_optimal)
        )
        if plot:
            print(direction, appx_cost)
        total_cost += appx_cost
    if plot:
        plt.plot(d_cutoffs, total_cost, "k-o")
        plt.show()

    best_idx = np.argmin(total_cost)
    d_cutoff = d_cutoffs[best_idx]
    d_refine = drefine_optimal[best_idx]
    return d_cutoff, d_refine


# prep step 2: find the minimum distance at which integrals are computed
# to the required tolerance
def _find_d_up_helper(kernel, nq, max_curvature, start_d, tol, kappa):
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
    circle_kappa, _ = upsample(circle, kappa)
    circle_upsample, interp_mat_upsample = upsample(circle_kappa, 2)

    # TODO: Write more about the underlying regularity assumptions!!
    # Why is it acceptable to use this test_density here? Empirically, any
    # well-resolved density has approximately the same error as integrating sin(x).
    # For example, integrating: 1, cos(x)^2.
    # If we integrate a poorly resolved density, we do see higher errors.
    #
    # How poorly resolved does the density need to be in order to see higher error?
    # It seems like an interpolation Linfinity error of around 1e-5 causes the d_up value to start to drift upwards.
    #
    # As a simple heuristic that seems to perform very well, we compute the
    # error when integrating a constant and then double the required distance
    # in order to account for integrands that are not quite so perfectly
    # resolved.
    # if assume_regularity:
    #     omega = 1.0
    # else:
    #     omega = 999.0# / max_curvature
    # f = lambda x: np.sin(omega * x)
    # test_density = interp_mat_upsample.dot(f(circle.pts[:,0]))
    # test_density_upsampled = f(circle_upsample.pts[:,0])
    # print('l2 err', np.linalg.norm(test_density - test_density_upsampled) / np.linalg.norm(test_density_upsampled))
    # print('linf err', np.max(np.abs(test_density - test_density_upsampled)))
    # test_density = f(circle.pts[:,0])
    # test_density = np.sin(999 * circle.pts[:,0])
    test_density = np.ones(circle_kappa.n_pts)

    d_up = 0
    for direction in [-1.0, 1.0]:
        d = start_d
        for i in range(50):
            # In actuality, we only need to test interior points because the curvature
            # of the surface ensures that more source panels are near the observation
            # points and, as a result, the error will be higher for any given value of d.
            L = np.repeat(circle_kappa.panel_length, circle_kappa.panel_order)
            dist = L * d
            test_pts = (
                circle_kappa.pts + direction * circle_kappa.normals * dist[:, None]
            )

            # Check to make sure that the closest distance to a source point is
            # truly `dist`.  This check might fail if the interior test_pts are
            # crossing over into the other half of the circle.
            min_src_dist = np.min(
                np.linalg.norm((test_pts[:, None] - circle_kappa.pts[None, :]), axis=2),
                axis=1,
            )
            if not np.allclose(min_src_dist, dist):
                return False, d

            upsample_mat = np.transpose(
                apply_interp_mat(
                    kernel._direct(test_pts, circle_upsample), interp_mat_upsample
                ),
                (0, 2, 1),
            )
            est_mat = np.transpose(kernel._direct(test_pts, circle_kappa), (0, 2, 1))

            # err = np.max(np.abs(upsample_mat - est_mat).sum(axis=2))
            err = np.max(
                np.abs(upsample_mat.dot(test_density) - est_mat.dot(test_density))
            )

            # print(d, err)
            if err < tol:
                d_up = max(d, d_up)
                break
            d *= 1.2
    return True, d_up


def find_d_up(kernel, nq, max_curvature, start_d, tol, kappa):
    d = start_d
    for i in range(10):
        d_up = _find_d_up_helper(kernel, nq, max_curvature * (0.8) ** i, d, tol, kappa)
        if d_up[0]:
            return d_up[1]
        d = d_up[1]


def final_check(kernel, src):
    density = np.ones_like(src.pts[:, 0])  # np.cos(source.pts[:,0] * src.pts[:,1])
    baseline = global_qbx_self(kernel, src, p=50, kappa=10, direction=1.0)
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
