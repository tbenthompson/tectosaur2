from dataclasses import dataclass

import numpy as np


@dataclass
class MaterialProps:
    a: np.ndarray
    # state-based velocity weakening effect
    b: np.ndarray
    # state evolution length scale (m)
    Dc: np.ndarray
    # baseline coefficient of friction
    f0: float
    # if V = V0 and at steady state, then f = f0, units are m/s
    V0: float
    # The radiation damping coefficient (kg / (m^2 * s))
    eta: float


def aging_law(p, V, state):
    return (p.b * p.V0 / p.Dc) * (np.exp((p.f0 - state) / p.b) - (V / p.V0))


def qd_equation(p, normal_stress, shear_stress, V, state):
    # The regularized rate and state friction equation
    F = normal_stress * p.a * np.arcsinh(V / (2 * p.V0) * np.exp(state / p.a))

    # The full shear stress balance:
    return shear_stress - p.eta * V - F


def qd_equation_dV(p, normal_stress, V, state):
    # First calculate the derivative of the friction law with respect to velocity
    # This is helpful for equation solving using Newton's method
    expsa = np.exp(state / p.a)
    Q = (V * expsa) / (2 * p.V0)
    dFdV = p.a * expsa * normal_stress / (2 * p.V0 * np.sqrt(1 + Q * Q))

    # The derivative of the full shear stress balance.
    return -p.eta - dFdV


def solve_friction(mp, normal_stress, shear_stress, V_old, state, tol=1e-13):
    V = V_old
    max_iter = 150
    # history=[]
    for i in range(max_iter):
        # Newton's method step!
        f = qd_equation(mp, normal_stress, shear_stress, V, state)
        dfdv = qd_equation_dV(mp, normal_stress, V, state)
        step = f / dfdv

        # We know that slip velocity should not be negative so any step that
        # would lead to negative slip velocities is "bad". In those cases, we
        # cut the step size in half iteratively until the step is no longer
        # bad. This is a backtracking line search.
        while True:
            bad = step > V
            if np.any(bad):
                step[bad] *= 0.5
            else:
                break

        # Take the step and check if everything has converged.
        Vn = V - step
        criterion = np.max(np.abs(step) / Vn)
        # history.append(criterion)
        if criterion < tol:
            break
        V = Vn
        if i == max_iter - 1:
            # print(history[-10:])
            return Vn, i, False

    # Return the solution and the number of iterations required.
    return Vn, i, True
