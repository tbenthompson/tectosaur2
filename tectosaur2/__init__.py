import numpy as np

from .integrate import integrate_term  # noqa: F401
from .mesh import gauss_rule, refine_surfaces  # noqa: F401


def tensor_dot(A, x):
    return np.sum(A * x[None, None, :, :], axis=(2, 3))
