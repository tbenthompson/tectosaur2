import numpy as np

from .integrate import integrate_term  # noqa: F401
from .mesh import gauss_rule, refine_surfaces  # noqa: F401


def tensor_dot(A, x):
    assert x.shape[0] == A.shape[2]
    if A.shape[-1] == 1 and x.ndim == 1:
        return A[:, :, :, 0].dot(x)
    assert x.shape[1] == A.shape[3]
    return np.sum(A * x[None, None, :, :], axis=(2, 3))
