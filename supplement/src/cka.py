"""Linear CKA (Centered Kernel Alignment) for representation similarity."""

import numpy as np


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n, d1) representation matrix.
        Y: (n, d2) representation matrix.

    Returns:
        CKA similarity in [0, 1]. 1.0 = identical, 0.0 = orthogonal.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = np.trace(XtX @ YtY)
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)
