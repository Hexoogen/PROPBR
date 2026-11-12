# -*- coding: utf-8 -*-
"""Accuracy metrics used in the paper.

Import-safe: no data files are read at import time.

    r2 / rmse     -- sklearn metrics re-exported for convenience
    r2f(y, p)     -- R^2 after a least-squares line fit between prediction and truth
                     (the convention used in the retrieval tables)
    report(y, p)  -- dict with RMSE, R^2 and fitted-line R^2
"""
import numpy as np
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse

__all__ = ["r2", "rmse", "r2f", "report"]


def r2f(y, p):
    """R^2 between truth `y` and prediction `p` after fitting a line p -> y.

    NaN / inf entries in either array are dropped; fewer than two valid points
    gives NaN.
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return float("nan")
    slope, intercept = np.polyfit(p[m], y[m], 1)
    return float(r2(y[m], p[m] * slope + intercept))


def report(y, p):
    """Pooled RMSE and both R^2 conventions for one trait."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    return {
        "n": int(m.sum()),
        "rmse": float(rmse(y[m], p[m])),
        "r2": float(r2(y[m], p[m])),
        "r2_fit": r2f(y, p),
    }
