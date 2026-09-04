"""Streaming metrics for the common-subset history sensitivity evaluation."""

from __future__ import annotations

import math

import numpy as np


class StreamingRegressionMetrics:
    """Accumulate Pearson R, RMSE, and prediction efficiency without storing samples."""

    def __init__(self):
        self.n = 0
        self.sum_y = 0.0
        self.sum_p = 0.0
        self.sum_y2 = 0.0
        self.sum_p2 = 0.0
        self.sum_yp = 0.0
        self.sum_sq_err = 0.0

    def update(self, y_true, y_pred):
        y = np.asarray(y_true, dtype=np.float64).ravel()
        p = np.asarray(y_pred, dtype=np.float64).ravel()
        if y.shape != p.shape:
            raise ValueError(f"Shape mismatch: y_true={y.shape}, y_pred={p.shape}")
        finite = np.isfinite(y) & np.isfinite(p)
        y = y[finite]
        p = p[finite]
        if y.size == 0:
            return
        self.n += int(y.size)
        self.sum_y += float(y.sum())
        self.sum_p += float(p.sum())
        self.sum_y2 += float(np.dot(y, y))
        self.sum_p2 += float(np.dot(p, p))
        self.sum_yp += float(np.dot(y, p))
        diff = y - p
        self.sum_sq_err += float(np.dot(diff, diff))

    def finalize(self):
        if self.n == 0:
            return {"n": 0, "r": math.nan, "rmse": math.nan, "pe": math.nan}
        n = float(self.n)
        rmse = math.sqrt(self.sum_sq_err / n)
        sst = self.sum_y2 - (self.sum_y * self.sum_y) / n
        pe = 1.0 - self.sum_sq_err / sst if sst > 0.0 else math.nan
        cov_num = self.sum_yp - (self.sum_y * self.sum_p) / n
        var_y = self.sum_y2 - (self.sum_y * self.sum_y) / n
        var_p = self.sum_p2 - (self.sum_p * self.sum_p) / n
        denom = math.sqrt(max(var_y, 0.0) * max(var_p, 0.0))
        r = cov_num / denom if denom > 0.0 else math.nan
        return {"n": self.n, "r": r, "rmse": rmse, "pe": pe}


def log_total_flux(flux):
    """Convert linear total energy flux to the manuscript log10 convention."""
    values = np.asarray(flux, dtype=np.float64)
    return np.log10(np.clip(values, 1e-6, None))


class StreamingHistogram2D:
    """Streaming observed-vs-predicted two-dimensional histogram."""

    def __init__(self, edges):
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a one-dimensional array with at least two values")
        self.edges = edges
        n_bins = edges.size - 1
        self.counts = np.zeros((n_bins, n_bins), dtype=np.int64)

    def update(self, x, y):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return
        hist, _, _ = np.histogram2d(x[finite], y[finite], bins=[self.edges, self.edges])
        self.counts += hist.astype(np.int64)
