"""Sparse identification of nonlinear dynamics (SINDy), from scratch.

Brunton, Proctor & Kutz, PNAS 113 (2016). Given noisy time series of a
system, find the *few* terms of a candidate library Theta(X) that explain
dX/dt = Theta(X) Xi, by sequentially thresholded least squares (STLSQ).

Here we ask: if we only had observations of satellites N(t) and fragments
n(t), could a machine re-discover the 2022 equations?
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.signal import savgol_filter


def polynomial_library(X: np.ndarray, names: list[str], degree: int = 2):
    cols, labels = [np.ones(len(X))], ["1"]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), deg):
            cols.append(np.prod(X[:, combo], axis=1))
            labels.append(" ".join(names[i] for i in combo))
    return np.column_stack(cols), labels


def stlsq(Theta: np.ndarray, dX: np.ndarray, threshold: float, ridge: float = 1e-6,
          max_iter: int = 20) -> np.ndarray:
    """Sequentially thresholded (ridge) least squares on column-normalised Theta."""
    norms = np.linalg.norm(Theta, axis=0)
    T = Theta / norms
    n_feat, n_tgt = T.shape[1], dX.shape[1]

    def solve(cols, y):
        A = T[:, cols]
        return np.linalg.solve(A.T @ A + ridge * np.eye(len(cols)), A.T @ y)

    Xi = np.zeros((n_feat, n_tgt))
    for j in range(n_tgt):
        active = np.arange(n_feat)
        coef = solve(active, dX[:, j])
        for _ in range(max_iter):
            # threshold relative to the largest coefficient (scale-free)
            keep = np.abs(coef) >= threshold * np.abs(coef).max()
            if keep.all():
                break
            active, coef = active[keep], solve(active[keep], dX[:, j])
        Xi[active, j] = coef
    return Xi / norms[:, None]


def smooth_derivative(x: np.ndarray, dt: float, window: int = 31, order: int = 3):
    """Savitzky-Golay smoothing and differentiation of noisy samples."""
    xs = savgol_filter(x, window, order, axis=0)
    dx = savgol_filter(x, window, order, deriv=1, delta=dt, axis=0)
    return xs, dx


class SINDy:
    def __init__(self, names: list[str], degree: int = 2, threshold: float = 0.05):
        self.names, self.degree, self.threshold = names, degree, threshold

    def fit(self, X: np.ndarray, dX: np.ndarray):
        Theta, self.labels = polynomial_library(X, self.names, self.degree)
        self.coef_ = stlsq(Theta, dX, self.threshold)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Theta, _ = polynomial_library(X, self.names, self.degree)
        return Theta @ self.coef_

    def equations(self, precision: int = 4) -> list[str]:
        out = []
        for j, name in enumerate(self.names):
            terms = [f"{c:+.{precision}g}" + ("" if lab == "1" else f" {lab}")
                     for c, lab in zip(self.coef_[:, j], self.labels) if c != 0]
            out.append(f"d{name}/dt = " + " ".join(terms))
        return out
