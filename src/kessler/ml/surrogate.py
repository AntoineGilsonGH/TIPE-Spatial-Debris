"""Neural surrogate of the shell model + global sensitivity analysis.

A 200-year simulation costs ~0.5 s. Uncertainty quantification needs
10^5-10^6 runs, a policy map 10^4: too slow. We therefore

1. sample the uncertain/policy parameters with a Latin hypercube,
2. run the physical model on those samples (in parallel),
3. train a neural network (MLP) to emulate input -> outcome,
4. use the cheap emulator for Sobol indices and policy maps,

and we always check the emulator on held-out simulations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..shells import ShellGrid, ShellParams, launch_profile, simulate_shells


@dataclass(frozen=True)
class Factor:
    name: str
    label: str
    low: float
    high: float
    log: bool = False

    def scale(self, u):
        if self.log:
            return np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        return self.low + u * (self.high - self.low)


FACTORS = [
    Factor("launch_mult", "Trafic de lancement (x scénario méga)", 0.1, 3.0, log=True),
    Factor("pmd", "Désorbitation fin de vie (PMD)", 0.5, 0.99),
    Factor("avoid_debris", "Évitement des débris", 0.0, 0.95),
    Factor("adr_per_year", "Retrait actif (ADR, objets/an)", 0.0, 30.0),
    Factor("frag_scale", "Fragments par collision (x NASA SBM)", 0.5, 2.0, log=True),
    Factor("solar_amplitude", "Amplitude du cycle solaire", 0.2, 1.0),
]
OUTPUTS = ["log10 débris 2226", "log10 collisions 2026-2226"]


def params_from_vector(v: np.ndarray, grid: ShellGrid = ShellGrid()) -> ShellParams:
    d = dict(zip([f.name for f in FACTORS], v))
    return ShellParams(
        grid=grid,
        launch=launch_profile(grid, "mega", d["launch_mult"]),
        pmd=d["pmd"], avoid_debris=d["avoid_debris"], adr_per_year=d["adr_per_year"],
        frag_scale=d["frag_scale"], solar_amplitude=d["solar_amplitude"],
    )


def run_one(v: np.ndarray) -> np.ndarray:
    r = simulate_shells(params_from_vector(v), years=200, n_points=3)
    return np.array([np.log10(r.N[-1].sum()), np.log10(1 + r.collisions[-1])])


def sample_inputs(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    U = qmc.LatinHypercube(d=len(FACTORS), seed=seed).random(n)
    X = np.column_stack([f.scale(U[:, i]) for i, f in enumerate(FACTORS)])
    return U, X


def build_dataset(n: int = 1500, seed: int = 0, workers: int | None = None):
    U, X = sample_inputs(n, seed)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        Y = np.array(list(ex.map(run_one, X, chunksize=8)))
    return U, X, Y


def train_surrogate(U: np.ndarray, Y: np.ndarray, seed: int = 0):
    """Train on unit-cube inputs (log factors are already uniform there)."""
    U_tr, U_te, Y_tr, Y_te = train_test_split(U, Y, test_size=0.2, random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh", alpha=1e-5,
                     learning_rate_init=2e-3, max_iter=4000, early_stopping=True,
                     n_iter_no_change=100, random_state=seed),
    )
    y_mean, y_std = Y_tr.mean(0), Y_tr.std(0)
    model.fit(U_tr, (Y_tr - y_mean) / y_std)

    def predict(U_):
        return model.predict(np.atleast_2d(U_)) * y_std + y_mean

    pred = predict(U_te)
    r2 = 1 - ((pred - Y_te) ** 2).sum(0) / ((Y_te - Y_te.mean(0)) ** 2).sum(0)
    return predict, {"U_test": U_te, "Y_test": Y_te, "Y_pred": pred, "r2": r2}


def sobol_indices(f, d: int, n: int = 2**15, seed: int = 0):
    """First-order (Saltelli 2010) and total (Jansen 1999) Sobol indices.

    f maps (m, d) unit-cube samples to (m, k) outputs. Returns (S1, ST) of shape (d, k).
    """
    base = qmc.Sobol(d=2 * d, seed=seed).random(n)
    A, B = base[:, :d], base[:, d:]
    fA, fB = f(A), f(B)
    var = np.var(np.vstack([fA, fB]), axis=0)
    S1, ST = [], []
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = f(ABi)
        S1.append(np.mean(fB * (fABi - fA), axis=0) / var)
        ST.append(0.5 * np.mean((fA - fABi) ** 2, axis=0) / var)
    return np.array(S1), np.array(ST)
