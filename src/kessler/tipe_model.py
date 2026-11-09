"""The 2022 TIPE model: two populations in low Earth orbit.

    dN/dt = A(t) - N / T_N(t) - x n N - 2 y N^2
    dn/dt = beta A(t) - n / T_n(t) + alpha x n N + gamma y N^2 - 2 z n^2

N  : intact satellites
n  : fragments larger than ~1 cm
A  : launch rate, a + b sin(c t + d)
T_N, T_n : atmospheric lifetimes, modulated by the 11-year solar cycle

The constants are the ones used in the original 2022 work (see ``legacy/``).
Compared with the original code this module

* fixes a typo of the 2022 scripts (``-2*z**2`` instead of ``-2*z*n**2``),
* uses an adaptive stiff/non-stiff solver (LSODA) with controlled tolerances,
* adds the *analysis* that was missing in 2022: equilibrium, its stability and
  a closed-form explanation of the asymptotic limit observed numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


@dataclass(frozen=True)
class TipeParams:
    # launches (satellites / year): a + b sin(c t + d)
    a: float = 31.41
    b: float = 7.794
    c: float = 1.935
    d: float = 0.1680
    # satellite lifetime (years): f + g sin(h t + k)
    f: float = 14420.0
    g: float = -10430.0
    h: float = 0.5712  # 2 pi / 11 yr -> solar cycle
    k: float = -0.9996
    # fragment lifetime (years): p + q sin(h t + k)
    p: float = 184.9
    q: float = -137.9
    # collision coefficients (1 / year / object)
    x: float = 6.895e-10  # fragment - satellite
    y: float = 1.369e-9   # satellite - satellite
    z: float = 2.869e-14  # fragment - fragment
    # fragments created
    alpha: float = 1e4     # per satellite destroyed by a fragment
    beta: float = 70.0     # per launch (rocket bodies, mission-related debris)
    gamma: float = 56000.0  # per satellite-satellite collision
    # initial state (2022)
    N0: float = 4960.0
    n0: float = 110400.0
    t0: float = 2022.0

    def with_(self, **kw) -> "TipeParams":
        return replace(self, **kw)

    # ---- time-dependent coefficients ---------------------------------
    def launch_rate(self, t):
        return self.a + self.b * np.sin(self.c * t + self.d)

    def sat_lifetime(self, t):
        return self.f + self.g * np.sin(self.h * t + self.k)

    def frag_lifetime(self, t):
        return self.p + self.q * np.sin(self.h * t + self.k)

    # ---- cycle-averaged coefficients ---------------------------------
    @property
    def mean_launch_rate(self) -> float:
        return self.a

    @property
    def eff_sat_lifetime(self) -> float:
        """Lifetime T such that 1/T = <1/T_N(t)> over a solar cycle.

        For T(t) = f + g sin(.) the average of 1/T is exactly
        1 / sqrt(f^2 - g^2), so the effective lifetime is the geometric-like
        mean sqrt(f^2 - g^2) - noticeably *shorter* than the arithmetic mean f:
        the solar maxima dominate the cleaning.
        """
        return float(np.sqrt(self.f**2 - self.g**2))

    @property
    def eff_frag_lifetime(self) -> float:
        return float(np.sqrt(self.p**2 - self.q**2))


def rhs(t, s, p: TipeParams, legacy_bug: bool = False, seasonal: bool = True):
    N, n = s
    if seasonal:
        A, TN, Tn = p.launch_rate(t), p.sat_lifetime(t), p.frag_lifetime(t)
    else:
        A, TN, Tn = p.mean_launch_rate, p.eff_sat_lifetime, p.eff_frag_lifetime
    ff = p.z**2 if legacy_bug else p.z * n * n
    dN = A - N / TN - p.x * n * N - 2 * p.y * N * N
    dn = p.beta * A - n / Tn + p.alpha * p.x * n * N + p.gamma * p.y * N * N - 2 * ff
    return np.array([dN, dn])


@dataclass
class Trajectory:
    t: np.ndarray
    N: np.ndarray
    n: np.ndarray


def simulate(
    p: TipeParams = TipeParams(),
    t_end: float = 2700.0,
    n_points: int = 5000,
    y0=None,
    legacy_bug: bool = False,
    seasonal: bool = True,
    rtol: float = 1e-8,
) -> Trajectory:
    y0 = [p.N0, p.n0] if y0 is None else y0
    t_eval = np.linspace(p.t0, t_end, n_points)
    sol = solve_ivp(
        rhs, (p.t0, t_end), y0, args=(p, legacy_bug, seasonal),
        method="LSODA", t_eval=t_eval, rtol=rtol, atol=1e-6,
    )
    if not sol.success:  # pragma: no cover
        raise RuntimeError(sol.message)
    return Trajectory(sol.t, sol.y[0], sol.y[1])


# ---------------------------------------------------------------------
# Analysis of the cycle-averaged (autonomous) system
# ---------------------------------------------------------------------
def kessler_threshold(p: TipeParams = TipeParams()) -> float:
    """Critical number of satellites N_c = 1 / (alpha x T_n).

    In dn/dt the fragments have a linear "growth rate" alpha x N - 1/T_n.
    Above N_c each fragment, during its lifetime, destroys enough satellites
    to create more than one new fragment: the cascade is self-sustaining.
    This is the exact analogue of the epidemic threshold of an SIR model.
    """
    return 1.0 / (p.alpha * p.x * p.eff_frag_lifetime)


def reproduction_number(N, p: TipeParams = TipeParams()):
    """R = alpha x N T_n: fragments generated, per fragment, over its lifetime."""
    return p.alpha * p.x * np.asarray(N) * p.eff_frag_lifetime


def analytic_equilibrium(p: TipeParams = TipeParams()) -> tuple[float, float]:
    """Leading-order closed form of the attracting equilibrium.

    When the cascade dominates (alpha >> beta, T_N very long, y z tiny):
        dn/dt = 0  ->  alpha x N* = 1 / T_n         ->  N* = N_c
        dN/dt = 0  ->  A = x n* N*                  ->  n* = alpha A T_n
    i.e. *every* launched satellite ends up destroyed by a fragment and turned
    into alpha fragments, which live T_n years. The initial state plays no role.
    """
    return kessler_threshold(p), p.alpha * p.mean_launch_rate * p.eff_frag_lifetime


def equilibrium(p: TipeParams = TipeParams()) -> tuple[float, float]:
    """Exact equilibrium of the averaged system (numerical root)."""
    guess = analytic_equilibrium(p)
    scale = np.array(guess)

    def f(u):
        return rhs(0.0, u * scale, p, seasonal=False) / np.array([p.a, p.alpha * p.a])

    u = fsolve(f, np.ones(2), xtol=1e-12)
    return tuple(u * scale)


def jacobian(state, p: TipeParams = TipeParams()) -> np.ndarray:
    N, n = state
    TN, Tn = p.eff_sat_lifetime, p.eff_frag_lifetime
    return np.array([
        [-1 / TN - p.x * n - 4 * p.y * N, -p.x * N],
        [p.alpha * p.x * n + 2 * p.gamma * p.y * N, -1 / Tn + p.alpha * p.x * N - 4 * p.z * n],
    ])


def stability(p: TipeParams = TipeParams()) -> dict:
    eq = equilibrium(p)
    eig = np.linalg.eigvals(jacobian(eq, p))
    return {
        "equilibrium": eq,
        "eigenvalues": eig,
        "stable": bool(np.all(eig.real < 0)),
        "relaxation_time_years": float(1 / np.min(np.abs(eig.real))),
        "oscillation_period_years": (float(2 * np.pi / abs(eig.imag).max())
                                     if abs(eig.imag).max() > 0 else np.inf),
    }


def collision_probability_per_year(n, p: TipeParams = TipeParams()):
    """Probability that a given satellite is hit by a fragment within a year."""
    return 1 - np.exp(-p.x * np.asarray(n))
