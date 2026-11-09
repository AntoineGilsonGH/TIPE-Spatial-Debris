import numpy as np
import pytest

from kessler import tipe_model as tm

P = tm.TipeParams()


def test_effective_lifetime_is_exact_cycle_average():
    t = np.linspace(0, 2 * np.pi / P.h, 200_001)
    assert 1 / np.mean(1 / P.frag_lifetime(t)) == pytest.approx(P.eff_frag_lifetime, rel=1e-4)


def test_analytic_equilibrium_matches_numerical_root():
    Na, na = tm.analytic_equilibrium(P)
    Ne, ne = tm.equilibrium(P)
    assert Ne == pytest.approx(Na, rel=0.02)
    assert ne == pytest.approx(na, rel=0.02)
    assert np.allclose(tm.rhs(0, [Ne, ne], P, seasonal=False), 0, atol=1e-6 * P.alpha * P.a)


def test_equilibrium_is_stable_focus():
    st = tm.stability(P)
    assert st["stable"]
    assert np.all(np.abs(st["eigenvalues"].imag) > 0)


@pytest.mark.parametrize("y0", [(4960, 110400), (9920, 110400), (4960, 0), (4960, 1.104e8)])
def test_limit_independent_of_initial_conditions(y0):
    """The 2022 observation: same limit whatever the initial state."""
    tr = tm.simulate(P, t_end=3500, n_points=500, y0=list(y0), seasonal=False)
    Ne, ne = tm.equilibrium(P)
    assert tr.N[-1] == pytest.approx(Ne, rel=1e-2)
    assert tr.n[-1] == pytest.approx(ne, rel=1e-2)


def test_legacy_bug_has_negligible_effect():
    a = tm.simulate(P, t_end=2300, n_points=200)
    b = tm.simulate(P, t_end=2300, n_points=200, legacy_bug=True)
    assert np.max(np.abs(a.n - b.n) / a.n) < 1e-3


def test_reproduction_number_is_one_at_threshold():
    assert tm.reproduction_number(tm.kessler_threshold(P), P) == pytest.approx(1.0)
