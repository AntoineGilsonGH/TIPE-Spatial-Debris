import numpy as np
import pytest

from kessler import tipe_model as tm
from kessler.ml.sindy import SINDy, polynomial_library, smooth_derivative
from kessler.ml.surrogate import sobol_indices


def test_library_labels():
    _, labels = polynomial_library(np.ones((3, 2)), ["a", "b"], 2)
    assert labels == ["1", "a", "b", "a a", "a b", "b b"]


def test_sindy_recovers_lotka_volterra_exactly():
    t = np.linspace(0, 20, 4001)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, s: [s[0] - 0.5 * s[0] * s[1], -2 * s[1] + 0.8 * s[0] * s[1]],
                    (0, 20), [3, 1], t_eval=t, rtol=1e-10, atol=1e-10)
    X = sol.y.T
    _, dX = smooth_derivative(X, t[1] - t[0], window=11, order=4)
    m = SINDy(["x", "y"], threshold=0.05).fit(X[10:-10], dX[10:-10])
    c = dict(zip(m.labels, m.coef_[:, 0]))
    assert c["x"] == pytest.approx(1.0, rel=1e-2)
    assert c["x y"] == pytest.approx(-0.5, rel=1e-2)
    assert np.count_nonzero(m.coef_) == 4


def test_sindy_finds_dominant_tipe_terms():
    p = tm.TipeParams()
    tr = tm.simulate(p, t_end=2222, n_points=1001, y0=[3000, 8e7], seasonal=False)
    X = np.c_[tr.N / 1e3, tr.n / 1e7]
    _, dX = smooth_derivative(X, tr.t[1] - tr.t[0])
    m = SINDy(["N", "n"], threshold=0.02).fit(X[20:-20], dX[20:-20])
    assert m.coef_[m.labels.index("N n"), 0] == pytest.approx(-p.x * 1e7, rel=0.03)


def test_sobol_on_additive_function():
    # f = x0 + 2 x1 : S1 = (1/5, 4/5), x2 inactive
    f = lambda U: (U[:, 0] + 2 * U[:, 1])[:, None]  # noqa: E731
    S1, ST = sobol_indices(f, 3, n=2**13)
    assert S1[:, 0] == pytest.approx([0.2, 0.8, 0.0], abs=0.03)
    assert ST[:, 0] == pytest.approx([0.2, 0.8, 0.0], abs=0.03)
