import numpy as np
import pytest

from kessler.shells import (DEB, SAT, ShellGrid, ShellModel, ShellParams, atmospheric_density,
                            breakup_fragments, launch_profile, residence_time_years,
                            simulate_shells)
from kessler.stochastic import simulate_mc

G = ShellGrid()


def test_density_decreases_with_altitude():
    rho = atmospheric_density(G.centers)
    assert np.all(np.diff(rho) < 0)
    assert atmospheric_density(400)[0] == pytest.approx(3.725e-12, rel=1e-6)


def test_lifetimes_grow_with_altitude_and_area_to_mass():
    tau_d, tau_s = residence_time_years(G, DEB), residence_time_years(G, SAT)
    assert np.all(np.diff(tau_d) > 0)
    assert np.all(tau_s > tau_d)


def test_breakup_model_value():
    # 0.1 * 520^0.75 * 0.1^-1.71 ~ 558 fragments > 10 cm
    assert breakup_fragments(520) == pytest.approx(558.5, rel=1e-3)


def test_fragment_kernel_conserves_or_loses_mass():
    m = ShellModel(ShellParams())
    col = m.K.sum(0)
    assert np.all(col <= 1 + 1e-9)
    assert col[G.n // 2] == pytest.approx(1, abs=0.01)  # far from borders: nothing lost


def test_no_collision_no_launch_only_decays():
    p = ShellParams(launch=launch_profile(G, "no_launch"), frag_scale=0.0)
    r = simulate_shells(p, years=50, n_points=11)
    assert np.all(np.diff(r.N.sum(1)) <= 1e-6)
    assert np.all(np.diff(r.S.sum(1)) <= 1e-6)


def test_policies_reduce_debris():
    base = simulate_shells(ShellParams(), years=100, n_points=3).N[-1].sum()
    good = simulate_shells(ShellParams(pmd=0.99), years=100, n_points=3).N[-1].sum()
    assert good < base


def test_monte_carlo_mean_close_to_ode():
    p = ShellParams(launch=launch_profile(G, "legacy"))
    t, S, D, N, C = simulate_mc(p, years=40, n_runs=60, seed=1)
    ode = simulate_shells(p, years=40, n_points=3)
    assert N[:, -1].mean() == pytest.approx(ode.N[-1].sum(), rel=0.05)
    assert np.all(N >= 0)
