"""Monte Carlo version of the shell model (tau-leaping).

Catastrophic collisions involving objects > 10 cm are *rare*: a few per
decade. An ODE only gives the mean; in reality the future depends on when the
next Iridium-Cosmos-like event happens. Here every event is drawn at random:

* collisions in each shell ~ Poisson(rate * dt)          (the "loi de Poisson"
  that opened the 2022 project, finally used for what it is good at)
* drag exits and retirements ~ Binomial(X, 1 - exp(-r dt))
* fragments of each collision ~ Poisson(yield), spread over shells.

Professional evolutionary codes (NASA LEGEND, ESA DELTA, CNES MEDEE) are run
the same way: hundreds of Monte Carlo runs, then statistics.
"""

from __future__ import annotations

import numpy as np

from .shells import ShellModel, ShellParams, initial_population, solar_factor


def simulate_mc(p: ShellParams = ShellParams(), years: float = 200.0, dt: float = 0.1,
                n_runs: int = 200, seed: int = 0, record_every: int = 10):
    """Run ``n_runs`` independent histories; returns (t, S, D, N, collisions).

    Arrays have shape (n_runs, n_records) and hold LEO totals.
    """
    rng = np.random.default_rng(seed)
    m = ShellModel(p)
    ns = m.ns
    y0 = np.rint(initial_population(p.grid)).astype(np.int64)
    S = np.tile(y0[:ns], (n_runs, 1))
    D = np.tile(y0[ns:2 * ns], (n_runs, 1))
    N = np.tile(y0[2 * ns:], (n_runs, 1))
    coll = np.zeros(n_runs, dtype=np.int64)

    n_steps = int(round(years / dt))
    rec = []
    t = p.t0
    p_exit = lambda rate: -np.expm1(-rate * dt)  # noqa: E731

    for step in range(n_steps + 1):
        if step % record_every == 0:
            rec.append((t, S.sum(1), D.sum(1), N.sum(1), coll.copy()))
        if step == n_steps:
            break
        sf = solar_factor(t, p.solar_amplitude)
        r = m.rates(t, S.astype(float), D.astype(float), N.astype(float))
        # collisions (Poisson), capped by what exists
        c = {k: rng.poisson(r[k] * dt) for k in ("c_SS", "c_SD", "c_DD", "c_SN", "c_DN")}
        big = c["c_SS"] + c["c_SD"] + c["c_DD"]
        small = c["c_SN"] + c["c_DN"]
        coll += (big + small).sum(1)
        mean_frag = (big * m.y_ss + small * m.y_sn) @ m.K.T
        frags = rng.poisson(mean_frag)

        retire = rng.binomial(S, p_exit(1 / p.mission_life))
        failed = rng.binomial(retire, 1 - p.pmd)
        dD = rng.binomial(D, p_exit(m._decay_sat * sf))
        dN = rng.binomial(N, p_exit(m._decay_deb * sf))
        launches = rng.poisson(r["launch"] * dt, size=S.shape)
        adr = rng.poisson(r["adr"] * dt)

        S = S + launches - retire - 2 * c["c_SS"] - c["c_SD"] - c["c_SN"]
        D = D + failed - dD - c["c_SD"] - 2 * c["c_DD"] - c["c_DN"] - adr
        N = N - dN - c["c_SN"] - c["c_DN"] + frags
        D[:, :-1] += dD[:, 1:]
        N[:, :-1] += dN[:, 1:]
        np.maximum(S, 0, out=S)
        np.maximum(D, 0, out=D)
        np.maximum(N, 0, out=N)
        t += dt

    t_rec, S_rec, D_rec, N_rec, C_rec = (np.array(v) for v in zip(*rec))
    return t_rec, S_rec.T, D_rec.T, N_rec.T, C_rec.T
