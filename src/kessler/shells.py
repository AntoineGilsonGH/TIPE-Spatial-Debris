"""Multi-shell source-sink model of LEO (in the spirit of MIT's MOCAT-SSEM).

The single-box model of 2022 cannot say *where* debris accumulates, nor
distinguish a manoeuvrable satellite from a dead one. Here LEO is cut into
altitude shells and three species are tracked in each shell:

    S : active satellites (manoeuvre to avoid collisions, deorbit at end of life)
    D : derelicts (dead satellites, rocket bodies)
    N : debris larger than 10 cm (the tracked population)

Physics included
----------------
* Atmospheric drag on circular orbits, da/dt = -rho(h) B sqrt(mu r), with the
  piecewise-exponential atmosphere of Vallado (Table 8-4) - the isothermal
  barometric law of the 2022 appendix, applied band by band - modulated by the
  11-year solar cycle. Objects flow from shell to shell, down to re-entry.
* Collisions from kinetic gas theory: rate = sigma v_rel / V_shell per pair.
* Fragments from the NASA standard breakup model (catastrophic collisions):
  N(>Lc) = 0.1 M^0.75 Lc^-1.71, spread over neighbouring shells.
* Space-traffic policies: post-mission disposal (PMD), collision avoidance,
  active debris removal (ADR).

All numbers describing *today's* population and future launch traffic are
illustrative orders of magnitude, not a calibrated catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

MU = 3.986004418e14  # m^3 s^-2
R_EARTH_KM = 6378.137
SEC_PER_YEAR = 365.25 * 86400
KM_PER_YEAR_PER_M_PER_S = SEC_PER_YEAR / 1000.0

# Vallado, "Fundamentals of Astrodynamics", Table 8-4 (exponential atmosphere)
_ATM = np.array([
    # base alt km, nominal density kg/m3, scale height km
    [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740],
    [200, 2.789e-10, 37.105],
    [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628],
    [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828],
    [500, 6.967e-13, 63.822],
    [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667],
    [800, 1.170e-14, 124.64],
    [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
    [1100, 1.170e-15, 268.00],  # extension: same scale height as 1000 km
])


def atmospheric_density(h_km) -> np.ndarray:
    """Piecewise exponential density (kg/m^3), mean solar activity."""
    h = np.atleast_1d(np.asarray(h_km, dtype=float))
    idx = np.clip(np.searchsorted(_ATM[:, 0], h, side="right") - 1, 0, len(_ATM) - 1)
    h0, rho0, H = _ATM[idx].T
    return rho0 * np.exp(-(h - h0) / H)


def solar_factor(t, amplitude: float = 0.7, period: float = 11.0, phase_year: float = 2025.0):
    """Multiplicative density factor exp(A sin(2 pi (t - t_max)/P + pi/2)).

    With A = 0.7 the density varies by a factor ~4 between solar minimum and
    maximum, the order of magnitude observed at 400-800 km.
    """
    return np.exp(amplitude * np.cos(2 * np.pi * (t - phase_year) / period))


@dataclass(frozen=True)
class ObjectClass:
    mass: float      # kg
    radius: float    # m (collision radius)
    area_to_mass: float  # m^2/kg
    cd: float = 2.2

    @property
    def ballistic(self) -> float:  # B = Cd A / m (m^2/kg)
        return self.cd * self.area_to_mass


SAT = ObjectClass(mass=260.0, radius=1.0, area_to_mass=0.012)
DEB = ObjectClass(mass=0.64, radius=0.05, area_to_mass=0.05)


def breakup_fragments(total_mass_kg: float, lc_m: float = 0.10) -> float:
    """NASA standard breakup model, catastrophic collision, fragments > Lc."""
    return 0.1 * total_mass_kg**0.75 * lc_m**-1.71


@dataclass(frozen=True)
class ShellGrid:
    h_min: float = 200.0
    h_max: float = 2000.0
    dh: float = 50.0

    @property
    def edges(self) -> np.ndarray:
        return np.arange(self.h_min, self.h_max + self.dh / 2, self.dh)

    @property
    def centers(self) -> np.ndarray:
        e = self.edges
        return 0.5 * (e[1:] + e[:-1])

    @property
    def n(self) -> int:
        return len(self.centers)

    @property
    def volumes_km3(self) -> np.ndarray:
        r = R_EARTH_KM + self.edges
        return 4 / 3 * np.pi * (r[1:] ** 3 - r[:-1] ** 3)


def gaussian_profile(grid: ShellGrid, center_km: float, width_km: float, total: float):
    w = np.exp(-0.5 * ((grid.centers - center_km) / width_km) ** 2)
    return total * w / w.sum()


@dataclass(frozen=True)
class ShellParams:
    grid: ShellGrid = field(default_factory=ShellGrid)
    v_rel_km_s: float = 10.0
    mission_life: float = 5.0          # years
    pmd: float = 0.90                  # post-mission disposal success
    avoid_large: float = 0.99          # active sats avoid tracked intact objects
    avoid_debris: float = 0.80         # ... and tracked debris (> 10 cm)
    adr_per_year: float = 0.0          # derelicts actively removed per year
    frag_scale: float = 1.0            # multiplier on breakup-model yield
    frag_spread_km: float = 100.0      # altitude spread of a fragment cloud
    solar_amplitude: float = 0.7
    launch: Callable[[float], np.ndarray] | None = None
    t0: float = 2026.0

    def with_(self, **kw) -> "ShellParams":
        return replace(self, **kw)


# ---------------------------------------------------------------------
# Scenarios (illustrative)
# ---------------------------------------------------------------------
def launch_profile(grid: ShellGrid, scenario: str, multiplier: float = 1.0):
    """Return launch(t) -> satellites/year in each shell."""
    legacy = gaussian_profile(grid, 700, 150, 150.0)
    if scenario == "no_launch":
        vec = np.zeros(grid.n)
    elif scenario == "legacy":       # pre-2019 pace, ~150 satellites/year
        vec = legacy
    elif scenario == "mega":         # large constellations replenished every 5 yr
        vec = (legacy
               + gaussian_profile(grid, 550, 25, 2000.0)
               + gaussian_profile(grid, 1150, 25, 300.0))
    else:
        raise ValueError(scenario)
    vec = vec * multiplier
    return lambda t: vec


def initial_population(grid: ShellGrid) -> np.ndarray:
    """Illustrative 2026 LEO population: ~10k active, ~3.5k derelicts, ~25k debris."""
    S = gaussian_profile(grid, 550, 25, 8000) + gaussian_profile(grid, 700, 150, 2000)
    D = gaussian_profile(grid, 800, 150, 2500) + gaussian_profile(grid, 1400, 200, 1000)
    N = gaussian_profile(grid, 800, 150, 20000) + gaussian_profile(grid, 1400, 250, 5000)
    return np.concatenate([S, D, N])


# ---------------------------------------------------------------------
# Rates (shared by the ODE and the stochastic simulator)
# ---------------------------------------------------------------------
class ShellModel:
    def __init__(self, p: ShellParams):
        self.p = p
        g = p.grid
        self.ns = g.n
        v = p.v_rel_km_s * SEC_PER_YEAR  # km / yr
        vol = g.volumes_km3

        def phi(r1, r2):  # collision rate coefficient per pair, 1/yr
            sigma_km2 = np.pi * ((r1 + r2) / 1000.0) ** 2
            return sigma_km2 * v / vol

        self.phi_ss = phi(SAT.radius, SAT.radius)
        self.phi_sn = phi(SAT.radius, DEB.radius)
        self.phi_nn = phi(DEB.radius, DEB.radius)

        fs = p.frag_scale
        self.y_ss = fs * breakup_fragments(2 * SAT.mass)
        self.y_sn = fs * breakup_fragments(SAT.mass + DEB.mass)

        # Drag: velocity of altitude loss at shell centres (km/yr), mean Sun
        r_m = (R_EARTH_KM + g.centers) * 1000
        rho = atmospheric_density(g.centers)
        self._decay_sat = rho * SAT.ballistic * np.sqrt(MU * r_m) * KM_PER_YEAR_PER_M_PER_S / g.dh
        self._decay_deb = rho * DEB.ballistic * np.sqrt(MU * r_m) * KM_PER_YEAR_PER_M_PER_S / g.dh

        # Fragment spreading kernel K[i, j]: share of fragments born in j landing in i.
        c = g.centers
        K = np.exp(-0.5 * ((c[:, None] - c[None, :]) / p.frag_spread_km) ** 2)
        # Normalise against an untruncated profile so fragments thrown outside the
        # grid are lost (below: re-enter; above: leave the studied region).
        full = np.sqrt(2 * np.pi) * p.frag_spread_km / g.dh
        self.K = K / full

        self.launch = p.launch or launch_profile(g, "mega")

    # Per-shell event rates (1/yr) for state (S, D, N) at time t
    def rates(self, t, S, D, N):
        p = self.p
        sf = solar_factor(t, p.solar_amplitude)
        ra, rd = 1 - p.avoid_large, 1 - p.avoid_debris
        r = {}
        r["launch"] = self.launch(t)
        r["retire"] = S / p.mission_life
        r["drag_D"] = D * self._decay_sat * sf
        r["drag_N"] = N * self._decay_deb * sf
        # collisions (events per year)
        r["c_SS"] = 0.5 * ra * self.phi_ss * S * S
        r["c_SD"] = ra * self.phi_ss * S * D
        r["c_DD"] = 0.5 * self.phi_ss * D * D
        r["c_SN"] = rd * self.phi_sn * S * N
        r["c_DN"] = self.phi_sn * D * N
        tot = D.sum()
        if p.adr_per_year > 0 and tot > 0:
            # target the most crowded shells (risk ~ D^2)
            w = D * D
            r["adr"] = np.minimum(p.adr_per_year * w / w.sum(), D * 10)
        else:
            r["adr"] = np.zeros_like(D)
        return r

    def fragments_born(self, r):
        big = r["c_SS"] + r["c_SD"] + r["c_DD"]
        small = r["c_SN"] + r["c_DN"]
        return self.K @ (big * self.y_ss + small * self.y_sn)

    def rhs(self, t, y):
        ns = self.ns
        S, D, N = (np.maximum(y[i * ns:(i + 1) * ns], 0) for i in range(3))
        r = self.rates(t, S, D, N)
        p = self.p

        def flow(out):  # drag: leave shell i, enter shell i-1
            inflow = np.zeros_like(out)
            inflow[:-1] = out[1:]
            return inflow - out

        dS = r["launch"] - r["retire"] - 2 * r["c_SS"] - r["c_SD"] - r["c_SN"]
        dD = ((1 - p.pmd) * r["retire"] + flow(r["drag_D"])
              - r["c_SD"] - 2 * r["c_DD"] - r["c_DN"] - r["adr"])
        dN = flow(r["drag_N"]) + self.fragments_born(r) - r["c_SN"] - r["c_DN"]
        n_coll = (r["c_SS"] + r["c_SD"] + r["c_DD"] + r["c_SN"] + r["c_DN"]).sum()
        return np.concatenate([dS, dD, dN, [n_coll]])


@dataclass
class ShellRun:
    t: np.ndarray
    S: np.ndarray  # (n_t, n_shells)
    D: np.ndarray
    N: np.ndarray
    collisions: np.ndarray  # cumulative
    grid: ShellGrid

    @property
    def totals(self):
        return self.S.sum(1), self.D.sum(1), self.N.sum(1)


def simulate_shells(p: ShellParams = ShellParams(), years: float = 200.0, n_points: int = 401,
                    y0: np.ndarray | None = None, rtol: float = 1e-5) -> ShellRun:
    m = ShellModel(p)
    y0 = initial_population(p.grid) if y0 is None else y0
    t_eval = np.linspace(p.t0, p.t0 + years, n_points)
    sol = solve_ivp(m.rhs, (p.t0, p.t0 + years), np.concatenate([y0, [0.0]]),
                    method="LSODA", t_eval=t_eval, rtol=rtol, atol=1e-3)
    if not sol.success:  # pragma: no cover
        raise RuntimeError(sol.message)
    ns = m.ns
    Y = np.maximum(sol.y.T, 0)
    return ShellRun(sol.t, Y[:, :ns], Y[:, ns:2 * ns], Y[:, 2 * ns:3 * ns], Y[:, -1], p.grid)


def residence_time_years(grid: ShellGrid = ShellGrid(), obj: ObjectClass = DEB) -> np.ndarray:
    """Time spent by an object in each shell under mean solar activity."""
    r_m = (R_EARTH_KM + grid.centers) * 1000
    v = atmospheric_density(grid.centers) * obj.ballistic * np.sqrt(MU * r_m) * KM_PER_YEAR_PER_M_PER_S
    return grid.dh / v
