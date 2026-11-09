"""Multi-shell model: scenarios, altitude structure, Monte Carlo."""

import numpy as np
import matplotlib.pyplot as plt

from kessler.plotting import COLORS, TEXT2, SEQ, setup, save
from kessler.shells import (DEB, SAT, ShellGrid, ShellParams, launch_profile,
                            residence_time_years, simulate_shells)
from kessler.stochastic import simulate_mc

setup()
g = ShellGrid()
SCEN = {
    "Arrêt total des lancements": ShellParams(launch=launch_profile(g, "no_launch")),
    "Rythme « historique » (~150 sat/an)": ShellParams(launch=launch_profile(g, "legacy")),
    "Méga-constellations (~2 450 sat/an), PMD 90 %": ShellParams(launch=launch_profile(g, "mega")),
    "Méga-constellations, PMD 99 % + ADR 10/an": ShellParams(
        launch=launch_profile(g, "mega"), pmd=0.99, adr_per_year=10),
}
runs = {k: simulate_shells(p) for k, p in SCEN.items()}

print("== Modèle multi-couches (populations finales en 2226) ==")
for k, r in runs.items():
    S, D, N = r.totals
    print(f"{k:50s} S={S[-1]:7.0f} D={D[-1]:7.0f} N={N[-1]:8.0f} collisions={r.collisions[-1]:6.1f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for c, (k, r) in zip(COLORS, runs.items()):
    for ax, y in zip(axes, r.totals):
        ax.plot(r.t, y, color=c, label=k)
for ax, title in zip(axes, ["Satellites actifs S", "Objets abandonnés D", "Débris > 10 cm"]):
    ax.set_title(title)
    ax.set_xlabel("Année")
axes[2].set_yscale("log")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
fig.suptitle("Modèle à 36 couches d'altitude (200-2000 km) : quatre politiques spatiales",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
fig.tight_layout()
save(fig, "04_scenarios_multicouches")

# --- Altitude x time map of the debris density (mega scenario)
setup()
r = runs["Méga-constellations (~2 450 sat/an), PMD 90 %"]
dens = r.N / g.volumes_km3 * 1e9  # objects per 10^9 km^3
fig, ax = plt.subplots(figsize=(9, 4.2))
mesh = ax.pcolormesh(r.t, g.centers, np.log10(np.maximum(dens.T, 1e-2)), cmap=SEQ,
                     shading="auto", vmin=0, vmax=np.log10(dens.max()))
cb = fig.colorbar(mesh, ax=ax, pad=0.01)
cb.set_label("log₁₀ densité de débris (objets / 10⁹ km³)", color=TEXT2)
ax.grid(False)
ax.set_xlabel("Année")
ax.set_ylabel("Altitude (km)")
ax.set_title("Où s'accumulent les débris ? (scénario méga-constellations, PMD 90 %)")
save(fig, "05_carte_altitude_temps")

# --- Physical ingredient: drag lifetime vs altitude
setup()
fig, ax = plt.subplots(figsize=(7, 3.6))
for c, obj, lab in [(COLORS[0], SAT, "Satellite (A/m = 0,012 m²/kg)"),
                    (COLORS[1], DEB, "Débris 10 cm (A/m = 0,05 m²/kg)")]:
    cum = np.cumsum(residence_time_years(g, obj))  # time to fall to 200 km
    ax.plot(g.edges[1:], cum, color=c, label=lab)
ax.set_yscale("log")
ax.set_xlim(200, 1500)
ax.set_ylim(1e-2, 1e5)
ax.set_xlabel("Altitude initiale (km)")
ax.set_ylabel("Temps avant rentrée (années)")
ax.set_title("Nettoyage naturel par l'atmosphère (activité solaire moyenne)")
ax.legend()
save(fig, "06_duree_de_vie_altitude")

# --- Monte Carlo: uncertainty due to the randomness of collisions
setup()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
for ax, (k, c) in zip(axes, [("Rythme « historique » (~150 sat/an)", COLORS[0]),
                             ("Méga-constellations (~2 450 sat/an), PMD 90 %", COLORS[1])]):
    t, S, D, N, C = simulate_mc(SCEN[k], n_runs=300, seed=42)
    lo, q25, med, q75, hi = np.percentile(N, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0, label="90 % des tirages")
    ax.fill_between(t, q25, q75, color=c, alpha=0.35, lw=0, label="50 % des tirages")
    for i in range(5):
        ax.plot(t, N[i], color=c, lw=0.7, alpha=0.8)
    ax.plot(t, med, color=c, lw=2.2, label="Médiane")
    ode = runs[k]
    ax.plot(ode.t, ode.N.sum(1), color=TEXT2, lw=1.5, ls="--", label="Modèle déterministe (ODE)")
    ax.set_title(k)
    ax.set_xlabel("Année")
    print(f"MC {k}: débris 2226 médiane {med[-1]:.0f}, [5%, 95%] = [{lo[-1]:.0f}, {hi[-1]:.0f}]")
axes[0].set_ylabel("Débris > 10 cm")
axes[0].legend(loc="upper left")
fig.suptitle("Monte Carlo (300 histoires, collisions tirées selon une loi de Poisson)",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
fig.tight_layout()
save(fig, "07_monte_carlo")
