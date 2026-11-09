"""Reproduce the 2022 TIPE results and explain the asymptotic limit."""

import numpy as np
import matplotlib.pyplot as plt

from kessler import tipe_model as tm
from kessler.plotting import COLORS, TEXT2, setup, save

setup()
p = tm.TipeParams()
fixed = tm.simulate(p)
legacy = tm.simulate(p, legacy_bug=True)
N_star, n_star = tm.equilibrium(p)
N_a, n_a = tm.analytic_equilibrium(p)
st = tm.stability(p)

print("== Modèle TIPE 2022 ==")
i, j = fixed.n.argmax(), fixed.N.argmax()
print(f"max fragments : {fixed.n[i]:.3g} en {fixed.t[i]:.0f}")
print(f"max satellites: {fixed.N[j]:.0f} en {fixed.t[j]:.0f}")
print(f"équilibre numérique : N* = {N_star:.0f}, n* = {n_star:.3g}")
print(f"formule fermée      : N* = 1/(alpha x Tn) = {N_a:.0f}, n* = alpha A Tn = {n_a:.3g}")
print(f"valeurs propres : {st['eigenvalues']}, relaxation ~ {st['relaxation_time_years']:.0f} ans")
print(f"seuil de Kessler N_c = {tm.kessler_threshold(p):.0f} ; R(2022) = {tm.reproduction_number(p.N0, p):.1f}")
print(f"écart max dû au bug de 2022 (z**2): {np.max(np.abs(fixed.n - legacy.n) / fixed.n):.2e} (relatif)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, y, ys, lab in [(axes[0], fixed.N, N_star, "Satellites intacts N"),
                       (axes[1], fixed.n, n_star, "Fragments > 1 cm n")]:
    ax.plot(fixed.t, y, color=COLORS[0], label="Solution (cycle solaire inclus)")
    ax.axhline(ys, color=COLORS[1], lw=1.5, ls="--", label="Équilibre du système moyenné")
    ax.set_title(lab)
    ax.set_xlabel("Année")
axes[1].legend(loc="lower right")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f} M"))
fig.suptitle("Modèle du TIPE 2022 : la « limite » observée est un point fixe attractif",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
fig.tight_layout()
save(fig, "01_tipe_2022")

# --- Phase portrait: every initial condition converges to the same point
setup()
fig, ax = plt.subplots(figsize=(7.5, 5.2))
rng = np.random.default_rng(1)
ics = [(p.N0, p.n0), (2 * p.N0, p.n0), (p.N0, 0.0), (p.N0, 1000 * p.n0)]
ics += [(rng.uniform(100, 12000), 10 ** rng.uniform(3, 8.3)) for _ in range(10)]
for k, (N0, n0) in enumerate(ics):
    tr = tm.simulate(p, t_end=2900, n_points=3000, y0=[N0, n0], seasonal=False)
    ax.plot(tr.N, np.maximum(tr.n, 1), color=COLORS[0] if k < 4 else "#9ec5f4",
            lw=1.8 if k < 4 else 1.0, zorder=3 if k < 4 else 2)
    ax.plot(N0, max(n0, 1), "o", ms=5, color=COLORS[0] if k < 4 else "#9ec5f4")
Nc = tm.kessler_threshold(p)
ax.axvline(Nc, color=COLORS[1], lw=1.5, ls="--")
ax.text(Nc * 1.05, 2e3, f"Seuil de Kessler\nN_c = 1/(α x Tₙ) ≈ {Nc:.0f}", color=TEXT2, fontsize=9)
ax.plot(N_star, n_star, "*", ms=16, color=COLORS[1], mec="white", zorder=5)
ax.annotate("Point fixe unique\n(stable, foyer)", (N_star, n_star), (3500, 2e8), color=TEXT2,
            fontsize=9, arrowprops=dict(arrowstyle="->", color=TEXT2))
ax.set_yscale("log")
ax.set_ylim(1e3, 1e9)
ax.set_xlabel("Satellites intacts N")
ax.set_ylabel("Fragments n")
ax.set_title("Portrait de phase : toutes les trajectoires convergent vers le même point")
save(fig, "02_portrait_de_phase")

# --- Reproduction number along the 2022 trajectory
setup()
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(fixed.t, tm.reproduction_number(fixed.N, p), color=COLORS[0])
ax.axhline(1, color=COLORS[1], lw=1.5, ls="--")
ax.text(2600, 1.25, "R = 1", color=TEXT2, fontsize=9)
ax.set_title("Nombre de reproduction des fragments R(t) = α x N(t) Tₙ")
ax.set_xlabel("Année")
save(fig, "03_nombre_de_reproduction")
