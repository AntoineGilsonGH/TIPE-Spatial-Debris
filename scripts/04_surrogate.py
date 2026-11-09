"""Machine learning #2: neural surrogate, Sobol sensitivity, policy map.

The dataset (1500 runs of the 36-shell model) is cached in data/.
Run with --rebuild to regenerate it (~2-4 min on 8 cores).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kessler.ml.surrogate import (FACTORS, OUTPUTS, build_dataset, sobol_indices,
                                  train_surrogate)
from kessler.plotting import COLORS, TEXT2, setup, save
from kessler.shells import initial_population, ShellGrid

DATA = Path(__file__).resolve().parents[1] / "data" / "surrogate_dataset.npz"

if __name__ == "__main__":
    if DATA.exists() and "--rebuild" not in sys.argv:
        d = np.load(DATA)
        U, X, Y = d["U"], d["X"], d["Y"]
    else:
        print("Simulation de 1500 futurs possibles...")
        U, X, Y = build_dataset(1500, seed=0)
        DATA.parent.mkdir(exist_ok=True)
        np.savez_compressed(DATA, U=U, X=X, Y=Y)

    predict, info = train_surrogate(U, Y)
    print("== Surrogate neuronal ==")
    for name, r2 in zip(OUTPUTS, info["r2"]):
        print(f"  R² test ({name}) = {r2:.4f}")

    # ---- parity plot
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for k, ax in enumerate(axes):
        yt, yp = info["Y_test"][:, k], info["Y_pred"][:, k]
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, color=TEXT2, lw=1, ls="--")
        ax.plot(yt, yp, "o", ms=4, color=COLORS[0], alpha=0.6, mec="none")
        ax.set_title(f"{OUTPUTS[k]}  (R² = {info['r2'][k]:.3f})", fontsize=10)
        ax.set_xlabel("Simulation physique")
        ax.set_ylabel("Réseau de neurones")
    fig.suptitle("Le réseau de neurones remplace le simulateur (300 simulations jamais vues)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, "09_surrogate_parite")

    # ---- Sobol indices (2^15 x 8 = 260k surrogate evaluations, ~1 s)
    S1, ST = sobol_indices(predict, len(FACTORS), n=2**15)
    print("\n== Indices de Sobol (débris en 2226) ==")
    for f, s1, st in zip(FACTORS, S1[:, 0], ST[:, 0]):
        print(f"  {f.label:42s} S1 = {s1:5.2f}   ST = {st:5.2f}")
    setup()
    order = np.argsort(ST[:, 0])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    yy = np.arange(len(FACTORS))
    ax.barh(yy + 0.2, ST[order, 0], height=0.38, color=COLORS[0], label="Effet total (S_T)")
    ax.barh(yy - 0.2, np.clip(S1[order, 0], 0, None), height=0.38, color=COLORS[1],
            label="Effet seul (S₁)")
    ax.set_yticks(yy, [FACTORS[i].label for i in order])
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Qu'est-ce qui détermine le nombre de débris en 2226 ? (indices de Sobol)")
    save(fig, "10_sobol")

    # ---- Policy map: launch traffic x PMD, other factors at nominal
    nominal = {"avoid_debris": 0.8, "adr_per_year": 0.0, "frag_scale": 1.0, "solar_amplitude": 0.7}
    def to_unit(f, v):
        if f.log:
            return (np.log(v) - np.log(f.low)) / (np.log(f.high) - np.log(f.low))
        return (v - f.low) / (f.high - f.low)
    L = np.geomspace(0.1, 3.0, 160)
    P = np.linspace(0.5, 0.99, 160)
    LL, PP = np.meshgrid(L, P)
    grid_u = np.zeros((LL.size, len(FACTORS)))
    for i, f in enumerate(FACTORS):
        if f.name == "launch_mult":
            grid_u[:, i] = to_unit(f, LL.ravel())
        elif f.name == "pmd":
            grid_u[:, i] = to_unit(f, PP.ravel())
        else:
            grid_u[:, i] = to_unit(f, nominal[f.name])
    Z = predict(grid_u)[:, 0].reshape(LL.shape)
    n_today = initial_population(ShellGrid())[-ShellGrid().n:].sum()
    setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(LL * 2450, PP * 100, 10**Z / 1e3, levels=np.geomspace(5, 400, 13),
                     cmap="Blues", extend="both", norm=plt.matplotlib.colors.LogNorm())
    cb = fig.colorbar(cs, ax=ax, pad=0.01, format="%.0f")
    cb.set_label("Débris > 10 cm en 2226 (milliers)", color=TEXT2)
    c0 = ax.contour(LL * 2450, PP * 100, Z, levels=[np.log10(n_today)], colors=COLORS[1],
                    linewidths=2.2)
    ax.clabel(c0, fmt={c0.levels[0]: "population actuelle"}, fontsize=9, colors=COLORS[1])
    ax.set_xscale("log")
    ax.set_xticks([300, 500, 1000, 2000, 5000], ["300", "500", "1 000", "2 000", "5 000"])
    ax.minorticks_off()
    ax.set_xlabel("Satellites lancés par an")
    ax.set_ylabel("Taux de désorbitation en fin de vie (%)")
    ax.grid(False)
    ax.set_title("Carte de décision : quel trafic l'orbite basse peut-elle supporter ?")
    save(fig, "11_carte_politique")
