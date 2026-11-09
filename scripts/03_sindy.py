"""Machine learning #1: re-discover the 2022 equations from noisy data (SINDy)."""

import numpy as np
import matplotlib.pyplot as plt

from kessler import tipe_model as tm
from kessler.ml.sindy import SINDy, smooth_derivative
from kessler.plotting import COLORS, TEXT2, setup, save
from scipy.integrate import solve_ivp

setup()
p = tm.TipeParams()
rng = np.random.default_rng(0)
NOISE = 0.01  # 1 % multiplicative measurement noise
SN, Sn = 1e3, 1e7  # units: thousands of satellites, tens of millions of fragments

ics = [(4960, 1.1e5), (500, 1e6), (3000, 8e7), (8000, 2e7), (200, 1e8), (1500, 5e5)]
X_all, dX_all, raw = [], [], []
for N0, n0 in ics:
    tr = tm.simulate(p, t_end=2222, n_points=2001, y0=[N0, n0], seasonal=False)
    X = np.c_[tr.N / SN, tr.n / Sn]
    Xn = X * (1 + NOISE * rng.standard_normal(X.shape))
    xs, dx = smooth_derivative(Xn, tr.t[1] - tr.t[0], window=51)
    X_all.append(xs[30:-30])
    dX_all.append(dx[30:-30])
    raw.append((tr.t, Xn))

model = SINDy(["N", "n"], degree=2, threshold=0.02).fit(np.vstack(X_all), np.vstack(dX_all))
print("== SINDy : équations découvertes (N en milliers, n en dizaines de millions) ==")
for eq in model.equations():
    print("  ", eq)

truth = {  # dominant true coefficients in the same units
    ("N", "1"): p.a / SN, ("N", "N n"): -p.x * Sn,
    ("n", "1"): p.beta * p.a / Sn, ("n", "n"): -1 / p.eff_frag_lifetime,
    ("n", "N n"): p.alpha * p.x * SN,
}
print("\n  terme            vrai          SINDy       erreur")
rows = []
for (var, term), v in truth.items():
    j = ["N", "n"].index(var)
    est = model.coef_[model.labels.index(term), j]
    err = abs(est - v) / abs(v)
    rows.append((f"d{var}/dt : {term}", v, est))
    print(f"  d{var}/dt {term:6s} {v:+.4e}  {est:+.4e}  {err:6.1%}")

# Integrate the discovered model from a new, unseen initial condition
def learned(t, s):
    return model.predict(s[None, :])[0]

N0, n0 = 6500, 3e6
true_tr = tm.simulate(p, t_end=2422, n_points=800, y0=[N0, n0], seasonal=False)
sol = solve_ivp(learned, (p.t0, 2422), [N0 / SN, n0 / Sn], t_eval=true_tr.t, rtol=1e-8)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, k, y_true, scale, title in [(axes[0], 0, true_tr.N, SN, "Satellites N"),
                                    (axes[1], 1, true_tr.n, Sn, "Fragments n")]:
    for t, Xn in raw[:3]:
        ax.plot(t, Xn[:, k] * scale, ".", ms=1.2, color="#b7d3f6", zorder=1)
    ax.plot(true_tr.t, y_true, color=COLORS[0], label="Vrai modèle (condition initiale jamais vue)")
    ax.plot(sol.t, sol.y[k] * scale, color=COLORS[1], ls="--", label="Modèle découvert par SINDy")
    ax.set_title(title)
    ax.set_xlabel("Année")
axes[0].plot([], [], ".", color="#b7d3f6", label="Données d'entraînement bruitées")
axes[0].legend(loc="upper right")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f} M"))
fig.suptitle("SINDy retrouve les équations différentielles à partir de données bruitées",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
fig.tight_layout()
save(fig, "08_sindy")
