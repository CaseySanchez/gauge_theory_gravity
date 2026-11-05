"""Runnable demonstrations.  From the repo root:  PYTHONPATH=src python -m gtg_csta.examples.scenarios [name ...]
Names: gauge_orbits holonomy sectors schwarzschild.  Figures go to ./output/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from .. import algebra as al, cartan, conformal as cf, fields as fl, gauge, transport as tp
from .. import curvature as cv, sta_embedding as se, invariants as iv

OUT = Path(__file__).resolve().parents[3] / "output"
OUT.mkdir(exist_ok=True)


def flat_invariant(Om):
    """eta^{mu rho} eta^{nu sigma} <F_{mu nu} ~F_{rho sigma}>_0 for a connection field: two metric contractions
    of the scalar part of F ~F (a map of four spacetime inputs)."""
    F = cv.curvature(Om)

    def I(x):
        Fx = F(x)
        return (Fx * ~Fx).restrict[0].contract(1, 3).contract(1, 2).kernel.reshape(())
    return I


def gauge_orbits():
    Om, V = fl.random_connection_field(1), fl.random_versor_field(2)
    Om2 = gauge.transform_connection(Om, V)
    I1, I2 = flat_invariant(Om), flat_invariant(Om2)
    s = np.linspace(-1, 1, 21)
    pts = [al.point(jnp.array([0.1, 0.2, 0.3, 0.0]) + si * jnp.array([0.3, -0.2, 0.1, 0.4])) for si in s]
    a = np.array([float(I1(p)) for p in pts])
    b = np.array([float(I2(p)) for p in pts])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(s, a, "o-", label="I[F]"); ax[0].plot(s, b, "x--", label="I[F'] (gauge-transformed)")
    ax[0].set_xlabel("s along a line"); ax[0].set_ylabel("<F F~>"); ax[0].legend()
    ax[1].semilogy(s, np.abs(a - b) + 1e-18); ax[1].set_ylabel("|I - I'|"); ax[1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(OUT / "gauge_orbits.png", dpi=120)
    print("gauge_orbits: max |I - I'| =", np.abs(a - b).max())


def holonomy():
    x0 = jnp.array([0.1, -0.2, 0.3, 0.25])
    eps = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    Om_gen = fl.random_connection_field(6, 0.4)
    Om_flat = gauge.pure_gauge(fl.random_versor_field(5, 0.3))
    one = np.eye(1, 32)[0]
    dev = lambda Om, e: float(np.abs(np.asarray(tp.holonomy_loop(Om, x0, 0, 1, e, 30, "magnus4")) - one).max())
    g = [dev(Om_gen, e) for e in eps]
    f = [dev(Om_flat, e) for e in eps]
    slope = np.polyfit(np.log(eps ** 2), np.log(g), 1)[0]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.loglog(eps ** 2, g, "o-", label=f"generic Omega (slope {slope:.2f})")
    ax.loglog(eps ** 2, np.maximum(f, 1e-17), "s--", label="pure gauge")
    ax.set_xlabel("loop area eps^2"); ax.set_ylabel("max |T_loop - 1|"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "holonomy.png", dpi=120)
    print("holonomy: log-log slope vs area (generic) =", slope, "| pure-gauge max dev =", max(f))


def _sector_norms(F):
    return [al.sector_size(F, k) for k in "MPKD"]


def sectors():
    X = al.point([1.3, 0.4, -0.7, 0.9])
    cases = {}
    rng = np.random.default_rng(1)
    A0, A1 = rng.normal(size=(4, 6)) * .3, rng.normal(size=(4, 6, 4)) * .3
    cases["Lorentz-only random"] = cv.curvature(lambda x: se.lorentz_connection(A0 + A1 @ x.kernel))(X)
    cases["generic random"] = cv.curvature(fl.random_connection_field(3))(X)
    cof_s = cartan.painleve_gullstrand(1.0); xs = al.point([0.3, 1.7, -0.9, 1.1])
    cases["Schwarzschild, f=0"] = cv.curvature(cf.connection(cof_s))(xs)
    cof_l = cartan.static_lapse()
    cases["lapse, Lorentz only (f=0)"] = cv.curvature(cf.connection(cof_l))(X)
    cases["lapse, normal f=Schouten"] = cv.curvature(cf.connection(cof_l, cf.schouten(cof_l)))(X)
    names = list(cases); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, k in enumerate("MPKD"):
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, [max(_sector_norms(cases[n])[i], 1e-16) for n in names], w, label=k)
    ax.set_yscale("log"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("max |F component| per sector"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "sectors.png", dpi=120)
    for n in names:
        print(f"sectors: {n:30s}", dict(zip("MPKD", [f"{v:.2e}" for v in _sector_norms(cases[n])])))


def schwarzschild():
    M = 1.0
    cof = cartan.painleve_gullstrand(M)
    Om = cartan.lorentz_connection(cof)
    rs = np.linspace(2.2, 10, 25)
    K, Kexact = [], []
    for r in rs:
        x = al.point([0.0, r / np.sqrt(3), r / np.sqrt(3), r / np.sqrt(3)])
        K.append(2 * float(iv.quadratic_invariant(cof, Om)(x).kernel.reshape(())))      # I = C^2/2
        Kexact.append(48 * M ** 2 / r ** 6)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.loglog(rs, Kexact, "-", label="48 M^2 / r^6"); ax.loglog(rs, K, "o", ms=4, label="2 <F F~> (CSTA)")
    ax.set_xlabel("r"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "schwarzschild.png", dpi=120)
    print("schwarzschild: max rel err of Kretschmann =", np.max(np.abs(np.array(K) / np.array(Kexact) - 1)))


ALL = dict(gauge_orbits=gauge_orbits, holonomy=holonomy, sectors=sectors, schwarzschild=schwarzschild)

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
