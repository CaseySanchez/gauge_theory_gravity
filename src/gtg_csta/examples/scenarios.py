"""Runnable demonstrations.  From the repo root:  PYTHONPATH=src python -m gtg_csta.examples.scenarios [name ...]
Names: gauge_orbits holonomy sectors schwarzschild.  Figures go to ./output/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numga.extensor import Extensor

from .. import algebra as al, cartan, conformal as cf, fields as fl, gauge, transport as tp
from .. import curvature as cv, sta_embedding as se, invariants as iv
from ..fields import Field

OUT = Path(__file__).resolve().parents[3] / "output"
OUT.mkdir(exist_ok=True)


def flat_invariant(omega: Field) -> Field:
    """eta^{mu rho} eta^{nu sigma} <F_{mu nu} ~F_{rho sigma}>_0 for a connection field: two metric contractions of
    the scalar part of F ~F (a map of four spacetime inputs)."""
    strength = cv.curvature(omega)

    def invariant(x: Extensor) -> Extensor:
        curved = strength(x)
        return (curved * ~curved).restrict[0].contract(1, 3).contract(1, 2)           # [] Scalar+

    return invariant


def gauge_orbits() -> None:
    omega, versor = fl.random_connection_field(1), fl.random_versor_field(2)
    transformed = gauge.transform_connection(omega, versor)
    before, after = flat_invariant(omega), flat_invariant(transformed)
    line = np.linspace(-1, 1, 21)
    points = [al.point(np.array([0.1, 0.2, 0.3, 0.0]) + s * np.array([0.3, -0.2, 0.1, 0.4])) for s in line]
    a = np.array([al.scalar(before(p)) for p in points])
    b = np.array([al.scalar(after(p)) for p in points])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(line, a, "o-", label="I[F]"); ax[0].plot(line, b, "x--", label="I[F'] (gauge-transformed)")
    ax[0].set_xlabel("s along a line"); ax[0].set_ylabel("<F F~>"); ax[0].legend()
    ax[1].semilogy(line, np.abs(a - b) + 1e-18); ax[1].set_ylabel("|I - I'|"); ax[1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(OUT / "gauge_orbits.png", dpi=120)
    print("gauge_orbits: max |I - I'| =", np.abs(a - b).max())


def holonomy_deviation(omega: Field, start: Extensor, size: float) -> float:
    """max |T_loop - 1| around a size x size loop in the (t, x) plane."""
    loop = tp.holonomy_loop(omega, start, 0, 1, size, 30, "magnus4")
    return float(np.abs(np.asarray((loop - 1).kernel)).max())


def holonomy() -> None:
    start = al.point([0.1, -0.2, 0.3, 0.25])
    sizes = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    generic = [holonomy_deviation(fl.random_connection_field(6, 0.4), start, size) for size in sizes]
    flat = [holonomy_deviation(gauge.pure_gauge(fl.random_versor_field(5, 0.3)), start, size) for size in sizes]
    slope = np.polyfit(np.log(sizes ** 2), np.log(generic), 1)[0]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.loglog(sizes ** 2, generic, "o-", label=f"generic Omega (slope {slope:.2f})")
    ax.loglog(sizes ** 2, np.maximum(flat, 1e-17), "s--", label="pure gauge")
    ax.set_xlabel("loop area eps^2"); ax.set_ylabel("max |T_loop - 1|"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "holonomy.png", dpi=120)
    print("holonomy: log-log slope vs area (generic) =", slope, "| pure-gauge max dev =", max(flat))


def sector_sizes(strength: Extensor) -> list[float]:
    return [al.sector_size(strength, name) for name in "MPKD"]


def sectors() -> None:
    x = al.point([1.3, 0.4, -0.7, 0.9])
    lorentz = fl.polynomial(se.LorentzConnection, 1)

    def lorentz_only(position: Extensor) -> Extensor:
        return se.lorentz_embedding(lorentz(position))

    schwarzschild, lapse = cartan.schwarzschild(1.0), cartan.static_lapse()
    cases = {
        "Lorentz-only random": cv.curvature(lorentz_only)(x),
        "generic random": cv.curvature(fl.random_connection_field(3))(x),
        "Schwarzschild, f=0": cv.curvature(cf.connection(schwarzschild))(al.point([0.3, 1.7, -0.9, 1.1])),
        "lapse, Lorentz only (f=0)": cv.curvature(cf.connection(lapse))(x),
        "lapse, normal f=Schouten": cv.curvature(cf.connection(lapse, cf.schouten(lapse)))(x),
    }
    names = list(cases)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, name in enumerate("MPKD"):
        ax.bar(np.arange(len(names)) + (i - 1.5) * width,
               [max(sector_sizes(cases[case])[i], 1e-16) for case in names], width, label=name)
    ax.set_yscale("log"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("max |F component| per sector"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "sectors.png", dpi=120)
    for case in names:
        print(f"sectors: {case:30s}", dict(zip("MPKD", [f"{v:.2e}" for v in sector_sizes(cases[case])])))


def schwarzschild() -> None:
    mass = 1.0
    coframe = cartan.schwarzschild(mass)
    density = iv.quadratic_invariant(coframe, cartan.lorentz_connection(coframe))
    radii = np.linspace(2.2, 10, 25)
    measured, exact = [], []
    for r in radii:
        x = al.point([0.0, r / np.sqrt(3), r / np.sqrt(3), r / np.sqrt(3)])
        measured.append(2 * al.scalar(density(x)))                                     # I = C^2/2
        exact.append(48 * mass ** 2 / r ** 6)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.loglog(radii, exact, "-", label="48 M^2 / r^6"); ax.loglog(radii, measured, "o", ms=4, label="2 <F F~> (CSTA)")
    ax.set_xlabel("r"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "schwarzschild.png", dpi=120)
    print("schwarzschild: max rel err of Kretschmann =", np.max(np.abs(np.array(measured) / np.array(exact) - 1)))


ALL = dict(gauge_orbits=gauge_orbits, holonomy=holonomy, sectors=sectors, schwarzschild=schwarzschild)

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
