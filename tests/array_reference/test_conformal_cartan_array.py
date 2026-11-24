"""Phase 7 (Tier B, items 1-3): translation sector as coframe; normal conformal connection."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al
from gauge_theory_gravity._array import cartan, conformal as cf, curvature as cv

X = jnp.array([1.3, 0.4, -0.7, 0.9])
SPACETIMES = {
    "schwarzschild": (cartan.painleve_gullstrand(1.0), jnp.array([0.3, 1.7, -0.9, 1.1])),
    "frw": (cf.frw(), X),
    "lapse": (cf.static_lapse(), X),
}


@pytest.mark.parametrize("name", SPACETIMES)
def test_translation_curvature_is_torsion_and_vanishes_for_levi_civita(name):
    cof, x = SPACETIMES[name]
    F = cv.curvature(cf.connection(cof))(x)
    assert np.abs(np.asarray(cf.sectors(F)["P"])).max() < 1e-9


@pytest.mark.parametrize("name", SPACETIMES)
def test_normal_conditions_solvable_and_f_is_proportional_to_schouten(name):
    cof, x = SPACETIMES[name]
    f, res, rank = cf.solve_normal_f(cof, x)
    assert res < 1e-9, "normal conditions (F_D=0, F_M trace-free) must be satisfiable"
    S, _ = cf.schouten_frame(cof, x)
    f, S = np.asarray(f), np.asarray(S)
    if np.abs(S).max() < 1e-10:                          # Ricci-flat: f = 0
        assert np.abs(f).max() < 1e-9
    else:
        ratio = (f * S).sum() / (S * S).sum()
        assert np.abs(f - ratio * S).max() < 1e-8
        assert abs(ratio - 1.0) < 1e-8          # f^a_mu = Schouten P^a_mu exactly, in this normalization


def test_schouten_proportionality_constant_is_universal():
    r = []
    for name in ("frw", "lapse"):
        cof, x = SPACETIMES[name]
        f, *_ = cf.solve_normal_f(cof, x)
        S, _ = cf.schouten_frame(cof, x)
        f, S = np.asarray(f), np.asarray(S)
        r.append((f * S).sum() / (S * S).sum())
    assert abs(r[0] - r[1]) < 1e-8


def test_special_conformal_curvature_is_cotton_and_vanishes_for_conformally_flat():
    """F_K = 0 for FRW (conformally flat), != 0 for the generic static-lapse metric."""
    out = {}
    for name in ("frw", "lapse"):
        cof, x = SPACETIMES[name]
        fn = lambda xx, cof=cof: cf.schouten_frame(cof, xx)[0]      # f = Schouten, as a field
        F = cv.curvature(cf.connection(cof, fn))(x)
        sec = cf.sectors(F)
        out[name] = {k: float(np.abs(np.asarray(v)).max()) for k, v in sec.items()}
    assert out["frw"]["K"] < 1e-8 and out["frw"]["P"] < 1e-8 and out["frw"]["D"] < 1e-8
    assert out["lapse"]["P"] < 1e-8 and out["lapse"]["D"] < 1e-8
    assert out["lapse"]["K"] > 1e-3                       # Cotton tensor nonzero


def test_lorentz_curvature_becomes_weyl_tracefree_part():
    cof, x = SPACETIMES["lapse"]
    f, *_ = cf.solve_normal_f(cof, x)
    F = cv.curvature(cf.connection(cof, lambda xx: f))(x)
    FM = cf.lorentz_full(cf.sectors(F)["M"])
    E = jnp.linalg.inv(cof(x))
    assert np.abs(np.asarray(jnp.einsum("mnab,ma->nb", FM, E))).max() < 1e-9
    # and it differs from the plain Riemann bivector (Ricci part removed)
    F0 = cv.curvature(cf.connection(cof))(x)
    FM0 = cf.lorentz_full(cf.sectors(F0)["M"])
    assert np.abs(np.asarray(FM - FM0)).max() > 1e-3


def test_schwarzschild_f_zero_and_weyl_equals_riemann():
    cof, x = SPACETIMES["schwarzschild"]
    f, res, _ = cf.solve_normal_f(cof, x)
    assert np.abs(np.asarray(f)).max() < 1e-9
    F = cv.curvature(cf.connection(cof, lambda xx: jnp.zeros((4, 4))))(x)
    sec = cf.sectors(F)
    assert np.abs(np.asarray(sec["P"])).max() < 1e-9
    assert np.abs(np.asarray(sec["K"])).max() < 1e-9      # Cotton = 0 in vacuum
    assert np.abs(np.asarray(sec["D"])).max() < 1e-9
