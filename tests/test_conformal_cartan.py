"""Phase 7 (Tier B, items 1-3) on extensors: translation sector as coframe; normal conformal connection."""
import numpy as np
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, conformal as cf, curvature as cv
from gtg_csta._array import conformal as acf, cartan as ac

X = al.point([1.3, 0.4, -0.7, 0.9])
SPACETIMES = {
    "schwarzschild": (cartan.painleve_gullstrand(1.0), al.point([0.3, 1.7, -0.9, 1.1])),
    "frw": (cartan.frw(), X),
    "lapse": (cartan.static_lapse(), X),
}
nz = lambda e: np.asarray(e.kernel)


@pytest.mark.parametrize("name", SPACETIMES)
def test_translation_curvature_is_torsion_and_vanishes_for_levi_civita(name):
    cof, x = SPACETIMES[name]
    F = cv.curvature(cf.connection(cof))(x)
    assert al.sector_size(F, "P") < 1e-9


@pytest.mark.parametrize("name", SPACETIMES)
def test_normal_conditions_solvable_and_f_is_the_schouten_tensor(name):
    cof, x = SPACETIMES[name]
    f, res, rank = cf.solve_normal_f(cof, x)
    assert res < 1e-9 and rank == 16, "normal conditions (F_D=0, F_M trace-free) must be satisfiable"
    P = cf.schouten(cof)(x)
    assert np.abs(nz(f) - nz(P)).max() < 1e-8                    # f^a_mu = Schouten P^a_mu exactly


@pytest.mark.parametrize("name", SPACETIMES)
def test_schouten_matches_index_implementation(name):
    cof, x = SPACETIMES[name]
    acof = {"schwarzschild": ac.painleve_gullstrand(1.0), "frw": acf.frw(), "lapse": acf.static_lapse()}[name]
    Pa, Ra = acf.schouten_frame(acof, x.kernel)
    assert np.abs(nz(cf.schouten(cof)(x)) - np.asarray(Pa)).max() < 1e-10
    assert abs(float(cf.ricci_scalar(cof, x).kernel.reshape(-1)[0]) - float(Ra)) < 1e-10


def test_special_conformal_curvature_is_cotton_and_vanishes_for_conformally_flat():
    """F_K = 0 for FRW (conformally flat), != 0 for the generic static-lapse metric."""
    out = {}
    for name in ("frw", "lapse"):
        cof, x = SPACETIMES[name]
        F = cv.curvature(cf.connection(cof, cf.schouten(cof)))(x)        # f = Schouten, as a field
        out[name] = {k: al.sector_size(F, k) for k in "PKD"}
    assert out["frw"]["K"] < 1e-8 and out["frw"]["P"] < 1e-8 and out["frw"]["D"] < 1e-8
    assert out["lapse"]["P"] < 1e-8 and out["lapse"]["D"] < 1e-8
    assert out["lapse"]["K"] > 1e-3                                       # Cotton tensor nonzero


def test_lorentz_curvature_becomes_weyl_tracefree_part():
    cof, x = SPACETIMES["lapse"]
    F = cv.curvature(cf.connection(cof, cf.schouten(cof)))(x)
    E = cof(x).inverse()
    assert np.abs(nz(cf.ricci_vector(F, E))).max() < 1e-9               # trace-free
    F0 = cv.curvature(cf.connection(cof))(x)
    assert np.abs(nz(al.sector(F, "M") - al.sector(F0, "M"))).max() > 1e-3     # differs from plain Riemann


def test_schwarzschild_f_zero_and_vacuum_sectors_vanish():
    cof, x = SPACETIMES["schwarzschild"]
    f, res, _ = cf.solve_normal_f(cof, x)
    assert np.abs(nz(f)).max() < 1e-9
    F = cv.curvature(cf.connection(cof))(x)
    for k in "PKD":
        assert al.sector_size(F, k) < 1e-9
