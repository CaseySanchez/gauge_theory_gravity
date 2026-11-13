"""Phase 7 (Tier B, items 1-3) on extensors: translation sector as coframe; normal conformal connection."""
import numpy as np
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, conformal as cf, curvature as cv
from gtg_csta._array import conformal as acf, cartan as ac

X = al.point([1.3, 0.4, -0.7, 0.9])
SPACETIMES = {
    "schwarzschild": (cartan.schwarzschild(1.0), al.point([0.3, 1.7, -0.9, 1.1]), ac.painleve_gullstrand(1.0)),
    "frw": (cartan.friedmann(), X, acf.frw()),
    "lapse": (cartan.static_lapse(), X, acf.static_lapse()),
}
nz = lambda e: np.asarray(e.kernel)


@pytest.mark.parametrize("name", SPACETIMES)
def test_coframe_as_identity_plus_dyad_matches_index_implementation(name):
    coframe, x, reference = SPACETIMES[name]
    assert np.abs(nz(coframe(x)) - np.asarray(reference(x.kernel))).max() < 1e-12


@pytest.mark.parametrize("name", SPACETIMES)
def test_translation_curvature_is_torsion_and_vanishes_for_levi_civita(name):
    coframe, x, _ = SPACETIMES[name]
    assert al.sector_size(cv.curvature(cf.connection(coframe))(x), "P") < 1e-9


@pytest.mark.parametrize("name", SPACETIMES)
def test_normal_conditions_solvable_and_f_is_the_schouten_tensor(name):
    coframe, x, _ = SPACETIMES[name]
    f, residual, rank = cf.solve_normal_f(coframe, x)
    assert residual < 1e-9 and rank == 16, "normal conditions (F_D=0, F_M trace-free) must be satisfiable"
    assert np.abs(nz(f) - nz(cf.schouten(coframe)(x))).max() < 1e-8              # f^a_mu = Schouten P^a_mu exactly


@pytest.mark.parametrize("name", SPACETIMES)
def test_schouten_matches_index_implementation(name):
    coframe, x, reference = SPACETIMES[name]
    schouten, scalar = acf.schouten_frame(reference, x.kernel)
    assert np.abs(nz(cf.schouten(coframe)(x)) - np.asarray(schouten)).max() < 1e-10
    assert abs(al.scalar(cf.ricci_scalar(coframe, x)) - float(scalar)) < 1e-10


def test_special_conformal_curvature_is_cotton_and_vanishes_for_conformally_flat():
    """F_K = 0 for FRW (conformally flat), != 0 for the generic static-lapse metric."""
    sizes = {}
    for name in ("frw", "lapse"):
        coframe, x, _ = SPACETIMES[name]
        strength = cv.curvature(cf.connection(coframe, cf.schouten(coframe)))(x)       # f = Schouten, as a field
        sizes[name] = {sector: al.sector_size(strength, sector) for sector in "PKD"}
    assert sizes["frw"]["K"] < 1e-8 and sizes["frw"]["P"] < 1e-8 and sizes["frw"]["D"] < 1e-8
    assert sizes["lapse"]["P"] < 1e-8 and sizes["lapse"]["D"] < 1e-8
    assert sizes["lapse"]["K"] > 1e-3                                                  # Cotton tensor nonzero


def test_lorentz_curvature_becomes_weyl_tracefree_part():
    coframe, x, _ = SPACETIMES["lapse"]
    strength = cv.curvature(cf.connection(coframe, cf.schouten(coframe)))(x)
    assert np.abs(nz(cf.ricci_vector(strength, coframe(x).inverse()))).max() < 1e-9       # trace-free
    plain = cv.curvature(cf.connection(coframe))(x)
    assert np.abs(nz(al.sector(strength, "M") - al.sector(plain, "M"))).max() > 1e-3     # differs from plain Riemann


def test_schwarzschild_f_zero_and_vacuum_sectors_vanish():
    coframe, x, _ = SPACETIMES["schwarzschild"]
    f, _, _ = cf.solve_normal_f(coframe, x)
    assert np.abs(nz(f)).max() < 1e-9
    strength = cv.curvature(cf.connection(coframe))(x)
    for sector in "PKD":
        assert al.sector_size(strength, sector) < 1e-9
