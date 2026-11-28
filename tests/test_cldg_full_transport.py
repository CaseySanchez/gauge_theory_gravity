"""Phase 2 of docs/PLAN_DIRAC_SOLUTION.md: the self-consistent Dirac cosmology under the CSTA's full twistor transport,
the target coupling. A self-gravitating Dirac solution of the CSTA itself, not of GTG embedded in it.

The action is S1 + beta S2 + gamma S3 + 4 kappa |det e| L_D with which = "full", kappa = 8 pi, gamma = Lambda/6 + 1/beta.
Three things change against GTG's Lorentz coupling (`test_cldg_csta`):

  * the special-conformal rows are sourced by the tensor bilinear C, so f is the Dirac-shifted field
    F = (1/beta)(1 + kappa tr C / 3) - (kappa/beta) C^T (`test_spin_torsion`). For this spinor C/rho is constant:
    (sqrt 2 / 4) sigma on the diagonal and a tensor part in the gamma_0 gamma_3 plane, which the phase chi (a rotation
    in gamma_1 gamma_2) leaves alone; with the boost it rotates with chi but stays a function of t alone;
  * the spinor equation stays LDG's, with the mass m - sqrt 2 (1 + 1/beta) + kappa sigma / (12 beta): the compensator
    mass sqrt 2 (1 + tr F / 4) and the tensor coupling -F_03 / sqrt 2;
  * eliminating f leaves 24/beta (inside Lambda), 4 kappa sqrt 2 sigma / beta (inside the mass) and the contact term
    (kappa / (2 beta)) ((tr C)^2 / 3 - C:C) = -kappa sigma^2 / (24 beta). The Friedmann constraint becomes
    3 H^2 = Lambda + 12 pi^2 a rho^2 + 8 pi m0 rho cos beta, a = 1 + 2/(9 beta), m0 = m - sqrt 2 (1 + 1/beta).

LDG's contorsion is unchanged, and the thesis's closed forms carry over with m0 and sqrt a (`cldg_solution`). All 60
connection rows and the 8 spinor rows vanish; with the Lorentz-coupled solution's H, rho, chi instead, the translation
rows and the spinor rows do not.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, dirac as dr, first_order as fo
from cldg_solution import KAPPA, Cosmology

CASES = {
    "particle": (4.0, dict(sign=1), [0.4, 0.3, -0.2, 0.5]),
    "antiparticle": (2.0, dict(sign=-1), [0.25, -0.6, 0.4, 0.1]),
    "boosted": (-1.0, dict(sign=1, xi=0.4, eta=0.7), [0.6, 0.2, 0.7, -0.4]),
}


def rows(cosmology, beta, x4):
    """The connection rows of the total action, [15, 4], the matter part alone, and the spinor rows, [8]."""
    gamma = cosmology.cosmological / 6 + 1 / beta
    omega = cosmology.csta_connection(cosmology.shifted_special(beta))
    gravity = fo.euler_lagrange_connection(omega, x4)
    matter = 4 * KAPPA * dr.euler_lagrange_connection(omega, cosmology.csta_spinor, x4, mass=cosmology.mass,
                                                      which="full")
    spinor = dr.euler_lagrange_spinor(omega, cosmology.csta_spinor, x4, mass=cosmology.mass, which="full")
    return gravity[0] + beta * gravity[1] + gamma * gravity[2] + matter, matter, spinor


def test_special_conformal_current_is_homogeneous():
    for t in (0.3, 0.6, 1.2):
        Z = dr.from_sta(Cosmology(sign=1).spinor_multivector([t, 0.0, 0.0, 0.0]))
        current = np.asarray(dr.special_conformal_current(Z).kernel).reshape(4, 4)
        rho = float(Cosmology(sign=1).density(t))
        expected = np.sqrt(2) / 4 * (np.eye(4) - np.eye(4)[[3, 1, 2, 0]] * np.array([1, 0, 0, 1])[:, None])
        assert np.abs(current / rho - expected).max() < 1e-12


@pytest.mark.parametrize("name", list(CASES))
def test_solves_every_csta_equation_under_the_full_transport(name):
    beta, options, x4 = CASES[name]
    cosmology = Cosmology(coupling=beta, **options)
    total, matter, spinor = rows(cosmology, beta, x4)
    for sector in "MPK":
        assert np.abs(matter[al.SECTORS[sector]]).max() > 0.1
    assert np.abs(total).max() < 1e-12
    assert np.abs(spinor).max() < 1e-12


def test_lorentz_coupled_solution_is_not():
    beta, options, x4 = CASES["particle"]
    total, _, spinor = rows(Cosmology(**options), beta, x4)
    assert np.abs(total[al.SECTORS["P"]]).max() > 1.0
    assert np.abs(spinor).max() > 0.1
