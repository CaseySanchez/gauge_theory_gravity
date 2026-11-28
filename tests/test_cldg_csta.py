"""Phase 1 of docs/PLAN_DIRAC_SOLUTION.md: the self-consistent Dirac cosmology of Challinor, Lasenby, Doran and Gull
(`cldg_solution`) in the CSTA, with GTG's Lorentz coupling. The first configuration in the repo that solves the coupled
gravity-Dirac system.

The action is S1 + beta S2 + gamma S3 + 4 kappa |det e| L_D (`test_spin_torsion`) with kappa = 8 pi, f = e / beta and
gamma fixed by Lambda = 6 (gamma - 1/beta) (`test_first_order`), Lambda the thesis's (repulsive, 3 H^2 = Lambda for de
Sitter; in LDG's density it enters as +Lambda det(h)^-1, `gtg_dirac_reference.action_density`). With the Newtonian-gauge
coframe, Levi-Civita plus LDG's contorsion 1/2 kappa g_mu . S, and Z = dirac.from_sta(psi):

  * all 60 connection rows (Lorentz, translation, special-conformal, dilation) and the 8 spinor rows vanish, for the
    particle and antiparticle branches, with and without the constant boost, at several events and for several beta;
  * without the contorsion the Lorentz and translation rows and the spinor rows fail: the solution needs the torsion.

This also checks two normalizations independently of PLAN_DIRAC: the coupling 4 kappa and the Lambda identification.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, dirac as dr, first_order as fo
from cldg_solution import KAPPA, Cosmology

CASES = {
    "particle": (Cosmology(sign=1), 4.0, [0.4, 0.3, -0.2, 0.5]),
    "antiparticle": (Cosmology(sign=-1), 2.0, [0.25, -0.6, 0.4, 0.1]),
    "boosted": (Cosmology(sign=1, xi=0.4, eta=0.7), -1.0, [0.6, 0.2, 0.7, -0.4]),
}


def rows(cosmology, beta, x4, torsion=1.0):
    """The connection rows of the total action, [15, 4], the matter part alone, and the spinor rows, [8]."""
    gamma = cosmology.cosmological / 6 + 1 / beta
    omega = cosmology.csta_connection(lambda q: cosmology.geometry()(q) / beta, torsion)
    gravity = fo.euler_lagrange_connection(omega, x4)
    matter = 4 * KAPPA * dr.euler_lagrange_connection(omega, cosmology.csta_spinor, x4, mass=cosmology.mass,
                                                      which="lorentz")
    spinor = dr.euler_lagrange_spinor(omega, cosmology.csta_spinor, x4, mass=cosmology.mass, which="lorentz")
    return gravity[0] + beta * gravity[1] + gamma * gravity[2] + matter, matter, spinor


def test_coframe_is_the_oracles():
    cosmology, _, x4 = CASES["particle"]
    csta = np.asarray(al.components(cosmology.geometry()(al.point(x4))))
    assert np.abs(csta - np.asarray(cosmology.coframe(np.asarray(x4)))).max() < 1e-15


@pytest.mark.parametrize("name", list(CASES))
def test_solves_every_csta_equation(name):
    cosmology, beta, x4 = CASES[name]
    total, matter, spinor = rows(cosmology, beta, x4)
    for sector in "MP":
        assert np.abs(matter[al.SECTORS[sector]]).max() > 1.0
    assert np.abs(total).max() < 1e-13
    assert np.abs(spinor).max() < 1e-13


def test_not_without_the_contorsion():
    cosmology, beta, x4 = CASES["particle"]
    total, _, spinor = rows(cosmology, beta, x4, torsion=0.0)
    assert np.abs(total[al.SECTORS["M"]]).max() > 1.0
    assert np.abs(total[al.SECTORS["P"]]).max() > 1.0
    assert np.abs(spinor).max() > 0.1
