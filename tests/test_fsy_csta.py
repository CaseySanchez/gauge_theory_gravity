"""Phase 1 of docs/PLAN_DIRAC_SOLITON.md: the Einstein-Dirac soliton of Finster, Smoller and Yau (`fsy_solution`) in
the CSTA, with GTG's Lorentz coupling. The first localized self-gravitating solution in the repo.

The action is S1 + beta S2 + gamma S3 + 4 kappa |det e| (L_1 + L_2), two fermions, kappa = 8 pi, f = e / beta and
gamma = 1 / beta (Lambda = 0). The coframe is identity plus the radial dyad (1/sqrt A - 1) P and (1/T - 1) on the time
part, equal to the oracle's; the connection is Levi-Civita with no contorsion, since the singlet's spin vanishes. Then:

  * all 60 connection rows and both fermions' 8 spinor rows vanish to the Chebyshev fit's floor (1e-10);
  * each fermion's spin source in the Lorentz rows is O(0.2) and the two cancel: one fermion at the singlet's total
    density (its matter rows doubled) fails the Lorentz rows.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, dirac as dr, first_order as fo
import fsy_solution as fs
from test_fsy_solution import EVENTS, STRONG

BETA = 4.0
M, P = al.SECTORS["M"], al.SECTORS["P"]


def rows(x4):
    """The gravitational connection rows, each fermion's matter rows ([15, 4] each) and spinor rows ([8] each)."""
    geometry = STRONG.geometry()
    omega = STRONG.csta_connection(lambda q: geometry(q) / BETA)
    gravity = fo.euler_lagrange_connection(omega, x4)
    gravity = gravity[0] + BETA * gravity[1] + gravity[2] / BETA
    matter = [4 * fs.KAPPA * dr.euler_lagrange_connection(omega, STRONG.csta_spinor(k), x4, mass=STRONG.mass,
                                                          which="lorentz") for k in range(2)]
    spinor = [dr.euler_lagrange_spinor(omega, STRONG.csta_spinor(k), x4, mass=STRONG.mass, which="lorentz")
              for k in range(2)]
    return gravity, matter, spinor


def test_coframe_is_the_oracles():
    x4 = EVENTS[0]
    geometry = STRONG.geometry()
    x = al.point(x4)
    coframe = np.asarray(al.components(geometry(x)))
    assert np.abs(coframe - np.asarray(STRONG.coframe(np.asarray(x4)))).max() < 1e-15
    assert np.abs(np.asarray(al.components(geometry.frame(x))) @ coframe - np.eye(4)).max() < 1e-15


def check(x4):
    gravity, matter, spinor = rows(x4)
    total = gravity + matter[0] + matter[1]
    assert min(np.abs(m[M]).max() for m in matter) > 0.05 and min(np.abs(m[P]).max() for m in matter) > 0.05
    assert np.abs(total).max() < 1e-10
    assert max(np.abs(s).max() for s in spinor) < 1e-12
    alone = gravity + 2 * matter[0]                                     # one fermion, at the singlet's total density
    assert np.abs(alone[M]).max() > 0.1           # 0.46 at r = 1.7, 0.14 at r = 2.6


def test_solves_every_csta_equation():
    check(EVENTS[0])


@pytest.mark.slow
def test_solves_every_csta_equation_at_another_radius():
    check(EVENTS[1])
