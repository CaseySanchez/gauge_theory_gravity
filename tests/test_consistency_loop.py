"""Phase 6 of docs/PLAN_DIRAC.md: the consistency loop with the action quadratic in curvature added,
S1 + beta S2 + gamma S3 + alpha <F ^ F ^ X ^ I> + 4 kappa |det e| L_D.

Lasenby, Doran and Gull (4.32-4.35) find that the minimally coupled Dirac action gives the minimally coupled equation
only for the Ricci-scalar action. Here, with X and I fixed, <F ^ F ^ X ^ I> sees only the Lorentz part of F,
R + (e ^ f terms); its R ^ R part is the Euler density and drops out, so the Lorentz equation gains
alpha eps (T ^ f - e ^ D f). Two outcomes follow:

  * Lorentz coupling (LDG's): the Dirac field does not source f, f = e / beta, the new term only rescales S1's, and a
    rescaled LDG contorsion solves the Lorentz rows for any alpha. The torsion stays traceless and the loop holds.
  * Full transport (the CSTA target): the Dirac field shifts f (`test_spin_torsion`), its gradient enters through
    e ^ D f, and the torsion acquires a trace linear in alpha (slow test). So the minimally coupled Dirac equation
    fails unless alpha = 0: with Dirac matter, the CSTA fixes the Weyl-squared admixture to zero.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, conformal as cf, dirac as dr, first_order as fo
from test_gtg_dirac_reference import GEOMETRIES, csta_spinor

KAPPA, BETA, MASS = 0.3, 4.0, 0.7
M = al.SECTORS["M"]
LORENTZ_UNITS = [al.GENS[k] * (al.SLICES[m] & al.Direction) for k in range(6) for m in range(4)]


def gravity_rows(omega, x4):
    """The Lorentz rows of S1 and of <F ^ F ^ X ^ I> (S2, S3 have none), [2, 6, 4]."""
    both = lambda connection, x: jnp.stack([fo.densities(connection, x)[0], fo.macdowell_mansouri(connection, x)])  # noqa: E731
    return fo.euler_lagrange_connection(omega, x4, density=both)[:, M]


def ldg_contorsion(base, scale):
    def omega(position):
        connection = base(position)
        spin = dr.spin_trivector(csta_spinor(position))
        legs = dr.coframe_of(connection)(al.DIRECTIONS).dual_inverse()
        return connection + sum(((0.5 * KAPPA * scale * (legs[mu] * spin).restrict_grade(2)).cast(al.Bivector)
                                 * (al.SLICES[mu] & al.Direction) for mu in range(4)), 0 * connection)
    return omega


@pytest.mark.parametrize("alpha", [0.01, 0.1])
def test_lorentz_coupling_keeps_the_loop_for_any_alpha(alpha):
    geometry, x4 = GEOMETRIES["kerr"]
    base = cf.connection(geometry, lambda q: geometry(q) / BETA)
    matter = 4 * KAPPA * dr.euler_lagrange_connection(base, csta_spinor, x4, mass=MASS, which="lorentz")[M]
    rows = [gravity_rows(ldg_contorsion(base, s), x4) for s in (0.0, 1.0)]
    total = [r[0] + alpha * r[1] + matter for r in rows]                         # affine in the scale s
    slope = (total[1] - total[0]).ravel()
    scale = -float(slope @ total[0].ravel()) / float(slope @ slope)
    assert abs(scale - 1.0) > 1e-3                                                # rescaled, not LDG's
    assert np.abs(total[0] + scale * (total[1] - total[0])).max() < 1e-11
    x = al.point(x4)
    assert np.abs(np.asarray(dr.torsion_trace(ldg_contorsion(base, scale), x).kernel)).max() < 1e-13


def shifted_special(geometry):
    """The Dirac-shifted special-conformal field of `test_spin_torsion`, as a field:
    F^a_b = (1/beta)(1 + kappa tr C / 3) delta - (kappa/beta) C^a_b."""
    def special(position):
        current = dr.special_conformal_current(csta_spinor(position))              # [a, b] = C^b_a
        trace = current[0, 0] + current[1, 1] + current[2, 2] + current[3, 3]
        coframe = geometry(position)
        total = 0 * coframe
        for a in range(4):
            for b in range(4):
                weight = (1 + KAPPA * trace / 3) / BETA * float(a == b) - KAPPA / BETA * current[b, a]
                total = total + weight * (al.DIRECTIONS[a] * (al.SLICES[b] & coframe))
        return total
    return special


@pytest.mark.slow
def test_full_transport_breaks_the_loop_linearly_in_alpha():
    """The Lorentz rows are affine in a constant contorsion at x (24 unknowns): solved for each alpha, with f the
    Dirac-shifted field (exact to first order in alpha)."""
    geometry, x4 = GEOMETRIES["kerr"]
    x = al.point(x4)
    base = cf.connection(geometry, shifted_special(geometry))

    def with_contorsion(amplitudes):
        shift = sum((LORENTZ_UNITS[i] * float(amplitudes[i]) for i in range(24)), 0 * LORENTZ_UNITS[0])
        return lambda position: base(position) + shift

    rows0 = gravity_rows(base, x4).reshape(2, -1)
    columns = [gravity_rows(with_contorsion(np.eye(24)[i]), x4).reshape(2, -1) - rows0 for i in range(24)]
    s1, mm = np.stack([c[0] for c in columns], 1), np.stack([c[1] for c in columns], 1)
    matter = 4 * KAPPA * dr.euler_lagrange_connection(base, csta_spinor, x4, mass=MASS, which="full")[M].ravel()
    traces = {}
    for alpha in (0.0, 1e-3, 1e-2):
        amplitudes = np.linalg.solve(s1 + alpha * mm, -(rows0[0] + alpha * rows0[1] + matter))
        traces[alpha] = np.asarray(dr.torsion_trace(with_contorsion(amplitudes), x).kernel).ravel()
    assert np.abs(traces[0.0]).max() < 1e-12
    rate = traces[1e-3] / 1e-3
    assert np.abs(rate).max() > 0.1
    assert np.abs(traces[1e-2] / 1e-2 - rate).max() < 0.05 * np.abs(rate).max()
