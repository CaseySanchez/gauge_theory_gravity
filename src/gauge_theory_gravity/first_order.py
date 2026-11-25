"""First-order form: the full CSTA connection as the only gravitational field, broken by two compensators.

The origin X = n_o and the infinity twistor I = n_inf (null, X . I = 1) break Spin(2,4) to the Lorentz group.  The
coframe is the covariant derivative of the origin, DX = Omega x n_o (the translation sector), and DI = Omega x n_inf
carries the special-conformal sector.  The action is a sum of top forms, each the pseudoscalar part of a wedge:

    S1 = <F ^ DX ^ DX ^ X ^ I>,    S2 = <DI ^ DI ^ DX ^ DX ^ X ^ I>,    S3 = <DX ^ DX ^ DX ^ DX ^ X ^ I>,

a conformal Stelle-West form: S1 is Einstein-Cartan plus a term linear in the special-conformal field, S2 makes that
field algebraic, S3 is a cosmological term.  The densities here are their components against d^4x, functions of a
connection field Omega : Bivector <- Direction; `euler_lagrange_connection` varies all 60 of its components.

For S1 + beta S2 + gamma S3 the solutions (`tests/test_first_order.py`) have f = e / beta, a torsion-free Lorentz
sector and an Einstein metric with Lambda = 6 (gamma - 1/beta); the dilation field drops out. With gamma = 0 they are
exactly the normal conformal connections of Einstein metrics, Lambda = -6 / beta.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from .curvature import curvature
from .fields import Field
from .variation import _euler, _monomial, _number

ORIGIN, INFINITY = al.n_o, al.n_inf
GENERATORS = al.GENS
# The 60 connection units G_k (dx^mu . ), [15, 4] maps Bivector <- Direction.
UNITS = [[g * (al.SLICES[m] & al.Direction) for m in range(4)] for g in GENERATORS]


def _levi_civita() -> np.ndarray:
    eps = np.zeros((4, 4, 4, 4))
    for p in permutations(range(4)):
        eps[p] = np.linalg.det(np.eye(4)[list(p)])
    return eps


EPSILON = _levi_civita()


def _top(*factors) -> jax.Array:
    """The pseudoscalar coefficient of a wedge of multivectors."""
    total = factors[0]
    for f in factors[1:]:
        total = total ^ f
    return total.kernel.reshape(-1)[0]


def covariant_points(omega: Extensor) -> tuple[list, list]:
    """DX(d_mu) and DI(d_mu) for constant X = n_o, I = n_inf: the commutator action of Omega(d_mu)."""
    dx = [omega(al.DIRECTIONS[m]).commutator(ORIGIN) for m in range(4)]
    di = [omega(al.DIRECTIONS[m]).commutator(INFINITY) for m in range(4)]
    return dx, di


def densities(omega: Field, x: Extensor) -> jax.Array:
    """(S1, S2, S3) at x as components of d^4x: sums over eps^{mu nu rho sigma}, with F(a, b) the curvature."""
    value = omega(x)
    strength = curvature(omega)(x)
    dx, di = covariant_points(value)
    s1 = s2 = s3 = 0.0
    for (m, n, r, s) in permutations(range(4)):
        sign = EPSILON[m, n, r, s]
        f = strength(al.DIRECTIONS[m], al.DIRECTIONS[n])
        s1 = s1 + sign * _top(f, dx[r], dx[s], ORIGIN, INFINITY)
        s2 = s2 + sign * _top(di[m], di[n], dx[r], dx[s], ORIGIN, INFINITY)
        s3 = s3 + sign * _top(dx[m], dx[n], dx[r], dx[s], ORIGIN, INFINITY)
    return jnp.stack([s1, s2, s3])


def _moved_connection(omega: Field, alpha, x, amplitude) -> Field:
    def moved(y: Extensor) -> Extensor:
        weight = _monomial(y - x, alpha)
        shift = sum((UNITS[k][m] * (amplitude[k, m] * weight) for k in range(15) for m in range(4)), 0 * UNITS[0][0])
        return omega(y) + shift
    return moved


def euler_lagrange_connection(omega: Field, x4) -> np.ndarray:
    """delta S_i / delta Omega^k_mu for the three densities: an array [3, 15, 4] (generator k in `algebra.LABELS`
    order, coordinate mu).  The densities are first order in Omega."""
    return _euler(lambda alpha, x, amplitude: densities(_moved_connection(omega, alpha, x, amplitude), x),
                  (15, 4), x4, order=1)
