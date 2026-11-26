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


def _top(form: Extensor) -> jax.Array:
    """The pseudoscalar coefficient of a pseudoscalar-valued 4-form evaluated on the volume of directions."""
    return al.on_volume(form).kernel.reshape(-1)[0]


def covariant_points(omega: Extensor) -> tuple[Extensor, Extensor]:
    """DX and DI for constant X = n_o, I = n_inf: the commutator action of Omega, maps Vector <- Direction."""
    return omega.commutator(ORIGIN), omega.commutator(INFINITY)


def densities(omega: Field, x: Extensor) -> jax.Array:
    """(S1, S2, S3) at x as components of d^4x: each wedge, a 4-form of directions, evaluated on `algebra.VOLUME`."""
    strength = curvature(omega)(x)                                              # Bivector <- (a, b)
    dx, di = covariant_points(omega(x))                                         # Vector <- Direction
    points = ORIGIN ^ INFINITY
    return jnp.stack([_top(strength ^ dx ^ dx ^ points),
                      _top(di ^ di ^ dx ^ dx ^ points),
                      _top(dx ^ dx ^ dx ^ dx ^ points)])


def macdowell_mansouri(omega: Field, x: Extensor) -> jax.Array:
    """<F ^ F ^ X ^ I> at x as a component of d^4x: the action quadratic in curvature, MacDowell-Mansouri's form with
    the conformal group's two compensators. With X = n_o and I = n_inf only the Lorentz sector of F survives, so it is
    Kaku, Townsend and van Nieuwenhuizen's eps_abcd F^ab ^ F^cd."""
    strength = curvature(omega)(x)
    return _top(strength ^ strength ^ ORIGIN ^ INFINITY)


def _moved_connection(omega: Field, alpha, x, amplitude) -> Field:
    def moved(y: Extensor) -> Extensor:
        weight = _monomial(y - x, alpha)
        shift = sum((UNITS[k][m] * (amplitude[k, m] * weight) for k in range(15) for m in range(4)), 0 * UNITS[0][0])
        return omega(y) + shift
    return moved


def euler_lagrange_connection(omega: Field, x4, density: Callable = densities) -> np.ndarray:
    """delta S / delta Omega^k_mu: an array [..., 15, 4] (generator k in `algebra.LABELS` order, coordinate mu), for
    `densities` (the three terms, [3, 15, 4]) or `macdowell_mansouri` ([15, 4]).  Both are first order in Omega."""
    return _euler(lambda alpha, x, amplitude: density(_moved_connection(omega, alpha, x, amplitude), x),
                  (15, 4), x4, order=1)
