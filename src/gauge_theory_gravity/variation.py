"""Field equations: the Euler-Lagrange derivative of a density with respect to the coframe, pointwise, by autodiff.

A density L[e](x) that depends on the coframe and its derivatives up to second order has a first variation
dL[h] = A(x) h + A^mu(x) d_mu h + sum_{mu <= nu} A^{mu nu}(x) d_mu d_nu h, and the field equations are

    E = A - d_mu A^mu + sum_{mu <= nu} d_mu d_nu A^{mu nu},     dS = integral of E . h  (up to a boundary term).

Each coefficient is read at x by perturbing the coframe with a polynomial centred there, h(y) = U (y - x)^alpha /
alpha!, whose derivatives at x are 1 for the multi-index alpha and 0 for the others, and differentiating L in the
perturbation's amplitude; the coefficients are then differentiated in x (the centre moves with x).  The amplitudes
run over the coordinate units U = d_a (dx^mu . ), so E comes out as an array E[a, mu] = dL/de^a_mu, indexed like
`algebra.components` of a coframe.  Nothing is integrated and no grid is used.

The densities here are those of `invariants` times the volume |det e|: Einstein-Hilbert and Weyl squared, and the
compensator density of the scale tractor, which depends on a scale sigma as well (`euler_lagrange_with_scale`).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from . import cartan, compensator, conformal, invariants
from .cartan import Coframe

Density = Callable[[Coframe, Extensor], jax.Array]      # (coframe field, event) -> a real number

# The coordinate units d_a (dx^mu . ), a [4, 4] stack of maps Direction <- Direction.
UNITS = al.DIRECTIONS[:, None] * (al.SLICES[None, :] & al.Direction)


def _number(scalar) -> jax.Array:
    return scalar if isinstance(scalar, float) else scalar.kernel.reshape(-1)[0]


def coframe_components(e: Extensor) -> jax.Array:
    """e^a_mu as an array [a, mu], traceable (unlike `algebra.components`, which leaves JAX)."""
    return al.coordinates(e(al.DIRECTIONS)).T


def volume(coframe: Coframe, x: Extensor) -> jax.Array:
    """sqrt|det g| = |det e|."""
    return jnp.abs(jnp.linalg.det(coframe_components(coframe(x))))


def einstein_hilbert(coframe: Coframe, x: Extensor) -> jax.Array:
    """sqrt|g| R."""
    return volume(coframe, x) * _number(conformal.ricci_scalar(coframe, x))


def weyl_squared(coframe: Coframe, x: Extensor) -> jax.Array:
    """sqrt|g| C_abcd C^abcd, twice the quadratic density of the normal conformal connection (spec 10)."""
    return volume(coframe, x) * _number(invariants.weyl_squared(coframe, x))


def _monomial(step: Extensor, alpha: tuple[int, ...]) -> jax.Array:
    """(y - x)^alpha / alpha! as a number."""
    value = 1.0
    for m in alpha:
        value = value * cartan.coordinate(step, m)
    return _number(value) / float(np.prod([math.factorial(alpha.count(m)) for m in set(alpha)]))


def _euler(perturbed: Callable, shape: tuple[int, ...], x4, order: int = 2) -> np.ndarray:
    """The Euler operator of order 1 or 2 from `perturbed(alpha, x, amplitude)`: the density at x with the field moved
    by amplitude * (y - x)^alpha / alpha!, amplitude an array of the field's shape."""
    def coefficient(alpha):
        def at(x4):
            x = al.point(x4)
            return jax.jacfwd(lambda amplitude: perturbed(alpha, x, amplitude))(jnp.zeros(shape))
        return at

    x4 = jnp.asarray(x4, dtype=jnp.float64)
    total = coefficient(())(x4)
    for m in range(4):
        total = total - jax.jacfwd(coefficient((m,)))(x4)[..., m]
    for m in range(4 if order >= 2 else 0):
        for n in range(m, 4):
            total = total + jax.hessian(coefficient((m, n)))(x4)[..., m, n]
    return np.asarray(total)


def _moved_coframe(coframe: Coframe, alpha, x, amplitude) -> Coframe:
    def moved(y: Extensor) -> Extensor:
        return coframe(y) + (UNITS * (amplitude * _monomial(y - x, alpha))).sum(axis=(0, 1))
    return moved


def _moved_scale(scale: Callable, alpha, x, amplitude) -> Callable:
    def moved(y: Extensor) -> Extensor:
        return scale(y) + amplitude * _monomial(y - x, alpha)
    return moved


def euler_lagrange(density: Density, coframe: Coframe, x4) -> np.ndarray:
    """E[a, mu] = delta S / delta e^a_mu at the event with coordinates x4, for a density of order <= 2."""
    return _euler(lambda alpha, x, amplitude: density(_moved_coframe(coframe, alpha, x, amplitude), x), (4, 4), x4)


def euler_lagrange_with_scale(density: Callable, coframe: Coframe, scale: Callable, x4) -> tuple[np.ndarray, float]:
    """For a density of a coframe and a scale, `density(coframe, scale, x)`: (delta S / delta e^a_mu [a, mu],
    delta S / delta sigma) at x4."""
    by_coframe = _euler(lambda alpha, x, amplitude: density(_moved_coframe(coframe, alpha, x, amplitude), scale, x),
                        (4, 4), x4)
    by_scale = _euler(lambda alpha, x, amplitude: density(coframe, _moved_scale(scale, alpha, x, amplitude), x),
                      (), x4)
    return by_coframe, float(by_scale)


def compensator_density(cosmological: float = 0.0) -> Callable:
    """sqrt|g| sigma^-4 (I . I + c), I the scale tractor: Weyl-invariant, and with g' = g/sigma^2 it is
    -sqrt|g'| (R' + 12 c)/12, Einstein-Hilbert with a cosmological term for the metric the scale picks out."""
    def density(coframe: Coframe, scale: Callable, x: Extensor) -> jax.Array:
        tractor = compensator.scale_tractor(coframe, scale)(x)
        sigma = _number(scale(x))
        return volume(coframe, x) * (_number(tractor | tractor) + cosmological) / sigma ** 4
    return density


# --- the identities every coframe density invariant under diffeomorphisms and local Lorentz turns satisfies -------
def lorentz_residual(equations: np.ndarray, coframe_at_x: np.ndarray) -> float:
    """Local Lorentz invariance makes eta_ac E[c, mu] e^b_mu symmetric in (a, b); the size of its antisymmetric part."""
    mixed = al.ETA4 @ equations @ coframe_at_x.T
    return float(np.abs(mixed - mixed.T).max())


def weyl_trace(equations: np.ndarray, coframe_at_x: np.ndarray) -> float:
    """E[a, mu] e^a_mu: zero pointwise for a density invariant under e -> e^lambda(x) e."""
    return float(np.sum(equations * coframe_at_x))
