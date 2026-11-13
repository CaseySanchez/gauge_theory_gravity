"""Curvature (spec section 7), on extensors."""
from __future__ import annotations

from numga.extensor import Extensor

from . import algebra as al
from .covariant import covariant_derivative
from .fields import Field, derivative_of, permute_inputs


def curvature(omega: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """F(a,b) = a.grad Omega(b) - b.grad Omega(a) + 1/2 [Omega(a), Omega(b)]   (constant coordinate vectors a, b,
    so the -Omega([a,b]) term drops; [X,Y] = XY - YX).  A map Bivector <- (Spacetime, Spacetime).

    `d = derivative(Omega)` has the inputs (b, step a) = a.grad Omega(b); swapping them gives the other term, and
    numga's `commutator` is (XY - YX)/2, exactly the 1/2 [Omega(a), Omega(b)] of the spec."""
    gradient = derivative_of(method, step)(omega)

    def curved(x: Extensor) -> Extensor:
        connection = omega(x)                                                            # Bivector <- Spacetime
        d = gradient(x)                                                                  # Bivector <- (b, a)
        return permute_inputs(d, (1, 0)) - d + connection.commutator(connection)         # Bivector <- (a, b)

    return curved


def commutator_of_derivatives(omega: Field, spinor: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """[D_a, D_b] Psi from the covariant derivative itself (second derivatives of Psi cancel): a map
    Even <- (Spacetime a, Spacetime b)."""
    covariant = covariant_derivative(omega, spinor, method, step)
    gradient = derivative_of(method, step)(covariant)

    def commutator(x: Extensor) -> Extensor:
        # D_a (D_b Psi) = a.grad (D_b Psi) + 1/2 Omega(a) (D_b Psi), with the inputs of Omega(a) D_b Psi reordered to (b, a).
        rotation = permute_inputs(omega(x) * covariant(x), (1, 0))                       # Even <- (b, a)
        twice = permute_inputs(gradient(x) + 0.5 * rotation, (1, 0))                     # Even <- (a, b): D_a D_b Psi
        return twice - permute_inputs(twice, (1, 0))

    return commutator


def bianchi_residual(omega: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """The cyclic sum over (l,a,b) of D_l F(a,b) = d_l F(a,b) + 1/2 [Omega(l), F(a,b)]; it must vanish.
    A map Bivector <- (Spacetime, Spacetime, Spacetime)."""
    gradient = derivative_of(method, step)(curvature(omega, method, step))
    strength = curvature(omega, method, step)

    def residual(x: Extensor) -> Extensor:
        covariant = permute_inputs(gradient(x), (2, 0, 1)) + omega(x).commutator(strength(x))     # inputs (l, a, b)
        return covariant + permute_inputs(covariant, (2, 0, 1)) + permute_inputs(covariant, (1, 2, 0))

    return residual


def sector_split(strength: Extensor) -> dict[str, Extensor]:
    """The M, P, K, D parts of a bivector-valued extensor, by the null-basis projectors."""
    return {name: al.sector(strength, name) for name in al.SECTOR_MAPS}
