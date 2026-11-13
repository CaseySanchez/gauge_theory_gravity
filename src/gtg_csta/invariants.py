"""Quadratic curvature invariant of the CSTA connection (spec section 10), on extensors."""
from __future__ import annotations

from numga.extensor import Extensor

from . import algebra as al
from . import cartan, conformal
from .cartan import Coframe
from .curvature import curvature
from .fields import Field

Spacetime = al.Spacetime


def quadratic_invariant(coframe: Coframe, omega: Field) -> Field:
    """I(x) = g^{mu rho} g^{nu sigma} < F_{mu nu} ~F_{rho sigma} >_0 with the full 15-bivector F.
    Evaluated in the frame, with F(E, E), the two index contractions run through the algebra's own metric."""
    strength = curvature(omega)

    def invariant(x: Extensor) -> Extensor:
        frame = coframe(x).inverse()
        framed = strength(x)(frame, frame)                                             # Bivector <- (a, b)
        return (framed * ~framed).restrict[0].contract(1, 3).contract(1, 2)           # [] Scalar+

    return invariant


def weyl_squared(coframe: Coframe, x: Extensor) -> Extensor:
    """C^2 = Riem^2 - 2 Ric^2 + R^2/3, with Riem^2 = 2 I of the Lorentz connection."""
    lorentz = cartan.lorentz_connection(coframe)
    frame = coframe(x).inverse()
    riemann_squared = 2 * quadratic_invariant(coframe, lorentz)(x)                    # [] Scalar+
    ricci = conformal.ricci_vector(curvature(lorentz)(x), frame).cast(Spacetime)(frame)    # Ric^b_a as an endomorphism
    return riemann_squared - 2 * ricci(ricci).trace() + ricci.trace() ** 2 / 3
