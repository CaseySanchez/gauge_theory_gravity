"""Quadratic curvature invariant of the CSTA connection (spec section 10), on extensors."""
from __future__ import annotations

from . import algebra as al
from .curvature import curvature
from . import conformal as cf, cartan

S = al.Spacetime


def quadratic_invariant(coframe, omega):
    """I(x) = g^{mu rho} g^{nu sigma} < F_{mu nu} ~F_{rho sigma} >_0 with the full 15-bivector F.
    In the frame, with F(E, E), the two index contractions run through the algebra's own metric."""
    F = curvature(omega)

    def I(x):
        E = coframe(x).inverse()
        Ff = F(x)(E, E)                                        # Bivector <- (frame a, frame b)
        return (Ff * ~Ff).restrict[0].contract(1, 3).contract(1, 2)
    return I


def weyl_squared(coframe, x):
    """C^2 = Riem^2 - 2 Ric^2 + R^2/3, with Riem^2 = 2 I of the Lorentz connection."""
    lorentz = cartan.lorentz_connection(coframe)
    riem2 = 2 * quadratic_invariant(coframe, lorentz)(x)
    e = coframe(x)
    E = e.inverse()
    T = cf.ricci_vector(curvature(lorentz)(x), E).cast(S)(E)    # Ric^b_a as an endomorphism
    return riem2 - 2 * T(T).trace() + T.trace() ** 2 / 3
