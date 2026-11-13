"""Covariant derivative (spec section 6), on extensors."""
from __future__ import annotations

from numga.extensor import Extensor

from .fields import Field, derivative_of


def covariant_derivative(omega: Field, spinor: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """D Psi = grad Psi + 1/2 Omega Psi: the map Even <- Spacetime of the direction, `D_a Psi`.
    `omega(x) * psi(x)` is the map a -> Omega(a) Psi."""
    gradient = derivative_of(method, step)(spinor)

    def covariant(x: Extensor) -> Extensor:
        return gradient(x) + 0.5 * (omega(x) * spinor(x))                                # Even <- Spacetime

    return covariant
