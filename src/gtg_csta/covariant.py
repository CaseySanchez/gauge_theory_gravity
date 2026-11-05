"""Covariant derivative (spec section 6), on extensors."""
from __future__ import annotations

from .fields import get_derivative


def covariant_derivative(omega, Psi, method: str = "jax", h: float = 1e-4):
    """D Psi = grad Psi + 1/2 Omega Psi : the map Even <- Spacetime, D_a Psi on direction a.
    `omega(x) * Psi(x)` is the map a -> Omega(a) Psi."""
    dPsi = get_derivative(method, h)(Psi)

    def D(x):
        return dPsi(x) + 0.5 * (omega(x) * Psi(x))
    return D
