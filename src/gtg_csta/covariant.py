"""Covariant derivative (spec section 6)."""
from __future__ import annotations

from .fields import biv, even, partials


def covariant_derivative(omega, Psi, method: str = "jax", h: float = 1e-4):
    """D_mu Psi = d_mu Psi + 1/2 Omega_mu Psi.  Returns x -> (4, 32) even coefficients."""
    dPsi = partials(Psi, method, h)

    def D(x):
        return (even(dPsi(x)) + 0.5 * biv(omega(x)) * even(Psi(x))).kernel

    return D
