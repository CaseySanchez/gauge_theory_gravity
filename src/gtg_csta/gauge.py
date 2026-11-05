"""Gauge transformations (spec section 5)."""
from __future__ import annotations

import jax.numpy as jnp

from .fields import biv, even, partials


def adjoint(V, X):
    """V X ~V."""
    return V * X * ~V


def transform_connection(omega, V, method: str = "jax", h: float = 1e-4):
    """Omega'(a) = V Omega(a) ~V - 2 (a.grad V) ~V,   evaluated on coordinate vectors e_mu.
    Returns a callable x -> (4, 15) bivector blade coefficients, plus the grade-2 residual."""
    dV = partials(V, method, h)

    def new(x):
        Vx = even(V(x))
        dVx = even(dV(x))                       # batch (4,)
        out = Vx * biv(omega(x)) * ~Vx - 2 * dVx * ~Vx
        return out.restrict[2].kernel

    return new


def transform_connection_residual(omega, V, x, method: str = "jax"):
    """Size of the non-bivector part of V Omega ~V - 2 dV ~V: spec claims it vanishes."""
    dV = partials(V, method)
    Vx = even(V(x))
    out = Vx * biv(omega(x)) * ~Vx - 2 * even(dV(x)) * ~Vx
    return float(jnp.abs((out - out.restrict[2]).kernel).max())


def pure_gauge(V, method: str = "jax", h: float = 1e-4):
    """Omega_mu = -2 (d_mu V) ~V  (what the transformation gives from Omega = 0)."""
    zero = lambda x: jnp.zeros((4, 15))
    return transform_connection(zero, V, method, h)
