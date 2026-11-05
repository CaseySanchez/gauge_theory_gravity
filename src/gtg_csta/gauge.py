"""Gauge transformations (spec section 5), on extensors."""
from __future__ import annotations

import jax.numpy as jnp

from . import algebra as al
from .fields import get_derivative

Bivector = al.Bivector


def transform_connection(omega, V, method: str = "jax", h: float = 1e-4):
    """Omega'(a) = V Omega(a) ~V - 2 (a . grad V) ~V.
    Omega is a map Bivector <- Spacetime and V an Even-valued field; `V >> Omega` is the sandwich of
    the map's output, and the derivative of V is the map Even <- Spacetime (the step direction a)."""
    d = get_derivative(method, h)
    dV = d(V)

    def new(x):
        Vx = V(x)
        return (Vx >> omega(x)) - 2 * (dV(x) * ~Vx).restrict[2]
    return new


def non_bivector_part(omega, V, x, method: str = "jax") -> float:
    """Size of the non-bivector part of V Omega ~V - 2 dV ~V: the spec claims it vanishes."""
    Vx = V(x)
    out = Vx * omega(x) * ~Vx - 2 * (get_derivative(method)(V)(x) * ~Vx)
    return float(jnp.abs((out - out.restrict[2]).kernel).max())


def pure_gauge(V, method: str = "jax", h: float = 1e-4):
    """Omega = -2 (grad V) ~V: what the transformation gives from Omega = 0."""
    d = get_derivative(method, h)(V)

    def new(x):
        return -2 * (d(x) * ~V(x)).restrict[2]
    return new
