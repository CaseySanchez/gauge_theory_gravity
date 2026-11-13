"""Gauge transformations (spec section 5), on extensors."""
from __future__ import annotations

import numpy as np
from numga.extensor import Extensor

from .fields import Field, derivative_of


def transform_connection(omega: Field, versor: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """Omega'(a) = V Omega(a) ~V - 2 (a . grad V) ~V.
    `V >> Omega` is the sandwich of the map's output, the input a being a direction of position and not
    transformed; the derivative of V is the map Even <- Spacetime of the step direction a."""
    gradient = derivative_of(method, step)(versor)

    def transformed(x: Extensor) -> Extensor:
        rotor = versor(x)
        return (rotor >> omega(x)) - 2 * (gradient(x) * ~rotor).restrict[2]            # Bivector <- Spacetime

    return transformed


def pure_gauge(versor: Field, method: str = "jax", step: float = 1e-4) -> Field:
    """Omega = -2 (grad V) ~V: what the transformation gives from Omega = 0, a flat connection."""
    gradient = derivative_of(method, step)(versor)

    def flat(x: Extensor) -> Extensor:
        return -2 * (gradient(x) * ~versor(x)).restrict[2]                              # Bivector <- Spacetime

    return flat


def non_bivector_part(omega: Field, versor: Field, x: Extensor, method: str = "jax") -> float:
    """Size of the non-bivector part of V Omega ~V - 2 dV ~V: the spec claims it vanishes."""
    rotor = versor(x)
    full = rotor * omega(x) * ~rotor - 2 * (derivative_of(method)(versor)(x) * ~rotor)   # Even <- Spacetime
    return float(np.abs(np.asarray((full - full.restrict[2]).kernel)).max())
