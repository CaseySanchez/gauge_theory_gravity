"""The first, array-based implementation (raw coefficient arrays and einsum), kept as an independent reference
for the extensor implementation. It is deliberately written in plain `jax.numpy`."""
import jax.numpy as jnp

from .. import algebra as _al


def to_generators(blades):
    """Blade coefficients (..., 15) -> generator coordinates, traceable by JAX."""
    return jnp.einsum("ij,...j->...i", _al.GEN_INV, blades)
