"""The first, array-based implementation (raw coefficient arrays and einsum), kept as an independent reference
for the extensor implementation. It is deliberately written in plain `jax.numpy`."""
import jax.numpy as jnp
import numpy as np

from .. import algebra as _al

# GEN[:, i] = blade coefficients of generator i; blade_coeffs = GEN @ generator_coeffs. The extensor code reads
# generator coordinates with the reciprocal generators instead (`algebra.generator_coordinates`).
GEN = np.stack([_al._kernel(g) for g in _al.GENS], axis=1)
GEN_INV = np.linalg.inv(GEN)


def to_generators(blades):
    """Blade coefficients (..., 15) -> generator coordinates, traceable by JAX."""
    return jnp.einsum("ij,...j->...i", GEN_INV, blades)
