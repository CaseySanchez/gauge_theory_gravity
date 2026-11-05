"""Field representation and derivatives.

A field is a callable on a point x (shape (4,), coordinates ordered t,x,y,z) returning raw numga
kernel coefficients:
  * connection  Omega(x): (4, 15)  -- Omega(e_mu), bivector blade coefficients
  * versor      V(x):     (32,)    -- even-subalgebra coefficients
  * spinor      Psi(x):   (32,)    -- even-subalgebra coefficients
Derivatives come from JAX autodiff ("jax") or central differences ("fd"), with one interface.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..algebra import mv

NDIM = 4


def biv(c):
    return mv.bivector(jnp.asarray(c))


def even(c):
    return mv.even(jnp.asarray(c))


def partials(f, method: str = "jax", h: float = 1e-4):
    """Return x -> array of shape (4, *f(x).shape): the coordinate derivatives d_mu f."""
    if method == "jax":
        jac = jax.jacfwd(f)
        return lambda x: jnp.moveaxis(jac(jnp.asarray(x)), -1, 0)
    if method == "fd":
        def d(x):
            x = jnp.asarray(x, dtype=float)
            out = []
            for m in range(NDIM):
                e = jnp.zeros(NDIM).at[m].set(h)
                out.append((8 * (f(x + e) - f(x - e)) - (f(x + 2 * e) - f(x - 2 * e))) / (12 * h))
            return jnp.stack(out)
        return d
    raise ValueError(method)


def random_bivector(rng, scale=1.0, shape=()):
    return rng.normal(size=(*shape, 15)) * scale


def random_versor_field(seed: int = 0, scale: float = 0.3):
    """V(x) = exp(B(x)), B(x) a smooth, genuinely x-dependent 15-component bivector: Spin(2,4)."""
    rng = np.random.default_rng(seed)
    B0 = rng.normal(size=15) * scale
    B1 = rng.normal(size=(15, 4)) * scale
    B2 = rng.normal(size=(15, 4)) * scale

    def V(x):
        return biv(B0 + B1 @ x + B2 @ jnp.sin(1.3 * x)).exp().kernel

    return V


def random_connection_field(seed: int = 0, scale: float = 0.3):
    """A generic (non-flat) smooth bivector-valued connection Omega(x): (4, 15)."""
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(4, 15)) * scale
    A1 = rng.normal(size=(4, 15, 4)) * scale
    A2 = rng.normal(size=(4, 15, 4)) * scale

    def Om(x):
        return A0 + A1 @ x + A2 @ jnp.cos(0.9 * x)

    return Om


def random_spinor_field(seed: int = 0, scale: float = 0.5):
    rng = np.random.default_rng(seed)
    P0 = rng.normal(size=32) * scale
    P1 = rng.normal(size=(32, 4)) * scale
    P2 = rng.normal(size=(32, 4)) * scale

    def Psi(x):
        return P0 + P1 @ x + P2 @ jnp.sin(0.7 * x)

    return Psi
