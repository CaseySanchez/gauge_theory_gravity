"""Quadratic curvature invariant of the CSTA connection (spec section 10) and an independent Weyl^2."""
from __future__ import annotations

import jax.numpy as jnp

from . import cartan
from .curvature import curvature
from .fields import biv


def quadratic_invariant(coframe, omega):
    """I(x) = g^{mu rho} g^{nu sigma} < F_{mu nu} ~F_{rho sigma} >_0  with the full 15-bivector F."""
    F = curvature(omega)
    g = cartan.metric(coframe)

    def I(x):
        Fx = biv(F(x))                                          # batch (4,4)
        ginv = jnp.linalg.inv(g(x))
        prod = (Fx[:, :, None, None] * ~Fx[None, None, :, :]).restrict[0].kernel[..., 0]   # [m,n,r,s]
        return jnp.einsum("mr,ns,mnrs->", ginv, ginv, prod)

    return I


def weyl_squared(coframe, x):
    """C_{abcd}C^{abcd} = Riem^2 - 2 Ric^2 + R^2/3, from the coordinate Riemann tensor."""
    R = cartan.riemann_coordinate(coframe)(x)                   # R^r_{s m n}
    g = cartan.metric(coframe)(x)
    gi = jnp.linalg.inv(g)
    Rl = jnp.einsum("ra,asmn->rsmn", g, R)                      # R_{r s m n}
    Ru = jnp.einsum("rsmn,ra,sb,mc,nd->abcd", Rl, gi, gi, gi, gi)
    riem2 = jnp.einsum("rsmn,rsmn->", Rl, Ru)
    Ric = jnp.einsum("rsrn->sn", R)
    Ricu = gi @ Ric @ gi
    ric2 = jnp.einsum("sn,sn->", Ric, Ricu)
    Rs = jnp.einsum("sn,sn->", gi, Ric)
    return riem2 - 2 * ric2 + Rs ** 2 / 3
