"""Path-ordered parallel transport (spec section 8).

Parallel transport of Psi along gamma solves gamma'.D Psi = 0, i.e. dPsi/ds = -1/2 Omega(gamma') Psi,
so Psi(s) = T(s) Psi(0) with dT/ds = -1/2 Omega(gamma'(s)) T.  Later points multiply on the LEFT:
T = exp(-1/2 Omega_N ds) ... exp(-1/2 Omega_1 ds).  This is spec's  P exp(-1/2 int Omega).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .fields import biv, even


def _omega_along(omega, path, dpath):
    def f(s):
        x, v = path(s), dpath(s)
        return jnp.einsum("m,mi->i", v, omega(x))   # Omega(gamma') by linearity in the vector slot
    return f


def transport(omega, path, steps: int = 200, s0: float = 0.0, s1: float = 1.0, scheme: str = "midpoint"):
    """Versor T along gamma(s), s in [s0,s1]. `path` is s -> x (4,). Returns even coefficients (32,).
    scheme: 'midpoint' (2nd order) or 'magnus4' (4th order, 2-point Gauss)."""
    dpath = jax.jacfwd(path)
    om = _omega_along(omega, path, dpath)
    ds = (s1 - s0) / steps
    ss = s0 + ds * jnp.arange(steps)
    if scheme == "midpoint":
        gens = jax.vmap(lambda s: om(s + 0.5 * ds))(ss)
        exps = (-0.5 * ds * biv(gens)).exp()
    elif scheme == "magnus4":
        c = np.sqrt(3.0) / 6.0
        a1 = jax.vmap(lambda s: om(s + (0.5 - c) * ds))(ss)
        a2 = jax.vmap(lambda s: om(s + (0.5 + c) * ds))(ss)
        A1, A2 = biv(-0.5 * a1), biv(-0.5 * a2)
        comm = (A1 * A2 - A2 * A1).restrict[2]
        B = 0.5 * ds * (A1 + A2) - (np.sqrt(3.0) / 12.0) * ds ** 2 * comm
        exps = B.exp()
    else:
        raise ValueError(scheme)
    K = exps.kernel                                   # (steps, 32)
    T0 = even(jnp.eye(1, 32)[0])
    return _fold(K, T0)


def _fold(K, T0):
    """Left-multiply the step versors in order: T = K[-1] ... K[1] K[0] T0."""
    def step(c, k):
        return (even(k) * even(c)).kernel, None
    T, _ = jax.lax.scan(step, T0.kernel, K)
    return T


def loop_path(x0, e_mu, e_nu, eps):
    """Rectangular loop of side eps in the (mu,nu) plane: x0 -> +mu -> +nu -> -mu -> -nu, s in [0,4]."""
    x0, e_mu, e_nu = map(jnp.asarray, (x0, e_mu, e_nu))
    corners = jnp.stack([x0, x0 + eps * e_mu, x0 + eps * (e_mu + e_nu), x0 + eps * e_nu, x0])

    def path(s):
        i = jnp.clip(jnp.floor(s).astype(int), 0, 3)
        f = s - i
        return corners[i] * (1 - f) + corners[i + 1] * f

    return path, corners


def holonomy_loop(omega, x0, mu, nu, eps, steps_per_side=40, scheme="midpoint"):
    """Transport around a rectangle by composing four straight legs."""
    e = jnp.eye(4)
    _, c = loop_path(x0, e[mu], e[nu], eps)
    T = jnp.eye(1, 32)[0]
    for i in range(4):
        a, b = c[i], c[i + 1]
        leg = lambda s, a=a, b=b: a + s * (b - a)
        Ti = transport(omega, leg, steps_per_side, 0.0, 1.0, scheme)
        T = (even(Ti) * even(T)).kernel
    return T


# Verified numerically (tests/test_transport.py): around the rectangle x0 -> +mu -> +nu -> -mu -> -nu
# of side eps, T = exp(-1/2 eps^2 F_{mu nu}) + O(eps^3), with F as in curvature.curvature.
