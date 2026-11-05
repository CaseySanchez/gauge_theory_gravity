"""Path-ordered parallel transport (spec section 8), on extensors.

Parallel transport of Psi along gamma solves gamma'.D Psi = 0, i.e. dPsi/ds = -1/2 Omega(gamma') Psi, so
Psi(s) = T(s) Psi(0) with dT/ds = -1/2 Omega(gamma'(s)) T.  Later points multiply on the LEFT:
T = exp(-1/2 Omega_N ds) ... exp(-1/2 Omega_1 ds).  This is the spec's P exp(-1/2 int Omega).
Binding the velocity into the connection, `omega(x)(v)`, is Omega(gamma').
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import algebra as al
from .fields import even


def _velocity(path):
    """s -> gamma'(s) as a Spacetime extensor."""
    jac = jax.jacfwd(lambda s: path(s).kernel)
    return lambda s: al.point(jac(s))


def transport(omega, path, steps: int = 200, s0: float = 0.0, s1: float = 1.0, scheme: str = "midpoint"):
    """Versor T(gamma) as even coefficients (32,).  `path` maps s -> Spacetime position extensor.
    scheme: 'midpoint' (2nd order) or 'magnus4' (4th order, 2-point Gauss)."""
    vel = _velocity(path)
    gen = lambda s: omega(path(s))(vel(s))                  # Omega(gamma'(s)): a bivector
    ds = (s1 - s0) / steps
    ss = s0 + ds * jnp.arange(steps)
    if scheme == "midpoint":
        gens = jax.vmap(lambda s: gen(s + 0.5 * ds))(ss)    # batch of bivectors
        exps = (-0.5 * ds * gens).exp()
    elif scheme == "magnus4":
        c = np.sqrt(3.0) / 6.0
        A1 = -0.5 * jax.vmap(lambda s: gen(s + (0.5 - c) * ds))(ss)
        A2 = -0.5 * jax.vmap(lambda s: gen(s + (0.5 + c) * ds))(ss)
        B = 0.5 * ds * (A1 + A2) - (np.sqrt(3.0) / 6.0) * ds ** 2 * A1.commutator(A2)   # commutator is [,]/2
        exps = B.exp()
    else:
        raise ValueError(scheme)
    return _fold(exps.kernel)


def _fold(K):
    """T = K[-1] ... K[1] K[0]  (left-multiply in order)."""
    one = jnp.eye(1, 32)[0]

    def step(c, k):
        return (even(k) * even(c)).kernel, None
    T, _ = jax.lax.scan(step, one, K)
    return T


def holonomy_loop(omega, x0, mu, nu, eps, steps_per_side=40, scheme="midpoint"):
    """Transport around a rectangle x0 -> +mu -> +nu -> -mu -> -nu by composing four straight legs.
    Verified: T = exp(-1/2 eps^2 F(e_mu, e_nu)) + O(eps^3)."""
    x0 = jnp.asarray(x0.kernel if hasattr(x0, "kernel") else x0)
    e = jnp.eye(4)
    c = [x0, x0 + eps * e[mu], x0 + eps * (e[mu] + e[nu]), x0 + eps * e[nu], x0]
    T = jnp.eye(1, 32)[0]
    for i in range(4):
        a, b = c[i], c[i + 1]
        leg = lambda s, a=a, b=b: al.point(a + s * (b - a))
        Ti = transport(omega, leg, steps_per_side, 0.0, 1.0, scheme)
        T = (even(Ti) * even(T)).kernel
    return T
