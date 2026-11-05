"""Curvature (spec section 7) and its sector decomposition."""
from __future__ import annotations

import jax.numpy as jnp

from . import algebra as al
from .fields import biv, partials

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def curvature(omega, method: str = "jax", h: float = 1e-4):
    """F_{mu nu} = d_mu Omega_nu - d_nu Omega_mu + 1/2 [Omega_mu, Omega_nu]  (coordinate vectors,
    so the -Omega([a,b]) term vanishes; [X,Y] = XY - YX).
    Returns x -> (4, 4, 15) blade coefficients; F[m, n] = F(e_m, e_n)."""
    dOm = partials(omega, method, h)             # (4, 4, 15): [mu, nu, :]

    def F(x):
        Om = biv(omega(x))                        # batch (4,)
        d = dOm(x)
        Omm = Om[:, None]
        Omn = Om[None, :]
        comm = (Omm * Omn - Omn * Omm).restrict[2].kernel   # (4,4,15)
        return d - jnp.swapaxes(d, 0, 1) + 0.5 * comm

    return F


def commutator_of_D(omega, Psi, method: str = "jax", h: float = 1e-4):
    """[D_mu, D_nu] Psi, computed directly from the covariant derivative (second derivatives of Psi
    cancel). Returns x -> (4,4,32)."""
    from .covariant import covariant_derivative
    from .fields import even

    D = covariant_derivative(omega, Psi, method, h)
    dD = partials(D, method, h)                  # (4[nu], 4[mu], 32): d_nu (D_mu Psi)

    def C(x):
        Om = biv(omega(x))
        DPsi = even(D(x))                        # (4,) over mu
        d = even(dD(x))                          # [nu, mu]
        # D_nu D_mu Psi = d_nu (D_mu Psi) + 1/2 Omega_nu (D_mu Psi)
        DD = d + 0.5 * Om[:, None] * DPsi[None, :]       # [nu, mu]
        return DD.kernel - jnp.swapaxes(DD.kernel, 0, 1)   # [mu,nu] = D_mu D_nu - D_nu D_mu
    return C


def bianchi_residual(omega, method: str = "jax", h: float = 1e-4):
    """Cyclic sum  d_l F_mn + 1/2 [Omega_l, F_mn]  over (l,m,n). Must vanish."""
    F = curvature(omega, method, h)
    dF = partials(F, method, h)                  # (4[l], 4, 4, 15)

    def R(x):
        Om = biv(omega(x))
        Fx = biv(F(x))
        adj = 0.5 * (Om[:, None, None] * Fx[None] - Fx[None] * Om[:, None, None]).restrict[2].kernel
        T = dF(x) + adj                          # [l, m, n, :]  (D_l F_mn)
        return T + jnp.transpose(T, (1, 2, 0, 3)) + jnp.transpose(T, (2, 0, 1, 3))

    return R


def sector_split(F_blade):
    """Blade coefficients (..., 15) -> dict of sector coordinates {M,P,K,D}."""
    g = al.to_generators(jnp.asarray(F_blade))
    return {k: g[..., s] for k, s in al.SECTORS.items()}
