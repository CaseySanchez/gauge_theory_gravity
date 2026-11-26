"""Cartan geometry from a tetrad: Lorentz connection, Riemann and Ricci (independent checks).

Tetrad convention: coframe e[a, nu] (frame index a, coordinate index nu) with
g_{mu nu} = eta_{ab} e^a_mu e^b_nu, eta = diag(+,-,-,-).  Frame vectors E_b^nu = inverse(e)[nu, b].
The Lorentz connection in the algebra is  Omega_mu = 1/2 omega_mu^{ab} e_a ^ e_b  with
omega_mu^{ab} = omega_mu^a_c eta^{cb}, so that 1/2 [Omega_mu, v] = omega_mu^a_c v^c e_a.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .. import algebra as al
from . import GEN

ETA = jnp.asarray(al.ETA4)


def metric(coframe):
    return lambda x: jnp.einsum("ab,am,bn->mn", ETA, coframe(x), coframe(x))


def christoffel(coframe):
    """Gamma[l, m, n] = Gamma^l_{mn}."""
    g = metric(coframe)
    dg = jax.jacfwd(g)                                    # dg[m, n, l] = d_l g_mn

    def Gam(x):
        d = dg(x)
        ginv = jnp.linalg.inv(g(x))
        T = jnp.einsum("snm->smn", d) + d - jnp.einsum("mns->smn", d)
        return 0.5 * jnp.einsum("ls,smn->lmn", ginv, T)

    return Gam


def riemann_coordinate(coframe):
    """R[r, s, m, n] = R^r_{s m n} from Christoffel symbols."""
    Gam = christoffel(coframe)
    dGam = jax.jacfwd(Gam)                                # dGam[l, m, n, k] = d_k Gamma^l_mn

    def R(x):
        G, dG = Gam(x), dGam(x)
        t1 = jnp.einsum("rnsm->rsmn", dG)                 # d_m Gamma^r_{n s}
        t2 = jnp.einsum("rmsn->rsmn", dG)                 # d_n Gamma^r_{m s}
        t3 = jnp.einsum("rml,lns->rsmn", G, G)
        t4 = jnp.einsum("rnl,lms->rsmn", G, G)
        return t1 - t2 + t3 - t4

    return R


def ricci_coordinate(coframe):
    R = riemann_coordinate(coframe)
    return lambda x: jnp.einsum("rsrn->sn", R(x))


def frame_vectors(coframe):
    return lambda x: jnp.linalg.inv(coframe(x))           # [nu, b]


def spin_connection(coframe):
    """omega[mu, a, b] = omega_mu^a_b = e^a_l (d_mu E_b^l + Gamma^l_{mu s} E_b^s)."""
    Gam = christoffel(coframe)
    Efun = frame_vectors(coframe)
    dE = jax.jacfwd(Efun)                                 # dE[l, b, mu]

    def om(x):
        e, E, G, d = coframe(x), Efun(x), Gam(x), dE(x)
        cov = jnp.einsum("lbm->mlb", d) + jnp.einsum("lms,sb->mlb", G, E)
        return jnp.einsum("al,mlb->mab", e, cov)

    return om


def riemann_frame_from_coordinate(coframe):
    """R^a_{b mu nu} = e^a_r R^r_{s mu nu} E_b^s."""
    R = riemann_coordinate(coframe)
    Efun = frame_vectors(coframe)

    def Rf(x):
        return jnp.einsum("ar,rsmn,sb->abmn", coframe(x), R(x), Efun(x))

    return Rf


def lorentz_connection_blades(coframe):
    """Omega(x): (4, 15) bivector blade coefficients, Lorentz sector only."""
    om = spin_connection(coframe)
    pairs = al.M_PAIRS

    def Om(x):
        w = om(x)                                         # [mu, a, b]
        wab = jnp.einsum("mac,cb->mab", w, ETA)           # omega^{ab} (eta^{cb} = eta_cb for diag +-1)
        c = jnp.stack([wab[:, a, b] for a, b in pairs], axis=-1)       # (4, 6)
        full = jnp.concatenate([c, jnp.zeros((4, 9))], axis=-1)
        return jnp.einsum("ij,mj->mi", jnp.asarray(GEN), full)

    return Om


def painleve_gullstrand(M: float = 1.0):
    """Schwarzschild in Painleve-Gullstrand Cartesian coordinates (t,x,y,z):
    e^0 = dt,  e^i = dx^i + beta^i dt,  beta = sqrt(2M/r) x/r.  ds^2 = dt^2 - sum (e^i)^2."""
    def coframe(x):
        r = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
        beta = jnp.sqrt(2.0 * M / r) * x[1:] / r
        e = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        e = e.at[1:, 0].set(beta)
        e = e.at[1:, 1:].set(jnp.eye(3))
        return e
    return coframe


