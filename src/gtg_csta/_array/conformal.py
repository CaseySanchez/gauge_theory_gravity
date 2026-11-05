"""Tier B: translation sector as coframe; normal conformal Cartan connection (spec 9.2, 10).

Connection:  Omega_mu = Lorentz(omega_mu) + e^a_mu P_a + f_{a mu} K_a + b_mu D,
with generator-coordinates (M, P, K, D) of al.GENS.  The Lorentz part is the torsion-free spin
connection of the coframe e.  The K coefficients f are fixed pointwise by the 'normal' conditions
(F_D = 0, F_M trace-free); this module finds them numerically and compares with the Schouten tensor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .. import algebra as al
from . import cartan
from .curvature import curvature

PAIRS = al.M_PAIRS


def connection(coframe, f_fn=None, b_fn=None):
    """Omega(x): (4,15) blade coefficients. f_fn(x) -> (4,4) with f[a, mu]; b_fn(x) -> (4,)."""
    om = cartan.spin_connection(coframe)
    GEN = jnp.asarray(al.GEN)
    ETA = cartan.ETA

    def Om(x):
        w = jnp.einsum("mac,cb->mab", om(x), ETA)                      # omega^{ab}_mu
        cM = jnp.stack([w[:, a, b] for a, b in PAIRS], axis=-1)        # (4,6)
        e = coframe(x).T                                               # (mu, a): translation coords
        f = jnp.zeros((4, 4)) if f_fn is None else f_fn(x).T           # (mu, a)
        b = jnp.zeros((4, 1)) if b_fn is None else b_fn(x)[:, None]
        c = jnp.concatenate([cM, e, f, b], axis=-1)                    # (4,15)
        return jnp.einsum("ij,mj->mi", GEN, c)

    return Om


def sectors(F_blade):
    g = al.to_generators(F_blade)
    return {k: g[..., s] for k, s in al.SECTORS.items()}


def lorentz_full(cM):
    """(..., 6) coordinates in M_PAIRS order -> antisymmetric (..., 4, 4) matrix X^{ab}."""
    out = jnp.zeros((*cM.shape[:-1], 4, 4))
    for k, (a, b) in enumerate(PAIRS):
        out = out.at[..., a, b].set(cM[..., k]).at[..., b, a].set(-cM[..., k])
    return out


def normal_residual(coframe, x, f):
    """Residual of the normal conditions for a constant-in-the-neighbourhood f (4,4): F_D (6 comps)
    and the trace T^b_nu = F^{ab}_{mu nu} E_a^mu (16 comps). Affine in f."""
    Om = connection(coframe, lambda _x: f)
    F = curvature(Om)(x)
    sec = sectors(F)
    FM = lorentz_full(sec["M"])                                   # [mu, nu, a, b]
    E = jnp.linalg.inv(coframe(x))                                # E[mu, a] = E_a^mu
    trace = jnp.einsum("mnab,ma->nb", FM, E)
    return jnp.concatenate([sec["D"].reshape(-1), trace.reshape(-1)])


def solve_normal_f(coframe, x):
    """Least-squares f (4,4) making the residual vanish; also returns the final residual norm."""
    f0 = jnp.zeros((4, 4))
    r0 = normal_residual(coframe, x, f0)
    J = jax.jacfwd(lambda v: normal_residual(coframe, x, v.reshape(4, 4)))(f0.reshape(-1))
    sol, *_ = jnp.linalg.lstsq(J, -r0)
    res = normal_residual(coframe, x, sol.reshape(4, 4))
    return sol.reshape(4, 4), float(jnp.abs(res).max()), int(jnp.linalg.matrix_rank(J))


def schouten_frame(coframe, x):
    """P_{ab} = 1/2 (Ric_ab - R g_ab / 6) in the frame, with Ric from cartan.ricci_coordinate
    (R_{sn} = R^r_{s r n}) and R = g^{sn} R_{sn};  returned as P[a, mu] = P_{ab} eta^{bb} ... see below."""
    e = coframe(x)
    E = jnp.linalg.inv(e)
    g = cartan.metric(coframe)(x)
    Ric = cartan.ricci_coordinate(coframe)(x)
    Rs = jnp.einsum("sn,sn->", jnp.linalg.inv(g), Ric)
    P = 0.5 * (Ric - Rs * g / 6.0)                                 # P_{mu nu}
    # f[a, mu]: frame index up, coordinate index down:  P_{mu nu} E_b^nu eta^{ba}... use P_{b mu} = E_b^nu P_{nu mu}
    Pb = jnp.einsum("nb,nm->bm", E, P)                             # P_{b mu}, frame-lower
    return jnp.einsum("ab,bm->am", jnp.asarray(al.ETA4), Pb), Rs   # raise with eta


# ---------- example spacetimes ----------

def frw(a=lambda t: t ** (2.0 / 3.0)):
    """Spatially flat FRW: e^0 = dt, e^i = a(t) dx^i (conformally flat, non-vacuum)."""
    def coframe(x):
        return jnp.diag(jnp.array([1.0, a(x[0]), a(x[0]), a(x[0])]))
    return coframe


def static_lapse(N=lambda x: 1.0 + 0.3 * jnp.sin(x[1]) * jnp.cos(x[2]) + 0.1 * x[3] ** 2):
    """e^0 = N(x) dt, e^i = dx^i: generically non-conformally-flat, non-vacuum."""
    def coframe(x):
        return jnp.diag(jnp.array([N(x), 1.0, 1.0, 1.0]))
    return coframe
