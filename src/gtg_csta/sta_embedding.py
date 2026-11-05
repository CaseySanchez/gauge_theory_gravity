"""The Lorentz sector: Cl(1,3) inside Cl(2,4), and the sign map to numga's own STA (spec section 9.1)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import algebra as al
from .fields import biv


def lorentz_connection(omega_ab):
    """Omega_mu from omega^{ab}_mu (shape (4, 6), ordered as M_PAIRS): e=f=b=0."""
    c = jnp.concatenate([jnp.asarray(omega_ab), jnp.zeros((*jnp.shape(omega_ab)[:-1], 9))], axis=-1)
    return jnp.einsum("ij,...j->...i", jnp.asarray(al.GEN), c)


def lorentz_rotor_field(B_ab_fn):
    """Return x -> even coefficients (32,) of exp of a spacetime bivector (a Lorentz rotor)."""
    def R(x):
        c = jnp.concatenate([jnp.asarray(B_ab_fn(x)), jnp.zeros(9)])
        return biv(jnp.einsum("ij,j->i", jnp.asarray(al.GEN), c)).exp().kernel
    return R


def sector_leakage(F_blade) -> dict:
    """Max |component| of F outside the Lorentz sector, per sector P, K, D."""
    g = al.to_generators(np.asarray(F_blade))
    return {k: float(np.abs(g[..., s]).max()) for k, s in al.SECTORS.items() if k != "M"}


def numga_sta_structure_constants() -> np.ndarray:
    """f_sta[i,j,k] for M_ab = e_a ^ e_b in numga's own STA 'x+y+z+t-' (opposite sign to ours),
    with generators ordered as al.M_PAIRS over (t,x,y,z)."""
    from numga import Algebra
    from numga.backend.jax import JaxContext
    A = Algebra("x+y+z+t-")
    m = JaxContext(A, np.float64).multivector
    es = [m.t, m.x, m.y, m.z]
    G = [es[a] ^ es[b] for a, b in al.M_PAIRS]
    Z = m.bivector(np.zeros(6))
    K = np.stack([np.asarray((Z + g).kernel) for g in G], axis=1)      # (6,6)
    Kinv = np.linalg.inv(K)
    f = np.zeros((6, 6, 6))
    for i, gi in enumerate(G):
        for j, gj in enumerate(G):
            f[i, j] = Kinv @ np.asarray((Z + (gi * gj - gj * gi).restrict[2]).kernel)
    return f
