"""The Lorentz sector: Cl(1,3) inside Cl(2,4), and the sign map to numga's own STA (spec section 9.1)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import algebra as al
from .fields import bivector


def lorentz_connection(omega_ab):
    """Omega(d_mu) from omega^{ab}_mu (4, 6) in al.M_PAIRS order: e = f = b = 0.  Bivector <- Spacetime."""
    c = jnp.concatenate([jnp.asarray(omega_ab), jnp.zeros((*jnp.shape(omega_ab)[:-1], 9))], axis=-1)
    return al.connection_from_columns(al.to_blades(c))


def lorentz_rotor(B_ab):
    """A Lorentz rotor exp(B) from 6 coefficients in al.M_PAIRS order (an Even extensor)."""
    c = jnp.concatenate([jnp.asarray(B_ab), jnp.zeros(9)])
    return bivector(al.to_blades(c)).exp()


def sector_leakage(F) -> dict:
    """Largest component of a bivector-valued extensor outside the Lorentz sector, per sector P, K, D."""
    return {k: al.sector_size(F, k) for k in "PKD"}


def numga_sta_structure_constants() -> np.ndarray:
    """f_sta[i,j,k] for M_ab = e_a ^ e_b in numga's own STA 'x+y+z+t-' (opposite sign to ours),
    generators ordered as al.M_PAIRS over (t,x,y,z)."""
    from numga import Algebra
    from numga.backend.jax import JaxContext
    A = Algebra("x+y+z+t-")
    m = JaxContext(A, np.float64).multivector
    es = [m.t, m.x, m.y, m.z]
    G = [es[a] ^ es[b] for a, b in al.M_PAIRS]
    Z = m.bivector(np.zeros(6))
    K = np.stack([np.asarray((Z + g).kernel) for g in G], axis=1)
    Kinv = np.linalg.inv(K)
    f = np.zeros((6, 6, 6))
    for i, gi in enumerate(G):
        for j, gj in enumerate(G):
            f[i, j] = Kinv @ np.asarray((Z + (gi * gj - gj * gi).restrict[2]).kernel)
    return f
