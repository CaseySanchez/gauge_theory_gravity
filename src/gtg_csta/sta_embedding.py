"""The Lorentz sector: Cl(1,3) inside Cl(2,4), and the sign map to numga's own STA (spec section 9.1)."""
from __future__ import annotations

import numpy as np
from numga import Algebra
from numga.backend.jax import JaxContext
from numga.extensor import Extensor

from . import algebra as al

# The spacetime bivectors, the Lorentz sector: a connection with only these is `e = f = b = 0`.
LorentzConnection = al.ALG.gatype((al.SpacetimeBivector, al.Spacetime))          # SpacetimeBivector <- Spacetime


def lorentz_embedding(omega: Extensor) -> Extensor:
    """A map into the spacetime bivectors as a connection of the full algebra: Bivector <- Spacetime."""
    return omega.cast(al.Bivector)


def lorentz_rotor(bivector: Extensor) -> Extensor:
    """The Lorentz rotor exp(B) of a spacetime bivector B."""
    return bivector.exp()


def sector_leakage(strength: Extensor) -> dict[str, float]:
    """The largest component of a bivector-valued extensor outside the Lorentz sector, per sector P, K, D."""
    return {name: al.sector_size(strength, name) for name in "PKD"}


def numga_sta_structure_constants() -> np.ndarray:
    """f_sta[i,j,k] for M_ab = e_a ^ e_b in numga's own STA 'x+y+z+t-' (opposite sign to ours),
    generators ordered as al.M_PAIRS over (t,x,y,z)."""
    sta = JaxContext(Algebra("x+y+z+t-"), np.float64).multivector
    basis = [sta.t, sta.x, sta.y, sta.z]
    generators = [basis[a] ^ basis[b] for a, b in al.M_PAIRS]
    zero = sta.bivector(np.zeros(6))
    columns = np.stack([np.asarray((zero + g).kernel) for g in generators], axis=1)
    constants = np.zeros((6, 6, 6))
    for i, gi in enumerate(generators):
        for j, gj in enumerate(generators):
            commutator = (zero + (gi * gj - gj * gi).restrict[2]).kernel
            constants[i, j] = np.linalg.solve(columns, np.asarray(commutator))
    return constants
