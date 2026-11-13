"""Cl(2,4;R) with the spec's metric diag(+,-,-,-,+,-), null basis and the 15 conformal generators.

Basis vectors: t,x,y,z (spacetime) and p (= e_+, square +1), n (= e_-, square -1).
Convention fixed once here; all other modules import from this one.
"""
from __future__ import annotations

import numpy as np
from numga import Algebra
from numga.backend.jax import JaxContext

SIGNATURE = "t+x-y-z-p+n-"
ALG = Algebra(SIGNATURE)
mv = JaxContext(ALG, np.float64).multivector

t, x, y, z, p, n = mv.t, mv.x, mv.y, mv.z, mv.p, mv.n
SPACETIME = [t, x, y, z]
ETA4 = np.diag([1.0, -1.0, -1.0, -1.0])                     # spacetime metric
ETA6 = np.diag([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])

n_o = (p + n) / np.sqrt(2.0)
n_inf = (p - n) / np.sqrt(2.0)

# Spec ordering of the 15 generators: M_ab (a<b), P_a, K_a, D.
M_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
LABELS = (
    [f"M_{'txyz'[a]}{'txyz'[b]}" for a, b in M_PAIRS]
    + [f"P_{c}" for c in "txyz"]
    + [f"K_{c}" for c in "txyz"]
    + ["D"]
)
SECTORS = {"M": slice(0, 6), "P": slice(6, 10), "K": slice(10, 14), "D": slice(14, 15)}


ZERO_B = mv.bivector(np.zeros(15))


def _kernel(b) -> np.ndarray:
    """Coefficients of a bivector-typed extensor in the full 15-blade bivector layout."""
    return np.asarray((ZERO_B + b).kernel)


def generators():
    """The 15 generator bivectors as numga extensors, in the order of LABELS."""
    gens = [SPACETIME[a] ^ SPACETIME[b] for a, b in M_PAIRS]
    gens += [e ^ n_inf for e in SPACETIME]
    gens += [e ^ n_o for e in SPACETIME]
    gens += [n_o ^ n_inf]
    return gens


GENS = generators()
# GEN[:, i] = blade coefficients of generator i; blade_coeffs = GEN @ generator_coeffs.
GEN = np.stack([_kernel(g) for g in GENS], axis=1)
GEN_INV = np.linalg.inv(GEN)


def to_blades(c):
    """Generator coordinates (..., 15) -> numga bivector blade coefficients (..., 15)."""
    return np.einsum("ij,...j->...i", GEN, c)


def to_generators(b):
    """numga bivector blade coefficients (..., 15) -> generator coordinates (..., 15)."""
    return np.einsum("ij,...j->...i", GEN_INV, b)


def bivector(c_gen):
    """numga bivector from generator coordinates."""
    return mv.bivector(to_blades(np.asarray(c_gen)))


def structure_constants() -> np.ndarray:
    """f[i,j,k] with [G_i, G_j] = XY - YX = sum_k f[i,j,k] G_k (full commutator)."""
    f = np.zeros((15, 15, 15))
    for i, gi in enumerate(GENS):
        for j, gj in enumerate(GENS):
            comm = (gi * gj - gj * gi)
            f[i, j] = to_generators(_kernel(comm.restrict[2]))
    return f


def scalar(a) -> float:
    """The value of a scalar extensor, as a float."""
    return float(np.asarray(a.kernel).reshape(-1)[0])


# ---------------------------------------------------------------------------------------------
# Extensor types.  Everything downstream is written in terms of these.
# ---------------------------------------------------------------------------------------------
from numga.gatype import Orientation  # noqa: E402

_EXT = Orientation.extrinsic
CTX = mv.context
Scalar = ALG.gatype.scalar()
Vector = ALG.gatype.vector()                      # all six directions
Bivector = ALG.gatype.bivector()                  # the 15 generators of so(2,4)
Even = ALG.gatype.even()                          # Spin(2,4) lives here
Spacetime = ALG.gatype(ALG.subspace.from_blades("t x y z"), orientation=_EXT)          # the 4 directions a, x
SpacetimeBivector = ALG.gatype(ALG.subspace.from_blades("tx ty tz xy xz yz"), orientation=_EXT)   # Lorentz sector
ConnectionType = ALG.gatype((Bivector, Spacetime))                   # Omega: Bivector <- Spacetime
CurvatureType = ALG.gatype((Bivector, Spacetime, Spacetime))         # F: Bivector <- (Spacetime, Spacetime)


def point(x4):
    """A spacetime position / direction (4 coefficients t,x,y,z) as an extensor."""
    return CTX.extensor(Spacetime, CTX.xp.asarray(x4))


def connection_from_columns(cols):
    """Omega from its values on the coordinate vectors: cols (4, 15) blade coefficients -> Bivector <- Spacetime."""
    return CTX.extensor(ConnectionType, CTX.xp.asarray(cols).T)


def connection_columns(Omega):
    """(4, 15) blade coefficients of Omega(e_mu), mu = t,x,y,z."""
    return Omega.kernel.T


# ----- sector projectors (Bivector <- Bivector), built from the null basis -----
# B = B_M + u ^ n_inf + v ^ n_o + b n_o ^ n_inf   with u, v spacetime vectors and B_M a Lorentz bivector:
#   n_o | B   = -u - b n_o,     n_inf | B = -v + b n_inf,     b = -n_inf | (n_o | B).
D_PART = (n_o ^ n_inf) * (-(n_inf | (n_o | Bivector)))                  # dilation
P_PART = -((n_o | Bivector) ^ n_inf) - D_PART                           # translations  u ^ n_inf
K_PART = -((n_inf | Bivector) ^ n_o) - D_PART                           # special conformal v ^ n_o
M_PART = Bivector - P_PART - K_PART - D_PART                            # Lorentz
SECTOR_MAPS = {"M": M_PART, "P": P_PART, "K": K_PART, "D": D_PART}


def sector(B, name):
    """The `name` ('M','P','K','D') part of a bivector or bivector-valued map B."""
    return SECTOR_MAPS[name](B).cast(Bivector)


def sector_size(B, name) -> float:
    """Largest coefficient magnitude of a sector of B (any batch/inputs)."""
    return float(np.abs(np.asarray(sector(B, name).kernel)).max())
