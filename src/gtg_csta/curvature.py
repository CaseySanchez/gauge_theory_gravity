"""Curvature (spec section 7), on extensors."""
from __future__ import annotations

from . import algebra as al
from .fields import get_derivative, permute_inputs, even
import jax.numpy as jnp


def curvature(omega, method: str = "jax", h: float = 1e-4):
    """F(a,b) = a.grad Omega(b) - b.grad Omega(a) + 1/2 [Omega(a), Omega(b)]   (constant coordinate
    vectors a,b, so the -Omega([a,b]) term drops; [X,Y] = XY - YX).
    Returns x -> a map  Bivector <- (Spacetime, Spacetime).

    d = derivative(Omega) has the inputs (b, step a) = a.grad Omega(b); swapping them gives the other term,
    and numga's `commutator` is (XY - YX)/2, exactly the 1/2 [Omega(a), Omega(b)] of the spec."""
    dOm = get_derivative(method, h)(omega)

    def F(x):
        Om = omega(x)
        d = dOm(x)                                 # Bivector <- (b, a): a . grad Omega(b)
        return permute_inputs(d, (1, 0)) - d + Om.commutator(Om)
    return F


def commutator_of_D(omega, Psi, method: str = "jax", h: float = 1e-4):
    """[D_a, D_b] Psi computed from the covariant derivative itself (second derivatives of Psi cancel):
    a map Even <- (Spacetime a, Spacetime b)."""
    from .covariant import covariant_derivative
    dd = get_derivative(method, h)
    D = covariant_derivative(omega, Psi, method, h)
    dD = dd(D)

    def C(x):
        # D_a(D_b Psi) = a.grad(D_b Psi) + 1/2 Omega(a) (D_b Psi); inputs (b, a) -> reorder to (a, b)
        DbPsi = D(x)                                # Even <- b
        Om_Db = permute_inputs(omega(x) * DbPsi, (1, 0))      # Omega(a) D_b Psi, inputs reordered to (b, a)
        T = dD(x) + 0.5 * Om_Db                     # Even <- (b, a)   [D_a D_b Psi]
        DaDb = permute_inputs(T, (1, 0))            # Even <- (a, b):  D_a D_b Psi
        return DaDb - permute_inputs(DaDb, (1, 0))
    return C


def bianchi_residual(omega, method: str = "jax", h: float = 1e-4):
    """Cyclic sum over (l,a,b) of  D_l F(a,b) = d_l F(a,b) + 1/2 [Omega(l), F(a,b)]; must vanish.
    Returns x -> a map Bivector <- (Spacetime, Spacetime, Spacetime)."""
    F = curvature(omega, method, h)
    dF = get_derivative(method, h)(F)

    def R(x):
        T = permute_inputs(dF(x), (2, 0, 1)) + omega(x).commutator(F(x))      # inputs (l, a, b)
        return T + permute_inputs(T, (2, 0, 1)) + permute_inputs(T, (1, 2, 0))
    return R


def sector_split(F):
    """{M,P,K,D} parts of a bivector-valued extensor, by the null-basis projectors."""
    return {k: al.sector(F, k) for k in al.SECTOR_MAPS}
