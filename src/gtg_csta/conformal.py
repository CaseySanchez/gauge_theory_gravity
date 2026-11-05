"""Tier B: translation sector as coframe; normal conformal Cartan connection (spec 9.2, 10), on extensors.

    Omega = Omega_Lorentz  +  e ^ n_inf  +  f ^ n_o  +  b (n_o ^ n_inf)

with e the coframe (Spacetime <- Spacetime), f another such map (the special-conformal part) and b a form
(the dilation part).  Each term is a map Bivector <- Spacetime built by one wedge product.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import algebra as al
from . import cartan
from .curvature import curvature

S = al.Spacetime
ALL = al.ALG.gatype((S, S))


def connection(coframe, f_fn=None, b_fn=None):
    """x -> Omega(x): Bivector <- Spacetime.  f_fn(x): Spacetime <- Spacetime;  b_fn(x): Scalar <- Spacetime."""
    lorentz = cartan.lorentz_connection(coframe)
    DIL = al.n_o ^ al.n_inf

    def Om(x):
        e = coframe(x)
        out = lorentz(x) + (e ^ al.n_inf).cast(al.Bivector)
        if f_fn is not None:
            out = out + (f_fn(x) ^ al.n_o).cast(al.Bivector)
        if b_fn is not None:
            out = out + (b_fn(x) * DIL).cast(al.Bivector)
        return out
    return Om


def ricci_vector(F, E):
    """Ric(nu) = sum_a e^a _| F_M(E_a, nu):  the trace of the Lorentz-sector curvature over its first
    coordinate input and one frame index; a map Vector <- Spacetime.  (= R^b_nu, frame index up.)"""
    FM = al.sector(F, "M")
    return (S | FM(E, S)).contract(1, 2)


def normal_residual_arrays(coframe, x, fk):
    """Residual of the normal conditions for a pointwise-constant f given as a (4,4) kernel:
    F_D (all components) and the trace Ric of F_M.  Affine in f."""
    f = al.CTX.extensor(ALL, fk)
    F = curvature(connection(coframe, lambda _x: f))(x)
    E = coframe(x).inverse()
    return jnp.concatenate([al.sector(F, "D").kernel.reshape(-1), ricci_vector(F, E).kernel.reshape(-1)])


def solve_normal_f(coframe, x):
    """Least-squares f (Spacetime <- Spacetime) making the normal conditions hold; also the residual and
    the rank of the (16-unknown) linear system."""
    f0 = jnp.zeros((4, 4))
    r0 = normal_residual_arrays(coframe, x, f0)
    J = jax.jacfwd(lambda v: normal_residual_arrays(coframe, x, v.reshape(4, 4)))(f0.reshape(-1))
    sol, *_ = jnp.linalg.lstsq(J, -r0)
    res = normal_residual_arrays(coframe, x, sol.reshape(4, 4))
    return (al.CTX.extensor(ALL, sol.reshape(4, 4)), float(jnp.abs(res).max()), int(jnp.linalg.matrix_rank(J)))


def schouten(coframe):
    """x -> P = 1/2 (Ric - R e / 6): Spacetime <- Spacetime (frame index up, coordinate index down), from
    the curvature of the Lorentz connection alone."""
    lorentz = cartan.lorentz_connection(coframe)
    F = curvature(lorentz)

    def P(x):
        e = coframe(x)
        E = e.inverse()
        T = ricci_vector(F(x), E).cast(S)                       # R^b_nu
        R = T(E).trace()                                         # scalar curvature
        return 0.5 * (T - (R / 6.0) * e)
    return P


def ricci_scalar(coframe, x):
    e = coframe(x)
    E = e.inverse()
    F = curvature(cartan.lorentz_connection(coframe))(x)
    return ricci_vector(F, E).cast(S)(E).trace()
