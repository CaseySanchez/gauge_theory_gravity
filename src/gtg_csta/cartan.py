"""Cartan geometry from a tetrad, on extensors.

The coframe is the extensor  e : Spacetime(frame) <- Spacetime(coordinate),  e(d_nu) = e^a_nu e_a.
The metric is the form  g = e | e  (the algebra's own metric eta, pulled back), the frame vectors are
E = e.inverse(), the Christoffel symbols come from the derivative of g, and the Lorentz connection is the
bivector-valued map  Omega : Bivector <- Spacetime  with  1/2 [Omega(d_mu), v] = (nabla_mu v)^a e_a,
so that the curvature of Omega is the Riemann bivector map (checked against an independent
index implementation in tests).
"""
from __future__ import annotations

import jax.numpy as jnp

from . import algebra as al
from .fields import derivative, permute_inputs

S = al.Spacetime
CoframeType = al.ALG.gatype((S, S))                   # Spacetime <- Spacetime


def coframe_field(matrix_fn):
    """Wrap x4 -> (4,4) matrix [a, nu] as a field of extensors  x -> (Spacetime <- Spacetime)."""
    return lambda x: al.CTX.extensor(CoframeType, matrix_fn(x.kernel))


def metric(coframe):
    """x -> g = e | e : Scalar <- (Spacetime, Spacetime)."""
    return lambda x: coframe(x) | coframe(x)


def christoffel_lower(coframe):
    """x -> Gamma_{lam mu nu} = 1/2 (d_mu g_{lam nu} + d_nu g_{lam mu} - d_lam g_{mu nu}) as a trilinear form."""
    dg = derivative(metric(coframe))                  # inputs (alpha, beta, step gamma): d_gamma g_{alpha beta}

    def Gl(x):
        d = dg(x)
        return 0.5 * (permute_inputs(d, (0, 2, 1)) + d - permute_inputs(d, (2, 0, 1)))
    return Gl


def lorentz_connection(coframe):
    """x -> Omega(x): Bivector <- Spacetime.
    W(c, mu, b) = Gamma_{c mu b} + e_c . (d_mu E_b)  are the lowered spin-connection coefficients in the frame,
    and the bivector is  Omega(d_mu) = 1/2 W(e_k, mu, e_l) e^k ^ e^l  (two metric contractions)."""
    Gl = christoffel_lower(coframe)
    dE = derivative(lambda x: coframe(x).inverse())   # coordinate vector <- (frame b, step mu)

    def Om(x):
        e = coframe(x)
        E = e.inverse()
        W = Gl(x)(E, S, E) + permute_inputs(S | e(dE(x)), (0, 2, 1))     # Scalar <- (c, mu, b)
        B = (W * (S ^ S)).contract(1, 4).contract(2, 3)                  # Bivector <- mu
        return (0.5 * B).cast(al.Bivector)
    return Om


def painleve_gullstrand(M: float = 1.0):
    """Schwarzschild in Painleve-Gullstrand Cartesian coordinates (t,x,y,z):
    e^0 = dt,  e^i = dx^i + beta^i dt,  beta = sqrt(2M/r) x/r.   ds^2 = dt^2 - sum (e^i)^2."""
    def matrix(x):
        r = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
        beta = jnp.sqrt(2.0 * M / r) * x[1:] / r
        e = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        return e.at[1:, 0].set(beta).at[1:, 1:].set(jnp.eye(3))
    return coframe_field(matrix)


def frw(a=lambda t: t ** (2.0 / 3.0)):
    """Spatially flat FRW: e^0 = dt, e^i = a(t) dx^i (conformally flat, non-vacuum)."""
    return coframe_field(lambda x: jnp.diag(jnp.array([1.0, a(x[0]), a(x[0]), a(x[0])])))


def static_lapse(N=lambda x: 1.0 + 0.3 * jnp.sin(x[1]) * jnp.cos(x[2]) + 0.1 * x[3] ** 2):
    """e^0 = N(x) dt, e^i = dx^i: generically non-conformally-flat, non-vacuum."""
    return coframe_field(lambda x: jnp.diag(jnp.array([N(x), 1.0, 1.0, 1.0])))


# ---------------- time-dependent test geometries ----------------

def boosted(coframe_matrix_fn, v: float):
    """The same geometry moving with velocity v along +x: lab coordinates x, rest-frame coordinates y = L x
    (a boost in the t-x plane), coframe e'^a_nu(x) = e^a_mu(L x) L^mu_nu.  Takes and returns matrix functions
    (x4 -> (4,4))."""
    g = 1.0 / jnp.sqrt(1.0 - v * v)
    L = jnp.eye(4).at[0, 0].set(g).at[0, 1].set(-g * v).at[1, 0].set(-g * v).at[1, 1].set(g)
    return lambda x: coframe_matrix_fn(L @ x) @ L


def pg_matrix(M_of_t):
    """Painleve-Gullstrand coframe matrix with a mass M(t) that may depend on time:
    e^0 = dt, e^i = dx^i + sqrt(2 M(t)/r) x^i/r dt.  With constant M this is Schwarzschild (vacuum); with M(t)
    varying it is a valid torsion-free geometry with matter (a toy, not Vaidya)."""
    def matrix(x):
        r = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
        beta = jnp.sqrt(2.0 * M_of_t(x[0]) / r) * x[1:] / r
        e = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        return e.at[1:, 0].set(beta).at[1:, 1:].set(jnp.eye(3))
    return matrix
