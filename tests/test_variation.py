"""Plan for 11.4, phase 1: field equations of coframe densities by the pointwise Euler-Lagrange operator.

`variation.euler_lagrange` gives E[a, mu] = delta S / delta e^a_mu.  Checked against independent index computations
from the array reference's Christoffel symbols:  Einstein-Hilbert gives E = -2 sqrt(g) e_{a nu} G^{mu nu}
(delta S / delta g_{mu nu} = -sqrt(g) G^{mu nu}), Weyl squared gives E = -8 sqrt(g) e_{a nu} B^{mu nu} with the Bach
tensor B (delta S / delta g_{mu nu} = -4 sqrt(g) B^{mu nu}).  Both satisfy the local Lorentz identity, Weyl squared
also the trace identity of its Weyl invariance, and Bach vanishes on Einstein and on conformally flat metrics.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, variation as vr
from gauge_theory_gravity._array import cartan as ac, conformal as acf

ETA = np.diag([1.0, -1.0, -1.0, -1.0])
X_LAPSE = [1.3, 0.4, -0.7, 0.9]


# --- independent references: index formulas on the array implementation -------------------------------------------
def _schouten(coframe):
    g, riemann = ac.metric(coframe), ac.riemann_coordinate(coframe)

    def P(x):
        metric = g(x)
        ricci = jnp.einsum("rsrn->sn", riemann(x))
        return 0.5 * (ricci - jnp.einsum("mn,mn", jnp.linalg.inv(metric), ricci) * metric / 6)
    return P


def einstein_reference(coframe, x4):
    """-2 sqrt(g) e_{a nu} G^{mu nu}, an array [a, mu]."""
    x = jnp.asarray(x4)
    metric, e = np.asarray(ac.metric(coframe)(x)), np.asarray(coframe(x))
    inverse = np.linalg.inv(metric)
    ricci = np.asarray(ac.ricci_coordinate(coframe)(x))
    einstein = ricci - 0.5 * np.einsum("mn,mn", inverse, ricci) * metric
    return -2 * abs(np.linalg.det(e)) * ETA @ e @ inverse @ einstein @ inverse


def bach_reference(coframe, x4):
    """-8 sqrt(g) e_{a nu} B^{mu nu},  B_ab = nabla^c nabla_c P_ab - nabla^c nabla_a P_bc + P^cd C_acbd."""
    g, symbols, riemann, P = ac.metric(coframe), ac.christoffel(coframe), ac.riemann_coordinate(coframe), _schouten(coframe)

    def dP(x):                                                     # T[a, b, k] = nabla_k P_ab
        G, p = symbols(x), P(x)
        return jax.jacfwd(P)(x) - jnp.einsum("lka,lb->abk", G, p) - jnp.einsum("lkb,al->abk", G, p)

    x = jnp.asarray(x4)
    G, T = symbols(x), dP(x)
    Q = (jax.jacfwd(dP)(x) - jnp.einsum("lma,lbk->abkm", G, T) - jnp.einsum("lmb,alk->abkm", G, T)
         - jnp.einsum("lmk,abl->abkm", G, T))                     # Q[a, b, k, m] = nabla_m nabla_k P_ab
    metric, p = g(x), P(x)
    inverse = jnp.linalg.inv(metric)
    lowered = jnp.einsum("re,esmn->rsmn", metric, riemann(x))
    weyl = lowered - (jnp.einsum("ac,bd->abcd", metric, p) - jnp.einsum("ad,bc->abcd", metric, p)
                      + jnp.einsum("bd,ac->abcd", metric, p) - jnp.einsum("bc,ad->abcd", metric, p))
    bach = (jnp.einsum("cd,abcd->ab", inverse, Q) - jnp.einsum("cd,bcad->ab", inverse, Q)
            + jnp.einsum("cd,acbd->ab", inverse @ p @ inverse, weyl))
    e, inverse = np.asarray(coframe(x)), np.asarray(inverse)
    return -8 * abs(np.linalg.det(e)) * ETA @ e @ inverse @ np.asarray(bach) @ inverse


def _coframe_at(geometry, x4):
    return al.components(geometry(al.point(x4)))


# --- Einstein-Hilbert ----------------------------------------------------------------------------------------------
def test_einstein_hilbert_gives_the_einstein_tensor():
    equations = vr.euler_lagrange(vr.einstein_hilbert, cartan.static_lapse(), X_LAPSE)
    reference = einstein_reference(acf.static_lapse(), X_LAPSE)
    assert np.abs(equations - reference).max() < 1e-13 and np.abs(reference).max() > 0.1
    assert vr.lorentz_residual(equations, _coframe_at(cartan.static_lapse(), X_LAPSE)) < 1e-13


def test_einstein_hilbert_vanishes_on_schwarzschild():
    assert np.abs(vr.euler_lagrange(vr.einstein_hilbert, cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1])).max() < 1e-13


@pytest.mark.slow
def test_einstein_hilbert_gives_the_einstein_tensor_on_frw():
    x4 = [1.3, 0.4, -0.7, 0.9]
    reference = einstein_reference(acf.frw(), x4)
    assert np.abs(vr.euler_lagrange(vr.einstein_hilbert, cartan.friedmann(), x4) - reference).max() < 1e-12


# --- Weyl squared: the Bach equations ------------------------------------------------------------------------------
def test_weyl_squared_gives_the_bach_tensor():
    equations = vr.euler_lagrange(vr.weyl_squared, cartan.static_lapse(), X_LAPSE)
    reference = bach_reference(acf.static_lapse(), X_LAPSE)
    assert np.abs(equations - reference).max() < 1e-13 and np.abs(reference).max() > 0.1
    coframe = _coframe_at(cartan.static_lapse(), X_LAPSE)
    assert vr.lorentz_residual(equations, coframe) < 1e-13
    assert abs(vr.weyl_trace(equations, coframe)) < 1e-13                       # Weyl invariance: B is trace-free


def test_bach_vanishes_on_schwarzschild():
    """Einstein metrics are Bach-flat."""
    assert np.abs(vr.euler_lagrange(vr.weyl_squared, cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1])).max() < 1e-13


@pytest.mark.slow
@pytest.mark.parametrize("name,geometry,x4", [
    ("frw", cartan.friedmann(), [1.3, 0.4, -0.7, 0.9]),                          # conformally flat
    ("de_sitter", cartan.de_sitter(0.5), [0.7, 0.4, -0.7, 0.9]),                 # Einstein with Lambda, conformally flat
    ("kerr", cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3]),                     # Einstein
])
def test_bach_vanishes(name, geometry, x4):
    assert np.abs(vr.euler_lagrange(vr.weyl_squared, geometry, x4)).max() < 1e-12
