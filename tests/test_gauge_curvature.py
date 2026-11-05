"""Phases 2-4 (spec sections 5-7) on the extensor implementation, plus cross-checks against the array reference."""
import numpy as np
import jax.numpy as jnp
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, fields as fl, versors as vs, gauge, covariant, curvature as cv
from gtg_csta._array import fields as afl, curvature as acv

X0 = al.point([0.1, -0.2, 0.3, 0.25])
nz = lambda e: np.asarray(e.kernel)


@pytest.mark.parametrize("seed", range(5))
def test_exp_matches_matrix_oracle_and_is_unit(seed):
    B = np.random.default_rng(seed).normal(size=15) * 0.5
    V = fl.bivector(B).exp()
    assert vs.unit_residual(V) < 1e-9
    assert np.abs(nz(V) - vs.exp_oracle(B)).max() < 1e-9


def test_double_cover_same_adjoint_action():
    V = fl.bivector(np.random.default_rng(3).normal(size=15) * 0.4).exp()
    X = fl.bivector(np.random.default_rng(4).normal(size=15))
    assert np.abs(nz(V * X * ~V) - nz((-V) * X * ~(-V))).max() < 1e-14


def test_adjoint_preserves_grade_and_inner_product():
    V = fl.bivector(np.random.default_rng(5).normal(size=15) * 0.4).exp()
    X, Y = (fl.bivector(np.random.default_rng(s).normal(size=15)) for s in (6, 7))
    VX, VY = V * X * ~V, V * Y * ~V
    assert float(jnp.abs((VX - VX.restrict[2]).kernel).max()) < 1e-12
    assert np.abs(nz((X * ~Y).restrict[0] - (VX * ~VY).restrict[0])).max() < 1e-11


def test_sandwich_operator_equals_explicit_product_on_maps():
    """`V >> Omega` (sandwich of a map's output) is V Omega ~V."""
    V = fl.random_versor_field(2)(X0)
    Om = fl.random_connection_field(1)(X0)
    assert np.abs(nz(V >> Om) - nz((V * Om * ~V).restrict[2])).max() < 1e-13


def test_spec_5_1_connection_stays_bivector():
    assert gauge.non_bivector_part(fl.random_connection_field(1), fl.random_versor_field(2), X0) < 1e-11


@pytest.mark.parametrize("method,tol", [("jax", 1e-11), ("fd", 1e-7)])
def test_spec_6_gauge_covariance_of_D(method, tol):
    Om, V, Psi = fl.random_connection_field(3), fl.random_versor_field(4), fl.random_spinor_field(5)
    Om2 = gauge.transform_connection(Om, V, method)
    Psi2 = lambda x: V(x) * Psi(x)
    lhs = covariant.covariant_derivative(Om2, Psi2, method)(X0)           # D' Psi'
    rhs = V(X0) * covariant.covariant_derivative(Om, Psi, method)(X0)     # V D Psi
    assert np.abs(nz(lhs - rhs)).max() < tol


@pytest.mark.parametrize("method,tol", [("jax", 1e-10), ("fd", 1e-6)])
def test_spec_7_1_commutator_of_D_equals_half_F(method, tol):
    Om, Psi = fl.random_connection_field(6), fl.random_spinor_field(7)
    C = cv.commutator_of_D(Om, Psi, method)(X0)
    F = cv.curvature(Om, method)(X0)
    assert np.abs(nz(C - 0.5 * (F * Psi(X0)))).max() < tol


def test_curvature_is_bivector_valued_and_antisymmetric():
    F = cv.curvature(fl.random_connection_field(8))(X0)
    assert F.gatype.arity == 2
    assert np.abs(nz(F + fl.permute_inputs(F, (1, 0)))).max() < 1e-13


def test_curvature_covariant_under_gauge():
    Om, V = fl.random_connection_field(9), fl.random_versor_field(10)
    F2 = cv.curvature(gauge.transform_connection(Om, V))(X0)
    expect = V(X0) >> cv.curvature(Om)(X0)
    assert np.abs(nz(F2 - expect)).max() < 1e-9


def test_pure_gauge_is_flat_and_generic_is_not():
    F_flat = cv.curvature(gauge.pure_gauge(fl.random_versor_field(11)))(X0)
    assert np.abs(nz(F_flat)).max() < 1e-9
    assert np.abs(nz(cv.curvature(fl.random_connection_field(12))(X0))).max() > 0.05


def test_bianchi_identity():
    r = cv.bianchi_residual(fl.random_connection_field(13))(X0)
    assert r.gatype.arity == 3
    assert np.abs(nz(r)).max() < 1e-9


def test_fd_matches_jax_curvature():
    Om = fl.random_connection_field(14)
    assert np.abs(nz(cv.curvature(Om, "jax")(X0) - cv.curvature(Om, "fd", 1e-3)(X0))).max() < 1e-8


def test_fd_convergence_order():
    Om = fl.random_connection_field(15)
    ref = cv.curvature(Om, "jax")(X0)
    e = [np.abs(nz(cv.curvature(Om, "fd", h)(X0) - ref)).max() for h in (0.2, 0.1)]
    assert e[0] / e[1] > 8


# ---- agreement with the independent array implementation ----

@pytest.mark.parametrize("seed", range(3))
def test_matches_array_reference_curvature(seed):
    F = cv.curvature(fl.random_connection_field(seed))(X0)
    Fa = acv.curvature(afl.random_connection_field(seed))(jnp.asarray(X0.kernel))      # [mu, nu, 15]
    assert np.abs(np.moveaxis(nz(F), 0, -1) - np.asarray(Fa)).max() < 1e-12


def test_sector_projectors_match_generator_coordinates():
    c = np.random.default_rng(0).normal(size=15)
    B = al.mv.bivector(al.to_blades(c))
    for name, s in al.SECTORS.items():
        ref = np.zeros(15)
        ref[s] = c[s]
        assert np.abs(np.asarray(al.to_generators(nz(al.sector(B, name)))) - ref).max() < 1e-12
    assert np.abs(sum(nz(al.sector(B, k)) for k in "MPKD") - nz(B)).max() < 1e-12
