"""Phases 2-4: spec sections 5-7."""
import numpy as np
import jax.numpy as jnp
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al
from gauge_theory_gravity._array import fields as fl
from gauge_theory_gravity._array import versors as vs
from gauge_theory_gravity._array import gauge, covariant, curvature as cv

X0 = jnp.array([0.1, -0.2, 0.3, 0.25])


@pytest.mark.parametrize("seed", range(5))
def test_exp_matches_matrix_oracle_and_is_unit(seed):
    B = np.random.default_rng(seed).normal(size=15) * 0.5
    V = vs.exp_bivector(B)
    assert vs.unit_residual(V) < 1e-9
    assert np.abs(np.asarray(V.kernel) - vs.exp_oracle(B)).max() < 1e-9


def test_double_cover_same_adjoint_action():
    B = np.random.default_rng(3).normal(size=15) * 0.4
    V = vs.exp_bivector(B)
    X = fl.biv(np.random.default_rng(4).normal(size=15))
    a = (V * X * ~V).kernel
    b = ((-V) * X * ~(-V)).kernel
    assert np.abs(np.asarray(a - b)).max() < 1e-14


def test_adjoint_preserves_grade_and_inner_product():
    V = vs.exp_bivector(np.random.default_rng(5).normal(size=15) * 0.4)
    X = fl.biv(np.random.default_rng(6).normal(size=15))
    Y = fl.biv(np.random.default_rng(7).normal(size=15))
    VX, VY = V * X * ~V, V * Y * ~V
    assert float(jnp.abs((VX - VX.restrict[2]).kernel).max()) < 1e-12
    s0 = (X * ~Y).restrict[0].kernel
    s1 = (VX * ~VY).restrict[0].kernel
    assert np.abs(np.asarray(s0 - s1)).max() < 1e-11


def test_spec_5_1_connection_stays_bivector():
    Om, V = fl.random_connection_field(1), fl.random_versor_field(2)
    assert gauge.transform_connection_residual(Om, V, X0) < 1e-11


@pytest.mark.parametrize("method,tol", [("jax", 1e-11), ("fd", 1e-7)])
def test_spec_6_gauge_covariance_of_D(method, tol):
    Om, V, Psi = fl.random_connection_field(3), fl.random_versor_field(4), fl.random_spinor_field(5)
    Om2 = gauge.transform_connection(Om, V, method)
    Psi2 = lambda x: (fl.even(V(x)) * fl.even(Psi(x))).kernel
    lhs = covariant.covariant_derivative(Om2, Psi2, method)(X0)           # D' Psi'
    D = covariant.covariant_derivative(Om, Psi, method)(X0)
    rhs = (fl.even(V(X0)) * fl.even(D)).kernel                            # V D Psi
    assert np.abs(np.asarray(lhs - rhs)).max() < tol


@pytest.mark.parametrize("method,tol", [("jax", 1e-10), ("fd", 1e-6)])
def test_spec_7_1_commutator_of_D_equals_half_F(method, tol):
    Om, Psi = fl.random_connection_field(6), fl.random_spinor_field(7)
    C = cv.commutator_of_D(Om, Psi, method)(X0)
    F = cv.curvature(Om, method)(X0)
    rhs = (0.5 * fl.biv(F) * fl.even(Psi(X0))).kernel
    assert np.abs(np.asarray(C - rhs)).max() < tol


def test_curvature_is_bivector_valued_and_antisymmetric():
    Om = fl.random_connection_field(8)
    F = np.asarray(cv.curvature(Om)(X0))
    assert F.shape == (4, 4, 15)
    assert np.abs(F + F.transpose(1, 0, 2)).max() < 1e-13
    assert np.abs(F[np.arange(4), np.arange(4)]).max() < 1e-13


def test_curvature_covariant_under_gauge():
    Om, V = fl.random_connection_field(9), fl.random_versor_field(10)
    Om2 = gauge.transform_connection(Om, V)
    F = fl.biv(cv.curvature(Om)(X0))
    F2 = cv.curvature(Om2)(X0)
    expect = (fl.even(V(X0)) * F * ~fl.even(V(X0))).restrict[2].kernel
    assert np.abs(np.asarray(F2 - expect)).max() < 1e-9


def test_pure_gauge_is_flat_and_generic_is_not():
    V = fl.random_versor_field(11)
    F_flat = np.asarray(cv.curvature(gauge.pure_gauge(V))(X0))
    assert np.abs(F_flat).max() < 1e-9
    F_gen = np.asarray(cv.curvature(fl.random_connection_field(12))(X0))
    assert np.abs(F_gen).max() > 0.05


def test_bianchi_identity():
    Om = fl.random_connection_field(13)
    r = np.asarray(cv.bianchi_residual(Om)(X0))
    assert np.abs(r).max() < 1e-9


def test_fd_matches_jax_curvature():
    Om = fl.random_connection_field(14)
    a = np.asarray(cv.curvature(Om, "jax")(X0))
    b = np.asarray(cv.curvature(Om, "fd", 1e-3)(X0))
    assert np.abs(a - b).max() < 1e-8


def test_fd_convergence_order():
    """Errors of the 4th-order stencil should drop ~16x when h halves."""
    Om = fl.random_connection_field(15)
    ref = np.asarray(cv.curvature(Om, "jax")(X0))
    e = [np.abs(np.asarray(cv.curvature(Om, "fd", h)(X0)) - ref).max() for h in (0.2, 0.1)]
    assert e[0] / e[1] > 8
