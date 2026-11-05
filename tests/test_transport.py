"""Phase 5 (spec section 8) on the extensor implementation."""
import numpy as np
import jax.numpy as jnp
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, fields as fl, gauge, curvature as cv, transport as tp, versors as vs
from gtg_csta._array import fields as afl, transport as atp

X0 = jnp.array([0.1, -0.2, 0.3, 0.25])


def straight(a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    return lambda s: al.point(a + s * (b - a))


def test_transport_is_a_spin_versor():
    T = tp.transport(fl.random_connection_field(1, 0.4), straight(X0, X0 + jnp.array([0.4, 0.3, -0.2, 0.5])), 100)
    assert vs.unit_residual(fl.even(T)) < 1e-10


def test_pure_gauge_transport_equals_V2_V1inv():
    V = fl.random_versor_field(2, 0.3)
    x1, x2 = X0, X0 + jnp.array([0.3, -0.2, 0.2, 0.4])
    T = tp.transport(gauge.pure_gauge(V), straight(x1, x2), 400, scheme="magnus4")
    exact = V(al.point(x2)) * ~V(al.point(x1))
    assert np.abs(np.asarray(T - exact.kernel)).max() < 1e-6


def test_convergence_order():
    V = fl.random_versor_field(3, 0.3)
    Om = gauge.pure_gauge(V)
    x2 = X0 + jnp.array([0.3, -0.2, 0.2, 0.4])
    exact = np.asarray((V(al.point(x2)) * ~V(al.point(X0))).kernel)
    err = lambda n, sch: np.abs(np.asarray(tp.transport(Om, straight(X0, x2), n, scheme=sch)) - exact).max()
    assert err(20, "midpoint") / err(40, "midpoint") > 3.3
    assert err(10, "magnus4") / err(20, "magnus4") > 12


def test_reparametrisation_invariance():
    Om = fl.random_connection_field(4, 0.4)
    a, b = X0, X0 + jnp.array([0.4, 0.3, -0.2, 0.5])
    lin = tp.transport(Om, straight(a, b), 300, scheme="magnus4")
    warped = tp.transport(Om, lambda s: al.point(a + (s ** 2) * (b - a)), 600, scheme="magnus4")
    assert np.abs(np.asarray(lin - warped)).max() < 1e-6


def test_pure_gauge_holonomy_trivial_but_generic_not():
    one = np.eye(1, 32)[0]
    h0 = tp.holonomy_loop(gauge.pure_gauge(fl.random_versor_field(5, 0.3)), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(np.asarray(h0) - one).max() < 1e-6
    h1 = tp.holonomy_loop(fl.random_connection_field(6, 0.4), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(np.asarray(h1) - one).max() > 1e-2


@pytest.mark.parametrize("plane", [(0, 1), (1, 2), (0, 3)])
def test_holonomy_is_exp_minus_half_curvature(plane):
    """T_loop = exp(-1/2 eps^2 F(e_mu, e_nu)) + O(eps^3) (orientation: +mu, +nu, -mu, -nu)."""
    mu, nu = plane
    Om = fl.random_connection_field(7, 0.4)
    F = cv.curvature(Om)(al.point(X0))
    Fmn = F(al.point(jnp.eye(4)[mu]), al.point(jnp.eye(4)[nu]))
    errs = {}
    for eps in (0.08, 0.04):
        T = tp.holonomy_loop(Om, X0, mu, nu, eps, 30, "magnus4")
        errs[eps] = {s: np.abs(np.asarray(T) - np.asarray((s * 0.5 * eps ** 2 * Fmn).exp().kernel)).max()
                     for s in (+1, -1)}
    assert errs[0.04][-1] < 0.2 * errs[0.04][+1]               # the sign is -1
    assert errs[0.08][-1] / errs[0.04][-1] > 5                 # error ~ eps^3


def test_matches_array_reference_transport():
    x1 = X0 + jnp.array([0.4, 0.3, -0.2, 0.5])
    T = tp.transport(fl.random_connection_field(1, 0.4), straight(X0, x1), 60, scheme="magnus4")
    Ta = atp.transport(afl.random_connection_field(1, 0.4), lambda s: X0 + s * (x1 - X0), 60, scheme="magnus4")
    assert np.abs(np.asarray(T - Ta)).max() < 1e-12
