"""Phase 5: spec section 8."""
import numpy as np
import jax.numpy as jnp
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import fields as fl, gauge, curvature as cv, transport as tp, versors as vs

X0 = jnp.array([0.1, -0.2, 0.3, 0.25])


def straight(a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    return lambda s: a + s * (b - a)


def test_transport_is_a_spin_versor():
    Om = fl.random_connection_field(1, 0.4)
    T = tp.transport(Om, straight(X0, X0 + jnp.array([0.4, 0.3, -0.2, 0.5])), 100)
    assert vs.unit_residual(fl.even(T)) < 1e-10


def test_pure_gauge_transport_equals_V2_V1inv():
    V = fl.random_versor_field(2, 0.3)
    Om = gauge.pure_gauge(V)
    x1, x2 = X0, X0 + jnp.array([0.3, -0.2, 0.2, 0.4])
    T = tp.transport(Om, straight(x1, x2), 400, scheme="magnus4")
    exact = (fl.even(V(x2)) * ~fl.even(V(x1))).kernel
    assert np.abs(np.asarray(T - exact)).max() < 1e-6


def test_convergence_order():
    V = fl.random_versor_field(3, 0.3)
    Om = gauge.pure_gauge(V)
    x2 = X0 + jnp.array([0.3, -0.2, 0.2, 0.4])
    exact = np.asarray((fl.even(V(x2)) * ~fl.even(V(X0))).kernel)
    err = lambda n, sch: np.abs(np.asarray(tp.transport(Om, straight(X0, x2), n, scheme=sch)) - exact).max()
    assert err(20, "midpoint") / err(40, "midpoint") > 3.3      # ~4: second order
    assert err(10, "magnus4") / err(20, "magnus4") > 12          # ~16: fourth order


def test_reparametrisation_invariance():
    Om = fl.random_connection_field(4, 0.4)
    a, b = X0, X0 + jnp.array([0.4, 0.3, -0.2, 0.5])
    lin = tp.transport(Om, straight(a, b), 300, scheme="magnus4")
    warped = tp.transport(Om, lambda s: a + (s ** 2) * (b - a), 600, scheme="magnus4")
    assert np.abs(np.asarray(lin - warped)).max() < 1e-6


def test_pure_gauge_holonomy_trivial_but_generic_not():
    V = fl.random_versor_field(5, 0.3)
    one = np.eye(1, 32)[0]
    h0 = tp.holonomy_loop(gauge.pure_gauge(V), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(np.asarray(h0) - one).max() < 1e-6
    h1 = tp.holonomy_loop(fl.random_connection_field(6, 0.4), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(np.asarray(h1) - one).max() > 1e-2


@pytest.mark.parametrize("plane", [(0, 1), (1, 2), (0, 3)])
def test_holonomy_matches_curvature_to_leading_order(plane):
    """T_loop = exp(-sigma/2 * eps^2 F_mu_nu) + O(eps^3), with a single universal sign sigma."""
    mu, nu = plane
    Om = fl.random_connection_field(7, 0.4)
    F = np.asarray(cv.curvature(Om)(X0))[mu, nu]
    Fc = fl.biv(F)
    res = {}
    for eps in (0.08, 0.04):
        T = tp.holonomy_loop(Om, X0, mu, nu, eps, 30, "magnus4")
        pred = {s: np.asarray((s * 0.5 * eps ** 2 * Fc).exp().kernel) for s in (+1, -1)}
        res[eps] = {s: np.abs(np.asarray(T) - pred[s]).max() for s in (+1, -1)}
    # exactly one sign matches, and the match improves ~ eps^3 (x8) with the small step
    best = min((+1, -1), key=lambda s: res[0.04][s])
    assert res[0.04][best] < 0.2 * res[0.04][-best]
    assert res[0.08][best] / res[0.04][best] > 5
    # record: the sign must be the same in every plane (checked via module-level state)
    test_holonomy_matches_curvature_to_leading_order.signs = getattr(
        test_holonomy_matches_curvature_to_leading_order, "signs", set()) | {best}
    assert len(test_holonomy_matches_curvature_to_leading_order.signs) == 1
