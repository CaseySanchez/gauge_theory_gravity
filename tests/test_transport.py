"""Phase 5 (spec section 8) on the extensor implementation."""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from adapters import array_connection, array_path
from gauge_theory_gravity import algebra as al, fields as fl, gauge, curvature as cv, transport as tp, versors as vs
from gauge_theory_gravity._array import transport as atp

X0 = al.point([0.1, -0.2, 0.3, 0.25])
nz = lambda e: np.asarray(e.kernel)


def test_transport_is_a_spin_versor():
    path = tp.straight_line(X0, X0 + al.point([0.4, 0.3, -0.2, 0.5]))
    assert vs.unit_residual(tp.transport(fl.random_connection_field(1, 0.4), path, 100)) < 1e-10


def test_pure_gauge_transport_equals_V2_V1inv():
    V = fl.random_versor_field(2, 0.3)
    X1 = X0 + al.point([0.3, -0.2, 0.2, 0.4])
    T = tp.transport(gauge.pure_gauge(V), tp.straight_line(X0, X1), 400, scheme="magnus4")
    assert np.abs(nz(T - V(X1) * ~V(X0))).max() < 1e-6


def test_convergence_order():
    V = fl.random_versor_field(3, 0.3)
    Om = gauge.pure_gauge(V)
    X1 = X0 + al.point([0.3, -0.2, 0.2, 0.4])
    exact = V(X1) * ~V(X0)

    def error(steps, scheme):
        return np.abs(nz(tp.transport(Om, tp.straight_line(X0, X1), steps, scheme=scheme) - exact)).max()

    assert error(20, "midpoint") / error(40, "midpoint") > 3.3
    assert error(10, "magnus4") / error(20, "magnus4") > 12


def test_reparametrization_invariance():
    Om = fl.random_connection_field(4, 0.4)
    X1 = X0 + al.point([0.4, 0.3, -0.2, 0.5])

    def squared(parameter):
        return X0 + parameter * parameter * (X1 - X0)

    straight = tp.transport(Om, tp.straight_line(X0, X1), 300, scheme="magnus4")
    warped = tp.transport(Om, squared, 600, scheme="magnus4")
    assert np.abs(nz(straight - warped)).max() < 1e-6


def test_pure_gauge_holonomy_trivial_but_generic_not():
    flat = tp.holonomy_loop(gauge.pure_gauge(fl.random_versor_field(5, 0.3)), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(nz(flat - 1)).max() < 1e-6
    curved = tp.holonomy_loop(fl.random_connection_field(6, 0.4), X0, 0, 1, 0.5, 100, "magnus4")
    assert np.abs(nz(curved - 1)).max() > 1e-2


@pytest.mark.parametrize("plane", [(0, 1), (1, 2), (0, 3)])
def test_holonomy_is_exp_minus_half_curvature(plane):
    """T_loop = exp(-1/2 eps^2 F(e_mu, e_nu)) + O(eps^3) (orientation: +mu, +nu, -mu, -nu)."""
    mu, nu = plane
    Om = fl.random_connection_field(7, 0.4)
    F = cv.curvature(Om)(X0)(al.DIRECTIONS[mu], al.DIRECTIONS[nu])
    errors = {}
    for size in (0.08, 0.04):
        T = tp.holonomy_loop(Om, X0, mu, nu, size, 30, "magnus4")
        errors[size] = {sign: np.abs(nz(T - (sign * 0.5 * size ** 2 * F).exp())).max() for sign in (+1, -1)}
    assert errors[0.04][-1] < 0.2 * errors[0.04][+1]               # the sign is -1
    assert errors[0.08][-1] / errors[0.04][-1] > 5                 # error ~ eps^3


def test_matches_array_reference_transport():
    X1 = X0 + al.point([0.4, 0.3, -0.2, 0.5])
    omega = fl.random_connection_field(1, 0.4)
    path = tp.straight_line(X0, X1)
    T = tp.transport(omega, path, 60, scheme="magnus4")
    reference = atp.transport(array_connection(omega), array_path(path), 60, scheme="magnus4")
    assert np.abs(nz(T) - np.asarray(reference)).max() < 1e-12
