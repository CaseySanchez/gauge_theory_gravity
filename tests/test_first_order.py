"""Plan for 11.4, first-order form: the full CSTA connection varied on its own.

`first_order` breaks Spin(2,4) with the origin X = n_o and the infinity twistor I = n_inf. The coframe is DX and the
special-conformal field is DI. The action S1 + beta S2 + gamma S3 (the Einstein-Cartan term, the term quadratic in
the special-conformal field, the cosmological term) is varied in all 60 components of Omega. Its solutions, found
here first by fitting and then checked:

  * the special-conformal field is algebraic, f = e / beta;
  * the Lorentz sector is torsion-free, and the metric of DX is Einstein with Lambda = 6 (gamma - 1/beta),
    here Lambda = 3 H^2 for de Sitter at Hubble rate H;
  * the dilation field b drops out of the action (it may even have curl);
  * when gamma = 0 (no explicit cosmological term) f is the Schouten tensor, -(Lambda/6) e in this sign of R, so the
    solution is exactly the normal conformal connection, with Lambda = -6 / beta coming from the f^2 coupling alone.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, first_order as fo

H = 0.5
X_HOLE = [0.3, 1.7, -0.9, 1.1]
X_DS = [0.7, 0.4, -0.7, 0.9]
X_LAPSE = [1.3, 0.4, -0.7, 0.9]


def _equations(omega, x4, beta, gamma):
    terms = fo.euler_lagrange_connection(omega, x4)
    return terms[0] + beta * terms[1] + gamma * terms[2]


def _with_special(geometry, k, dilation=None):
    """(Levi-Civita, e, f = k e, b): the trial connection."""
    return cf.connection(geometry, lambda x: k * geometry(x), dilation)


def _size(e):
    return float(np.abs(np.asarray(e.kernel)).max())


def test_compensators_read_coframe_and_special_conformal_field():
    geometry, x = cartan.static_lapse(), al.point(X_LAPSE)
    schouten = cf.schouten(geometry)
    dx, di = fo.covariant_points(cf.connection(geometry, schouten)(x))
    for m in range(4):
        assert _size(dx(al.DIRECTIONS[m]) - geometry(x)(al.DIRECTIONS[m]).dual_inverse()) < 1e-15
        assert _size(di(al.DIRECTIONS[m]) - schouten(x)(al.DIRECTIONS[m]).dual_inverse()) < 1e-15


@pytest.mark.parametrize("name,geometry,x4", [
    ("schwarzschild", cartan.schwarzschild(1.0), X_HOLE),
    ("kerr", cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3]),
])
def test_ricci_flat_solutions(name, geometry, x4):
    """Lambda = 0: gamma = 1/beta, f = e/beta (here beta = 4)."""
    equations = _equations(_with_special(geometry, 0.25), x4, beta=4.0, gamma=0.25)
    assert np.abs(equations).max() < 1e-13


def test_de_sitter_solves_with_the_normal_connection():
    """gamma = 0: the solution is the normal conformal connection, f = Schouten = -(Lambda/6) e, beta = -6/Lambda."""
    geometry, x = cartan.de_sitter(H), al.point(X_DS)
    k = -H ** 2 / 2                                                          # -Lambda / 6
    assert _size(cf.schouten(geometry)(x) - k * geometry(x)) < 1e-15
    equations = _equations(cf.connection(geometry, cf.schouten(geometry)), X_DS, beta=1 / k, gamma=0.0)
    assert np.abs(equations).max() < 1e-13


def test_de_sitter_off_the_normal_connection():
    """Lambda = 6 (gamma - 1/beta) holds for other couplings too, with f = e / beta not the Schouten tensor."""
    beta = 2.0
    equations = _equations(_with_special(cartan.de_sitter(H), 1 / beta), X_DS, beta=beta, gamma=1 / beta + H ** 2 / 2)
    assert np.abs(equations).max() < 1e-13


def test_wrong_cosmological_constant_is_not_a_solution():
    equations = _equations(_with_special(cartan.schwarzschild(1.0), 0.25), X_HOLE, beta=4.0, gamma=0.3)
    assert np.abs(equations).max() > 0.1


def test_dilation_field_drops_out_even_with_curl():
    def dilation(x):
        return (0.2 * cartan.coordinate(x, 1) * (al.SLICES[0] & al.Direction)
                + 0.1 * cartan.coordinate(x, 0) * (al.SLICES[2] & al.Direction))    # d b != 0

    equations = _equations(_with_special(cartan.schwarzschild(1.0), 0.25, dilation), X_HOLE, beta=4.0, gamma=0.25)
    assert np.abs(equations).max() < 1e-13


def test_torsion_is_not_a_solution():
    geometry = cartan.schwarzschild(1.0)
    base = _with_special(geometry, 0.25)
    twist = 0.1 * al.GENS[3] * (al.SLICES[1] & al.Direction)                       # a Lorentz piece, constant

    def contorted(x):
        return base(x) + twist

    assert np.abs(_equations(contorted, X_HOLE, beta=4.0, gamma=0.25)).max() > 1e-2


def test_static_lapse_has_no_solution():
    """f is algebraic, so the trial family (Levi-Civita, e, k e) is the only candidate: for no k, beta, gamma do the
    equations vanish on a metric that is not Einstein. The terms are polynomials in k (S1 linear, S2 quadratic without
    constant, S3 constant), fitted from k = 0, 1, 2, then scanned."""
    geometry = cartan.static_lapse()
    terms = [fo.euler_lagrange_connection(_with_special(geometry, k), X_LAPSE) for k in (0.0, 1.0, 2.0)]
    linear0, linear1 = terms[0][0], terms[1][0] - terms[0][0]
    square = (terms[2][1] - 2 * terms[1][1]) / 2
    single = terms[1][1] - square
    constant = terms[0][2]
    smallest = np.inf
    for k in np.concatenate([np.linspace(-4, 4, 801), [0.25, 1 / 8]]):
        columns = np.stack([(k * single + k * k * square).ravel(), constant.ravel()], axis=1)
        target = -(linear0 + k * linear1).ravel()
        couplings, *_ = np.linalg.lstsq(columns, target, rcond=None)
        smallest = min(smallest, float(np.abs(columns @ couplings - target).max()))
    assert smallest > 1e-2
