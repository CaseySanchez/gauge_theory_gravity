"""Plan for 11.4, phase 3B: the action quadratic in curvature, <F ^ F ^ X ^ I>, with the full connection varied.

With the origin X = n_o and the infinity twistor I = n_inf only the Lorentz sector of F survives (Kaku, Townsend and
van Nieuwenhuizen's eps_abcd F^ab F^cd). Unlike de Sitter MacDowell-Mansouri, [P, P] = 0 in the conformal algebra, so
F_M holds no e ^ e term and the action does not contain Einstein-Hilbert. Found here:

  * at the normal connection the density is -sqrt(g) C^2, minus the Weyl-squared action of section 10;
  * the special-conformal equation forces f = Schouten (it holds at the normal connection off shell);
  * the Lorentz equation is 4 eps_abcd e^c ^ F_K^d, with F_K = D P the Cotton 2-form: it requires a Cotton-free
    (harmonic Weyl) metric in the scale the compensators fix;
  * the dilation field drops out.
So the solutions are not Einstein's: conformally flat FRW solves it. And a Weyl-rescaled Schwarzschild, Bach-flat and
gauge-equivalent to a solution, does not, because the fixed compensators pick a scale.
"""
from itertools import permutations

import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, first_order as fo, variation as vr
from gauge_theory_gravity.curvature import curvature
from test_weyl_rescaling import rescaled

X_HOLE = [0.3, 1.7, -0.9, 1.1]
X_LAPSE = [1.3, 0.4, -0.7, 0.9]
SECTORS = {"M": slice(0, 6), "P": slice(6, 10), "K": slice(10, 14), "D": slice(14, 15)}
# The Levi-Civita symbol, for the independent index formula below.
EPSILON = np.zeros((4, 4, 4, 4))
for _p in permutations(range(4)):
    EPSILON[_p] = np.linalg.det(np.eye(4)[list(_p)])


def normal(geometry):
    return cf.connection(geometry, cf.schouten(geometry))


def _equations(omega, x4):
    return fo.euler_lagrange_connection(omega, x4, fo.macdowell_mansouri)


def _sizes(equations):
    return {name: float(np.abs(equations[part]).max()) for name, part in SECTORS.items()}


def cotton_equation(geometry, x4):
    """4 eps_abcd e^c ^ F_K^d as the [6, 4] Lorentz-generator components, F_K the special-conformal sector of the
    normal connection's curvature, read as frame vectors through its commutator with n_inf."""
    x = al.point(x4)
    special = al.sector(curvature(normal(geometry))(x), "K")
    cotton = np.zeros((4, 4, 4))                                          # [d, rho, sigma], frame index up
    for r in range(4):
        for s in range(4):
            v = special(al.DIRECTIONS[r], al.DIRECTIONS[s]).commutator(al.n_inf)
            cotton[:, r, s] = [al.scalar(v | al.SPACETIME[a]) * al.ETA4[a, a] for a in range(4)]
    eta, e = al.ETA4, al.components(geometry(x))
    lowered = np.einsum("dk,krs->drs", eta, cotton)
    return np.stack([4 * eta[a, a] * eta[b, b] * np.einsum("cd,mnrs,cn,drs->m", EPSILON[a, b], EPSILON,
                                                            eta @ e, lowered) for a, b in al.M_PAIRS])


@pytest.mark.parametrize("name,geometry,x4", [
    ("lapse", cartan.static_lapse(), X_LAPSE),
    ("schwarzschild", cartan.schwarzschild(1.0), X_HOLE),
    ("kerr", cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3]),
])
def test_density_at_the_normal_connection_is_minus_weyl_squared(name, geometry, x4):
    x = al.point(x4)
    density, weyl = float(fo.macdowell_mansouri(normal(geometry), x)), float(vr.weyl_squared(geometry, x))
    assert abs(density + weyl) < 1e-13 and abs(weyl) > 1e-2


@pytest.mark.parametrize("name,geometry,x4", [
    ("schwarzschild", cartan.schwarzschild(1.0), X_HOLE),
    ("frw", cartan.friedmann(), X_LAPSE),                                 # not Einstein: the theory is not GR
])
def test_normal_connection_solves(name, geometry, x4):
    assert np.abs(_equations(normal(geometry), x4)).max() < 1e-13


@pytest.mark.slow
@pytest.mark.parametrize("name,geometry,x4", [
    ("kerr", cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3]),
    ("de_sitter", cartan.de_sitter(0.5), [0.7, 0.4, -0.7, 0.9]),
])
def test_normal_connection_solves_slow(name, geometry, x4):
    assert np.abs(_equations(normal(geometry), x4)).max() < 1e-13


def test_off_shell_the_special_conformal_equation_holds_and_the_lorentz_one_is_cotton():
    """On the static lapse: f = Schouten satisfies its own equation, the dilation field drops out, and the Lorentz
    equation is exactly 4 eps e ^ Cotton."""
    geometry = cartan.static_lapse()
    equations = _equations(normal(geometry), X_LAPSE)
    sizes = _sizes(equations)
    assert sizes["K"] < 1e-14 and sizes["D"] == 0.0 and sizes["M"] > 0.1 and sizes["P"] > 0.1
    assert np.abs(equations[SECTORS["M"]] - cotton_equation(geometry, X_LAPSE)).max() < 1e-14


@pytest.mark.slow
def test_weyl_rescaled_schwarzschild_is_not_a_solution():
    """Bach-flat and gauge-equivalent to Schwarzschild's normal connection, but its Cotton tensor is not zero: the
    fixed compensators break the Weyl invariance."""
    geometry = rescaled(cartan.schwarzschild(1.0))
    equations = _equations(normal(geometry), X_HOLE)
    assert _sizes(equations)["M"] > 0.1 and _sizes(equations)["K"] < 1e-14
    assert np.abs(equations[SECTORS["M"]] - cotton_equation(geometry, X_HOLE)).max() < 1e-13


@pytest.mark.slow
def test_without_the_schouten_tensor_frw_is_not_a_solution():
    assert np.abs(_equations(cf.connection(cartan.friedmann()), X_LAPSE)).max() > 1.0
