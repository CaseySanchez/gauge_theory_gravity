"""Phase 7.4 (reduced) and Phase 8 item 4 on extensors: quadratic action density, conformal action of versors."""
import numpy as np
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, conformal as cf, invariants as iv
from gtg_csta._array import cartan as ac, conformal as acf, invariants as aiv

CASES = {
    "schw": (cartan.schwarzschild(1.0), ac.painleve_gullstrand(1.0), al.point([.3, 1.7, -.9, 1.1])),
    "frw": (cartan.friedmann(), acf.frw(), al.point([1.3, .4, -.7, .9])),
    "lapse": (cartan.static_lapse(), acf.static_lapse(), al.point([1.3, .4, -.7, .9])),
}


@pytest.mark.parametrize("name", CASES)
def test_quadratic_density_of_normal_connection_is_half_weyl_squared(name):
    coframe, reference, x = CASES[name]
    density = al.scalar(iv.quadratic_invariant(coframe, cf.connection(coframe, cf.schouten(coframe)))(x))
    assert abs(density - 0.5 * al.scalar(iv.weyl_squared(coframe, x))) < 1e-9
    assert abs(density - 0.5 * float(aiv.weyl_squared(reference, x.kernel))) < 1e-9       # independent index formula


def test_plain_lorentz_sector_density_is_not_weyl_squared_when_ricci_nonzero():
    coframe, _, x = CASES["lapse"]
    plain = al.scalar(iv.quadratic_invariant(coframe, cf.connection(coframe))(x))
    assert abs(plain - 0.5 * al.scalar(iv.weyl_squared(coframe, x))) > 1e-3


# ---------- conformal embedding and generator roles ----------

def point(x):
    """X = n_o + x - 1/2 x^2 n_inf  (null for n_o.n_inf = +1)."""
    x = np.asarray(x, float)
    return al.n_o + sum(float(x[i]) * al.SPACETIME[i] for i in range(4)) - 0.5 * float(x @ al.ETA4 @ x) * al.n_inf


def read_point(X):
    """x^a = X.e_a eta_aa / (X.n_inf), and the weight X.n_inf."""
    weight = al.scalar(X.dot(al.n_inf))
    return np.array([al.scalar(X.dot(e)) / weight * al.ETA4[i, i] for i, e in enumerate(al.SPACETIME)]), weight


def versor(generator_coordinates):
    return al.bivector(generator_coordinates).exp()


def act(V, X):
    return V * X * ~V


def same(a, b):
    return np.abs(np.asarray((a - b).kernel)).max() < 1e-12


def test_embedded_points_are_null():
    assert abs(al.scalar(point([0.3, -0.2, 0.5, 0.1]) ** 2)) < 1e-14


def test_P_generators_translate_and_fix_n_inf():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    for a in range(4):
        c = np.zeros(15); c[6 + a] = 0.7
        V = versor(c)
        assert same(act(V, al.n_inf), al.n_inf)
        shifted, weight = read_point(act(V, point(x0)))
        d = shifted - x0
        assert abs(weight - 1) < 1e-12 and np.abs(np.delete(d, a)).max() < 1e-12 and abs(d[a] - 1.4) < 1e-12   # 2 c


def test_K_generators_fix_n_o():
    for a in range(4):
        c = np.zeros(15); c[10 + a] = 0.6
        assert same(act(versor(c), al.n_o), al.n_o)


def test_D_generator_dilates():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    c = np.zeros(15); c[14] = 0.4
    scaled, _ = read_point(act(versor(c), point(x0)))
    ratio = scaled / x0
    assert np.allclose(ratio, ratio[0], atol=1e-12) and abs(ratio[0] - np.exp(-0.8)) < 1e-12    # exp(-2 lambda)


def test_special_conformal_preserves_null_cone_and_lorentz_preserves_interval():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    image = act(versor(np.eye(15)[11] * 0.2), point(x0))
    assert abs(al.scalar(image * image)) < 1e-12
    assert np.abs(read_point(image)[0] - x0).max() > 1e-3
    moved, weight = read_point(act(versor(np.eye(15)[1] * 0.3), point(x0)))
    assert abs(moved @ al.ETA4 @ moved - x0 @ al.ETA4 @ x0) < 1e-12 and abs(weight - 1) < 1e-12
