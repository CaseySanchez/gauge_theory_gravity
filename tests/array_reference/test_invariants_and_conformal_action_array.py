"""Phase 7.4 (reduced) and Phase 8 item 4: quadratic action density, conformal action of versors."""
import numpy as np
import jax.numpy as jnp
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al
from gtg_csta._array import cartan, conformal as cf, invariants as iv, fields as fl

CASES = {
    "schw": (cartan.painleve_gullstrand(1.0), jnp.array([.3, 1.7, -.9, 1.1])),
    "frw": (cf.frw(), jnp.array([1.3, .4, -.7, .9])),
    "lapse": (cf.static_lapse(), jnp.array([1.3, .4, -.7, .9])),
}


@pytest.mark.parametrize("name", CASES)
def test_quadratic_density_of_normal_connection_is_half_weyl_squared(name):
    cof, x = CASES[name]
    fn = lambda xx: cf.schouten_frame(cof, xx)[0]
    I = float(iv.quadratic_invariant(cof, cf.connection(cof, fn))(x))
    W = float(iv.weyl_squared(cof, x))
    assert abs(I - 0.5 * W) < 1e-9


def test_plain_lorentz_sector_density_is_not_weyl_squared_when_ricci_nonzero():
    """Without the f sector, the Lorentz-only density sees Ricci too (for the lapse metric)."""
    cof, x = CASES["lapse"]
    I0 = float(iv.quadratic_invariant(cof, cf.connection(cof))(x))
    W = float(iv.weyl_squared(cof, x))
    assert abs(I0 - 0.5 * W) > 1e-3


# ---------- conformal embedding and generator roles ----------

def point(xv):
    """X = n_o + x - 1/2 x^2 n_inf  (null for n_o.n_inf = +1)."""
    xv = np.asarray(xv, float)
    x2 = float(xv @ al.ETA4 @ xv)
    vec = sum(float(xv[i]) * al.SPACETIME[i] for i in range(4))
    return al.n_o + vec - 0.5 * x2 * al.n_inf


def read_point(X):
    """Recover x from a conformal point by X = w (n_o + x - x^2/2 n_inf): x^a = X.e^a / (X.n_inf)."""
    w = al.scalar(X.dot(al.n_inf))
    out = []
    for i, e in enumerate(al.SPACETIME):
        out.append(al.scalar(X.dot(e)) / w * al.ETA4[i, i])
    return np.array(out), w


def test_embedded_points_are_null():
    X = point([0.3, -0.2, 0.5, 0.1])
    assert abs(al.scalar(X * X)) < 1e-14


def versor(gen_coords):
    return al.bivector(gen_coords).exp()


def act(V, X):
    return V * X * ~V


def test_P_generators_translate_and_fix_n_inf():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    for a in range(4):
        c = np.zeros(15); c[6 + a] = 0.7
        V = versor(c)
        Y = act(V, point(x0))
        # n_inf is fixed
        assert np.abs(np.asarray((act(V, al.n_inf) - al.n_inf).kernel)).max() < 1e-12
        shift, w = read_point(Y)
        assert abs(w - 1) < 1e-12                                     # no rescaling
        d = shift - x0
        assert np.abs(np.delete(d, a)).max() < 1e-12 and abs(d[a] - 2 * 0.7) < 1e-12   # shift = 2 c along a


def test_K_generators_fix_n_o():
    for a in range(4):
        c = np.zeros(15); c[10 + a] = 0.6
        V = versor(c)
        assert np.abs(np.asarray((act(V, al.n_o) - al.n_o).kernel)).max() < 1e-12


def test_D_generator_dilates():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    lam = 0.4
    c = np.zeros(15); c[14] = lam
    Y = act(versor(c), point(x0))
    shift, w = read_point(Y)
    s = shift
    ratio = s / x0
    assert np.allclose(ratio, ratio[0], atol=1e-12)                   # uniform scaling
    assert abs(ratio[0] - np.exp(-2 * lam)) < 1e-12                   # x -> exp(-2 lambda) x


def test_special_conformal_is_inversion_conjugated_translation_and_preserves_null_cone():
    x0 = np.array([0.3, -0.2, 0.5, 0.1])
    V = versor(np.eye(15)[10 + 1] * 0.2)
    Y = act(V, point(x0))
    assert abs(al.scalar(Y * Y)) < 1e-12
    shift, w = read_point(Y)
    assert np.abs(shift - x0).max() > 1e-3                            # non-trivial map
    # Lorentz generator: moves x but keeps x^2
    V = versor(np.eye(15)[1] * 0.3)
    shift, w = read_point(act(V, point(x0)))
    assert abs(shift @ al.ETA4 @ shift - x0 @ al.ETA4 @ x0) < 1e-12 and abs(w - 1) < 1e-12
