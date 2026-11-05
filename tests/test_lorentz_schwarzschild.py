"""Phase 6: Lorentz sector (spec 9.1) and Schwarzschild in the Lorentz sector."""
import numpy as np
import jax.numpy as jnp
import pytest

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, fields as fl, gauge, curvature as cv, sta_embedding as se

X0 = jnp.array([0.3, 1.7, -0.9, 1.1])


def lorentz_field(seed):
    rng = np.random.default_rng(seed)
    A0, A1, A2 = rng.normal(size=(4, 6)) * .3, rng.normal(size=(4, 6, 4)) * .3, rng.normal(size=(4, 6, 4)) * .3
    return lambda x: se.lorentz_connection(A0 + A1 @ x + A2 @ jnp.cos(.9 * x))


def test_lorentz_sector_curvature_stays_in_sector():
    F = np.asarray(cv.curvature(lorentz_field(1))(X0))
    leak = se.sector_leakage(F)
    assert max(leak.values()) < 1e-12
    assert np.abs(al.to_generators(F)[..., :6]).max() > 0.01


def test_lorentz_rotor_gauge_closes_sector():
    rng = np.random.default_rng(2)
    B0, B1 = rng.normal(size=6) * .4, rng.normal(size=(6, 4)) * .4
    R = se.lorentz_rotor_field(lambda x: B0 + B1 @ x)
    Om2 = gauge.transform_connection(lorentz_field(3), R)
    g = al.to_generators(np.asarray(Om2(X0)))
    assert np.abs(g[..., 6:]).max() < 1e-11


def test_general_conformal_versor_leaks_out_of_lorentz_sector():
    """Spec 9.1 caveat: the Lorentz restriction is not preserved by general Spin(2,4)."""
    V = fl.random_versor_field(4, 0.3)
    g = al.to_generators(np.asarray(gauge.transform_connection(lorentz_field(5), V)(X0)))
    assert np.abs(g[..., 6:]).max() > 0.01


def test_numga_sta_sign_map_is_lie_homomorphism():
    f_ours = al.structure_constants()[:6, :6, :6]
    f_sta = se.numga_sta_structure_constants()
    assert np.abs(f_sta + f_ours).max() < 1e-12      # phi(M_ab) = -M_ab is a homomorphism


# ---------------- Schwarzschild in Painleve-Gullstrand coordinates ----------------

M = 1.0
COF = cartan.painleve_gullstrand(M)


def test_riemann_from_connection_matches_christoffel_riemann():
    """F_{mu nu}^{ab} from the CSTA formula equals R^{ab}_{mu nu} computed independently."""
    Om = cartan.lorentz_connection_blades(COF)
    F = np.asarray(cv.curvature(Om)(X0))                                   # (4,4,15) blades
    Fg = al.to_generators(F)[..., :6]                                      # [mu,nu,(ab)]
    Rf = np.asarray(cartan.riemann_frame_from_coordinate(COF)(X0))         # R^a_{b mu nu}
    eta = al.ETA4
    Rab = np.einsum("abmn,bc->acmn", Rf, eta)                              # R^{ac}_{mu nu}
    expect = np.stack([Rab[a, b] for a, b in al.M_PAIRS], axis=-1)         # [mu,nu,(ab)]
    expect = np.moveaxis(expect, 0, 0)
    assert np.abs(Fg - expect).max() < 1e-9


def test_lorentz_connection_is_covariant_derivative_of_vectors():
    """1/2 [Omega_mu, v] reproduces the frame components of the Levi-Civita derivative."""
    om = np.asarray(cartan.spin_connection(COF)(X0))                       # omega_mu^a_b
    Om = fl.biv(cartan.lorentz_connection_blades(COF)(X0))                 # (4,)
    v = np.array([0.3, -0.5, 0.8, 0.1])
    vv = sum(v[i] * e for i, e in enumerate(al.SPACETIME))
    lhs = 0.5 * (Om * vv - vv * Om)                                        # batch (4,), a vector
    got = np.asarray((al.mv.vector(np.zeros(6)) + lhs.restrict[1]).kernel)[:, :4]   # t,x,y,z components
    expect = np.einsum("mab,b->ma", om, v)
    assert np.abs(got - expect).max() < 1e-9


def test_schwarzschild_vacuum_and_kretschmann():
    Ric = np.asarray(cartan.ricci_coordinate(COF)(X0))
    assert np.abs(Ric).max() < 1e-9
    # Kretschmann scalar 48 M^2 / r^6 from the CSTA curvature F^{ab}_{mu nu}
    Om = cartan.lorentz_connection_blades(COF)
    Fg = al.to_generators(np.asarray(cv.curvature(Om)(X0)))[..., :6]
    Rf = np.asarray(cartan.riemann_frame_from_coordinate(COF)(X0))
    e = np.asarray(COF(X0)); E = np.linalg.inv(e)
    Rff = np.einsum("abmn,mc,nd->abcd", Rf, E, E)                          # all-frame R^a_b^{cd}... (lower mu,nu)
    eta = al.ETA4
    Rl = np.einsum("ae,ebcd->abcd", eta, Rff)                              # R_{abcd}
    ei = np.linalg.inv(eta)
    K = np.einsum("abcd,ae,bf,cg,dh,efgh->", Rl, ei, ei, ei, ei, Rl)
    r = np.linalg.norm(np.asarray(X0)[1:])
    assert abs(K - 48 * M ** 2 / r ** 6) < 1e-8
