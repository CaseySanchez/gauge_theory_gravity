"""Phase 6 (Lorentz sector, spec 9.1; Schwarzschild) on extensors, with the array implementation as oracle."""
import numpy as np
import jax.numpy as jnp

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, fields as fl, gauge, curvature as cv, sta_embedding as se
from gtg_csta._array import cartan as ac

X0 = al.point([0.3, 1.7, -0.9, 1.1])
nz = lambda e: np.asarray(e.kernel)


def lorentz_field(seed):
    rng = np.random.default_rng(seed)
    A0, A1, A2 = rng.normal(size=(4, 6)) * .3, rng.normal(size=(4, 6, 4)) * .3, rng.normal(size=(4, 6, 4)) * .3
    return lambda x: se.lorentz_connection(A0 + A1 @ x.kernel + A2 @ jnp.cos(.9 * x.kernel))


def test_lorentz_sector_curvature_stays_in_sector():
    F = cv.curvature(lorentz_field(1))(X0)
    assert max(se.sector_leakage(F).values()) < 1e-12
    assert al.sector_size(F, "M") > 0.01


def test_lorentz_rotor_gauge_closes_sector():
    rng = np.random.default_rng(2)
    B0, B1 = rng.normal(size=6) * .4, rng.normal(size=(6, 4)) * .4
    R = lambda x: se.lorentz_rotor(B0 + B1 @ x.kernel)
    Om2 = gauge.transform_connection(lorentz_field(3), R)(X0)
    assert max(se.sector_leakage(Om2).values()) < 1e-11


def test_general_conformal_versor_leaks_out_of_lorentz_sector():
    """Spec 9.1 caveat: the Lorentz restriction is not preserved by general Spin(2,4)."""
    Om2 = gauge.transform_connection(lorentz_field(5), fl.random_versor_field(4, 0.3))(X0)
    assert max(se.sector_leakage(Om2).values()) > 0.01


def test_numga_sta_sign_map_is_lie_homomorphism():
    f_ours = al.structure_constants()[:6, :6, :6]
    assert np.abs(se.numga_sta_structure_constants() + f_ours).max() < 1e-12


# ---------------- Schwarzschild in Painleve-Gullstrand coordinates ----------------

M = 1.0
COF = cartan.painleve_gullstrand(M)
ACOF = ac.painleve_gullstrand(M)


def test_extensor_spin_connection_matches_index_implementation():
    Om = cartan.lorentz_connection(COF)(X0)
    ref = ac.lorentz_connection_blades(ACOF)(X0.kernel)                    # (4, 15)
    assert np.abs(nz(Om).T - np.asarray(ref)).max() < 1e-12


def test_metric_christoffel_match_index_implementation():
    g = cartan.metric(COF)(X0)
    assert np.abs(nz(g)[0] - np.asarray(ac.metric(ACOF)(X0.kernel))).max() < 1e-12
    Gl = nz(cartan.christoffel_lower(COF)(X0))[0]
    Gam = np.asarray(ac.christoffel(ACOF)(X0.kernel))                      # Gamma^l_{mn}
    assert np.abs(Gl - np.einsum("sl,lmn->smn", nz(g)[0], Gam)).max() < 1e-12


def test_riemann_from_connection_matches_christoffel_riemann():
    """F(a,b) of the Lorentz connection equals R^{ab}_{mu nu} computed independently from Christoffel symbols."""
    F = cv.curvature(cartan.lorentz_connection(COF))(X0)
    Fg = np.asarray(al.to_generators(np.moveaxis(nz(F), 0, -1)))[..., :6]   # [mu, nu, (ab)]
    Rf = np.asarray(ac.riemann_frame_from_coordinate(ACOF)(X0.kernel))      # R^a_{b mu nu}
    Rab = np.einsum("abmn,bc->acmn", Rf, al.ETA4)
    expect = np.stack([Rab[a, b] for a, b in al.M_PAIRS], axis=-1)
    assert np.abs(Fg - expect).max() < 1e-9


def test_lorentz_connection_is_covariant_derivative_of_vectors():
    """1/2 [Omega(d_mu), v] reproduces the frame components of the Levi-Civita derivative."""
    om = np.asarray(ac.spin_connection(ACOF)(X0.kernel))                   # omega_mu^a_b
    Om = cartan.lorentz_connection(COF)(X0)
    v = np.array([0.3, -0.5, 0.8, 0.1])
    vv = al.point(v)
    cols = [Om(al.point(np.eye(4)[m])) for m in range(4)]
    got = np.stack([nz(0.5 * (c * vv - vv * c).restrict[1]) for c in cols])    # (4 mu, vector comps)
    expect = np.einsum("mab,b->ma", om, v)
    assert np.abs(got[:, :4] - expect).max() < 1e-9


def test_schwarzschild_vacuum_and_kretschmann():
    Ric = np.asarray(ac.ricci_coordinate(ACOF)(X0.kernel))
    assert np.abs(Ric).max() < 1e-9
    from gtg_csta import invariants as iv
    I = float(iv.quadratic_invariant(COF, cartan.lorentz_connection(COF))(X0).kernel.reshape(-1)[0])
    r = np.linalg.norm(np.asarray(X0.kernel)[1:])
    assert abs(2 * I - 48 * M ** 2 / r ** 6) < 1e-8            # Kretschmann = 2 <F F~>
