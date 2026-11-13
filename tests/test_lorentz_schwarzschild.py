"""Phase 6 (Lorentz sector, spec 9.1; Schwarzschild) on extensors, with the array implementation as oracle."""
import numpy as np

import gtg_csta  # noqa: F401
from gtg_csta import algebra as al, cartan, fields as fl, gauge, curvature as cv, sta_embedding as se
from gtg_csta._array import cartan as ac

X0 = al.point([0.3, 1.7, -0.9, 1.1])
nz = lambda e: np.asarray(e.kernel)


def lorentz_field(seed):
    """A smooth connection with only spacetime-bivector (Lorentz) components."""
    polynomial = fl.polynomial(se.LorentzConnection, seed)

    def connection(x):
        return se.lorentz_embedding(polynomial(x))

    return connection


def test_lorentz_sector_curvature_stays_in_sector():
    F = cv.curvature(lorentz_field(1))(X0)
    assert max(se.sector_leakage(F).values()) < 1e-12
    assert al.sector_size(F, "M") > 0.01


def test_lorentz_rotor_gauge_closes_sector():
    bivectors = fl.polynomial(al.SpacetimeBivector, 2, 0.4)

    def rotor(x):
        return se.lorentz_rotor(bivectors(x))

    transformed = gauge.transform_connection(lorentz_field(3), rotor)(X0)
    assert max(se.sector_leakage(transformed).values()) < 1e-11


def test_general_conformal_versor_leaks_out_of_lorentz_sector():
    """Spec 9.1 caveat: the Lorentz restriction is not preserved by general Spin(2,4)."""
    transformed = gauge.transform_connection(lorentz_field(5), fl.random_versor_field(4, 0.3))(X0)
    assert max(se.sector_leakage(transformed).values()) > 0.01


def test_numga_sta_sign_map_is_lie_homomorphism():
    ours = al.structure_constants()[:6, :6, :6]
    assert np.abs(se.numga_sta_structure_constants() + ours).max() < 1e-12


# ---------------- Schwarzschild in Painleve-Gullstrand coordinates ----------------

M = 1.0
COF = cartan.schwarzschild(M)
ACOF = ac.painleve_gullstrand(M)


def test_extensor_spin_connection_matches_index_implementation():
    Om = cartan.lorentz_connection(COF)(X0)
    reference = ac.lorentz_connection_blades(ACOF)(X0.kernel)                    # (4, 15)
    assert np.abs(nz(Om).T - np.asarray(reference)).max() < 1e-12


def test_metric_christoffel_match_index_implementation():
    g = cartan.metric(COF)(X0)
    assert np.abs(nz(g)[0] - np.asarray(ac.metric(ACOF)(X0.kernel))).max() < 1e-12
    lower = nz(cartan.christoffel_lower(COF)(X0))[0]
    symbols = np.asarray(ac.christoffel(ACOF)(X0.kernel))                        # Gamma^l_{mn}
    assert np.abs(lower - np.einsum("sl,lmn->smn", nz(g)[0], symbols)).max() < 1e-12


def test_riemann_from_connection_matches_christoffel_riemann():
    """F(a,b) of the Lorentz connection equals R^{ab}_{mu nu} computed independently from Christoffel symbols."""
    F = cv.curvature(cartan.lorentz_connection(COF))(X0)
    ours = al.to_generators(np.moveaxis(nz(F), 0, -1))[..., :6]                  # [mu, nu, (ab)]
    riemann = np.asarray(ac.riemann_frame_from_coordinate(ACOF)(X0.kernel))      # R^a_{b mu nu}
    raised = np.einsum("abmn,bc->acmn", riemann, al.ETA4)
    expected = np.stack([raised[a, b] for a, b in al.M_PAIRS], axis=-1)
    assert np.abs(ours - expected).max() < 1e-9


def test_lorentz_connection_is_covariant_derivative_of_vectors():
    """1/2 [Omega(d_mu), v] reproduces the frame components of the Levi-Civita derivative."""
    omega = np.asarray(ac.spin_connection(ACOF)(X0.kernel))                      # omega_mu^a_b
    connection = cartan.lorentz_connection(COF)(X0)
    v = np.array([0.3, -0.5, 0.8, 0.1])
    vector = al.point(v)
    rows = []
    for direction in al.SPACETIME:
        bivector = connection(direction)
        rows.append(nz((al.mv.vector(np.zeros(6)) + (bivector.commutator(vector)).restrict[1])))
    assert np.abs(np.stack(rows)[:, :4] - np.einsum("mab,b->ma", omega, v)).max() < 1e-9


def test_schwarzschild_vacuum_and_kretschmann():
    from gtg_csta import invariants as iv
    assert np.abs(np.asarray(ac.ricci_coordinate(ACOF)(X0.kernel))).max() < 1e-9
    density = al.scalar(iv.quadratic_invariant(COF, cartan.lorentz_connection(COF))(X0))
    r = np.linalg.norm(nz(X0)[1:])
    assert abs(2 * density - 48 * M ** 2 / r ** 6) < 1e-8            # Kretschmann = 2 <F F~>
