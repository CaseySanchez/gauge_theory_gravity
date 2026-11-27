"""Phase 0 of docs/PLAN_DIRAC.md: Dirac spinors as twistors, the map from Lasenby, Doran and Gull's spinors, the
algebraic action of the conformal sectors on a Dirac field, and the normalization of the first-order action.

Findings pinned here:
  * Gamma_a = I e_a p (p = (n_o + n_inf)/sqrt 2) is a Dirac algebra on twistors whose bar pairing is the twistor norm;
    the dilation generator is gamma_5 and the omega/pi halves are the Weyl spinors.
  * In the action, through 1/2 Gamma^a Omega(a), the translation part of the full twistor connection adds
    sqrt 2 sigma (sigma = bar(psi) psi), a parity-even mass made of the compensators; the special-conformal part adds
    (sqrt 2 / 4)(tr f sigma + f^ab T_ab), T the antisymmetric tensor bilinear (sqrt 2 sigma at f = e). The dilation
    part adds nothing (its real part vanishes identically).
  * S1 = -2 sqrt(g) R + 12 sqrt(g) tr(f e^-1), S2 = -24 k^2 sqrt(g) and S3 = -24 sqrt(g) for f = k e.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, dirac as dr, first_order as fo
from gauge_theory_gravity import twistor as tw, variation as va

RNG = np.random.default_rng(11)
I = dr.I
t, x, y, z = al.SPACETIME
ETA = dr.SQUARES
EVEN_STA = [1 + 0 * t * t, t * x, t * y, t * z, x * y, x * z, y * z, t * x * y * z]   # the even spacetime algebra


def nz(e):
    return np.asarray(e.kernel)


def number(e):
    return float(nz(e).reshape(-1)[0])


def random_twistor():
    return tw.from_coordinates(RNG.normal(size=4) + 1j * RNG.normal(size=4))


def random_sta_spinor():
    return sum((c * b for c, b in zip(RNG.normal(size=8), EVEN_STA)), 0 * t * t)


def h(first, second):
    return complex(tw.as_complex(dr.bar(first, second)))


def scalar_part(m):
    return number(m.restrict_grade(0)) if m.restrict_grade(0).kernel.size else 0.0


# --- the Dirac algebra on twistors ------------------------------------------------------------------------------
def test_gammas_form_a_dirac_algebra_and_rotate_like_gtg_spinors():
    for a in range(4):
        for b in range(4):
            anti = tw.matrix(dr.GAMMA[a] * dr.GAMMA[b] + dr.GAMMA[b] * dr.GAMMA[a])
            assert np.abs(anti - 2 * al.ETA4[a, b] * np.eye(4)).max() < 1e-14
            if a != b:      # the Lorentz generators: Gamma_a Gamma_b is the spacetime bivector e_a e_b
                assert np.abs(tw.matrix(dr.GAMMA[a] * dr.GAMMA[b]) - tw.matrix(al.SPACETIME[a] * al.SPACETIME[b])).max() < 1e-14


def test_dilation_is_chirality_and_the_twistor_halves_are_weyl_spinors():
    product = dr.GAMMA[0] * dr.GAMMA[1] * dr.GAMMA[2] * dr.GAMMA[3]
    assert np.abs(tw.matrix(dr.CHIRALITY) - tw.matrix(I * product)).max() < 1e-14
    assert np.abs(tw.matrix(dr.CHIRALITY) - np.diag([-1, -1, 1, 1])).max() < 1e-14         # omega: -1, pi: +1
    assert np.abs(tw.matrix(dr.LEFT) - tw.matrix(tw.OMEGA_PART)).max() < 1e-14
    assert np.abs(tw.matrix(dr.RIGHT) - tw.matrix(tw.PI_PART)).max() < 1e-14


def test_translations_and_special_conformal_generators_are_chiral_gammas():
    for a, e in enumerate(al.SPACETIME):
        translation = -np.sqrt(2) * I * dr.GAMMA[a] * dr.RIGHT
        special = -np.sqrt(2) * I * dr.GAMMA[a] * dr.LEFT
        assert np.abs(tw.matrix(e ^ al.n_inf) - tw.matrix(translation)).max() < 1e-14
        assert np.abs(tw.matrix(e ^ al.n_o) - tw.matrix(special)).max() < 1e-14


def test_twistor_norm_is_the_dirac_bar_pairing():
    """bar(psi) gamma^0 psi = psi^dagger psi > 0, the current is timelike and future pointing, and Fierz holds:
    J^2 = sigma^2 + omega^2 = -A^2, J . A = 0."""
    for _ in range(8):
        Z = random_twistor()
        J, A = nz(dr.current(Z)).ravel(), nz(dr.axial_current(Z)).ravel()
        sigma, omega = number(dr.scalar_density(Z)), number(dr.pseudoscalar_density(Z))
        assert J[0] > 0
        assert abs(J @ (ETA * J) - (sigma ** 2 + omega ** 2)) < 1e-11 * (J @ (ETA * J))
        assert abs(A @ (ETA * A) + J @ (ETA * J)) < 1e-11 * (J @ (ETA * J))
        assert abs(J @ (ETA * A)) < 1e-11 * (J @ (ETA * J))
        assert abs(h(Z, Z).imag) < 1e-13                                                     # Hermitian


# --- the map from Lasenby, Doran and Gull's spinors ---------------------------------------------------------------
def test_reference_twistor():
    R = dr.REFERENCE
    assert abs(h(R, R) - 1) < 1e-14
    assert np.abs(nz(dr.act(dr.GAMMA[0], R) - R)).max() < 1e-15
    assert np.abs(nz(dr.act(y * x, R) - dr.act(I, R))).max() < 1e-15


def test_map_is_a_real_isomorphism_carrying_the_dirac_operations():
    columns = [np.concatenate([tw.coordinates(dr.from_sta(b)).real, tw.coordinates(dr.from_sta(b)).imag])
               for b in EVEN_STA]
    assert np.linalg.matrix_rank(np.stack(columns, 1), 1e-10) == 8
    psi = random_sta_spinor()
    Z = dr.from_sta(psi)
    for mu, e in enumerate(al.SPACETIME):                                                    # gamma_mu psi gamma_0
        assert np.abs(nz(dr.from_sta(e * psi * t) - dr.act(dr.GAMMA[mu], Z))).max() < 1e-13
    assert np.abs(nz(dr.from_sta(psi * (y * x)) - dr.act(I, Z))).max() < 1e-13                  # psi I sigma_3


def test_map_carries_the_bilinears_and_the_kinetic_form():
    psi, phi = random_sta_spinor(), random_sta_spinor()
    Z, W = dr.from_sta(psi), dr.from_sta(phi)
    pseudoscalar = t * x * y * z
    assert abs(scalar_part(psi * ~psi) - number(dr.scalar_density(Z))) < 1e-12
    assert abs(scalar_part(psi * ~psi * pseudoscalar) + number(dr.pseudoscalar_density(Z))) < 1e-12
    J, A = nz(dr.current(Z)).ravel(), nz(dr.axial_current(Z)).ravel()
    for a, e in enumerate(al.SPACETIME):
        assert abs(scalar_part(psi * t * ~psi * e) - J[a]) < 1e-12                           # psi gamma_0 ~psi
        assert abs(scalar_part(psi * z * ~psi * e) + A[a]) < 1e-12                           # psi gamma_3 ~psi
        # <gamma^a phi I gamma_3 ~psi>_0 = Re h(Z_psi, I Gamma^a Z_phi): the kinetic term of their action (3.56)
        kinetic = scalar_part(e * ETA[a] * phi * pseudoscalar * z * ~psi)
        assert abs(kinetic - h(Z, dr.act(I * dr.GAMMA_UP[a], W)).real) < 1e-12


# --- the conformal sectors in the Dirac action ---------------------------------------------------------------------
def test_translation_and_special_conformal_sectors_add_a_mass_and_dilation_drops_out():
    """Re h(Z, I 1/2 Gamma^a G_a Z), summed over the frame, for G_a = P_a, K_a: each is sqrt 2 sigma. For the dilation
    generator the real part vanishes for every frame index."""
    for _ in range(4):
        Z = random_twistor()
        sigma = number(dr.scalar_density(Z))
        for null in (al.n_inf, al.n_o):
            term = sum(0.5 * h(Z, dr.act(I * dr.GAMMA_UP[a] * (e ^ null), Z)).real for a, e in enumerate(al.SPACETIME))
            assert abs(term - np.sqrt(2) * sigma) < 1e-12
        for a in range(4):
            assert abs(h(Z, dr.act(I * dr.GAMMA_UP[a] * dr.CHIRALITY, Z)).real) < 1e-13


def test_special_conformal_coupling_sees_the_trace_and_the_antisymmetric_part_of_f():
    """For a general special-conformal field f_ab (frame indices, lowered with eta) the term is
    (sqrt 2 / 4)(tr f sigma + f^ab T_ab) with T antisymmetric, the tensor bilinear bar(psi) sigma_ab (1 - gamma_5) psi:
    a symmetric trace-free f does not couple."""
    Z = random_twistor()
    sigma = number(dr.scalar_density(Z))
    coupling = np.array([[0.5 * h(Z, dr.act(I * dr.GAMMA_UP[a] * (e ^ al.n_o), Z)).real for e in al.SPACETIME]
                         for a in range(4)])                                                 # [a, b]: Gamma^a K_b
    lowered = ETA[:, None] * coupling
    trace_part = 0.5 * (lowered + lowered.T)
    assert np.abs(trace_part - np.sqrt(2) / 4 * sigma * al.ETA4).max() < 1e-12
    assert np.abs(0.5 * (lowered - lowered.T)).max() > 1e-2


# --- the normalization of the first-order action ---------------------------------------------------------------
@pytest.mark.parametrize("name,geometry", [("lapse", cartan.static_lapse()), ("friedmann", cartan.friedmann())])
def test_first_order_terms_against_the_volume_and_scalar_curvature(name, geometry):
    x = al.point([1.3, 0.4, -0.7, 0.9])
    volume, curvature = float(va.volume(geometry, x)), float(va.einstein_hilbert(geometry, x))
    for k in (0.0, 0.3):
        s1, s2, s3 = np.asarray(fo.densities(cf.connection(geometry, lambda y, k=k: k * geometry(y)), x))
        assert abs(s1 - (-2 * curvature + 48 * k * volume)) < 1e-10                         # tr(f e^-1) = 4k
        assert abs(s2 + 24 * k ** 2 * volume) < 1e-10
        assert abs(s3 + 24 * volume) < 1e-10
