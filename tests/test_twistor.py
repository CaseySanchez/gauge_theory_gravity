"""Twistors as the minimal left ideal of the even subalgebra: complex structure, norm, incidence, light rays,
and local twistor transport by the normal conformal connection."""
import numpy as np
import pytest
import scipy.linalg

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, curvature as cv, fields, invariants as iv
from gauge_theory_gravity import transport as tr, twistor as tw

nz = lambda e: np.asarray(e.kernel)
RNG = np.random.default_rng(7)
EVENT = al.point([0.3, 1.1, -0.4, 0.7])
X = al.point([1.3, 0.4, -0.7, 0.9])
# The twistor norm in the coordinates (omega, pi): h = 2 Re(conj(omega) . pi).
HERMITIAN = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])


def random_coordinates(*shape):
    return RNG.normal(size=(*shape, 4)) + 1j * RNG.normal(size=(*shape, 4))


def test_pseudoscalar_is_a_central_imaginary_unit_of_the_even_subalgebra():
    assert np.allclose(nz(tw.PSEUDOSCALAR * tw.PSEUDOSCALAR), -1)
    psi = al.mv.even(RNG.normal(size=32))
    assert np.abs(nz(tw.PSEUDOSCALAR * psi - psi * tw.PSEUDOSCALAR)).max() < 1e-14


def test_idempotent_is_primitive_and_self_reverse():
    P = tw.IDEMPOTENT
    assert np.abs(nz(P * P - P)).max() < 1e-15 and np.abs(nz(~P - P)).max() < 1e-15
    # Even P is spanned over the reals by the four basis twistors and I times them: eight real dimensions.
    columns = [nz(tw.twistor(al.mv.even(row))) for row in np.eye(32)]
    assert np.linalg.matrix_rank(np.stack(columns), 1e-10) == 8
    basis = [nz(tw.BASIS[k]) for k in range(4)] + [nz((tw.PSEUDOSCALAR * tw.BASIS[k]).cast(al.Even)) for k in range(4)]
    assert np.linalg.matrix_rank(np.stack(basis + columns), 1e-10) == 8


def test_coordinates_round_trip_and_spinor_columns_lie_in_the_ideal():
    values = random_coordinates(3)
    assert np.abs(tw.coordinates(tw.from_coordinates(values)) - values).max() < 1e-13
    Z = tw.twistor(al.mv.even(RNG.normal(size=32)))
    assert np.abs(nz(tw.from_coordinates(tw.coordinates(Z)) - Z)).max() < 1e-13


def test_norm_is_hermitian_with_signature_two_two():
    first, second = tw.from_coordinates(random_coordinates()), tw.from_coordinates(random_coordinates())
    forward = tw.as_complex(tw.inner(first, second))
    assert abs(forward - np.conj(tw.as_complex(tw.inner(second, first)))) < 1e-13
    gram = np.array([[tw.as_complex(tw.inner(tw.BASIS[j], tw.BASIS[k])) for k in range(4)] for j in range(4)])
    assert np.abs(gram - HERMITIAN).max() < 1e-14
    assert sorted(np.sign(np.linalg.eigvalsh(gram))) == [-1, -1, 1, 1]
    values = random_coordinates()
    assert abs(tw.norm(tw.from_coordinates(values)) - 2 * np.real(np.conj(values[:2]) @ values[2:])) < 1e-13


def test_spin_2_4_acts_as_su_2_2():
    versor = fields.random_versor_field(seed=2)(EVENT)
    M = tw.matrix(versor)
    assert np.abs(M.conj().T @ HERMITIAN @ M - HERMITIAN).max() < 1e-11
    assert abs(np.linalg.det(M) - 1) < 1e-11
    generator = tw.matrix(fields.polynomial(al.Bivector, seed=3)(EVENT))
    assert abs(np.trace(generator)) < 1e-13
    assert np.abs(generator.conj().T @ HERMITIAN + HERMITIAN @ generator).max() < 1e-13


def test_omega_and_pi_are_the_null_kernels_of_infinity_and_the_origin():
    Z = tw.from_coordinates(random_coordinates())
    assert np.abs(nz(tw.omega(Z) + tw.pi(Z) - Z)).max() < 1e-14
    assert np.abs(nz(al.n_inf * tw.omega(Z))).max() < 1e-14 and np.abs(nz(al.n_o * tw.pi(Z))).max() < 1e-14
    assert abs(tw.norm(tw.omega(Z))) < 1e-14 and abs(tw.norm(tw.pi(Z))) < 1e-14


def test_conformal_point_is_the_translated_origin():
    vector = tw.as_vector(EVENT)
    translator = 1 + 0.5 * vector * al.n_inf
    X_ = tw.conformal_point(EVENT)
    assert np.abs(nz(translator * al.n_o * ~translator - X_)).max() < 1e-14
    assert abs(al.scalar(X_ | X_)) < 1e-14


def test_incidence_alpha_planes_and_the_klein_correspondence():
    """X Z = 0 cuts out a complex 2-plane, swept by `through`; omega = 1/2 x n_inf pi is a 2x2 matrix in x whose
    determinant is proportional to x^2, so two alpha-planes meet exactly when their events are null separated."""
    momentum = tw.from_coordinates(random_coordinates())
    assert np.abs(nz(tw.incidence(EVENT, tw.through(EVENT, momentum)))).max() < 1e-14
    products = np.stack([nz(tw.incidence(EVENT, tw.BASIS[k])) for k in range(4)]
                        + [nz((tw.incidence(EVENT, tw.PSEUDOSCALAR * tw.BASIS[k]))) for k in range(4)], axis=1)
    assert np.linalg.matrix_rank(products, 1e-10) == 4                       # real kernel 4: a complex 2-plane

    def block(event):
        return tw.matrix(0.5 * tw.as_vector(event) * al.n_inf)[:2, 2:]      # pi -> omega
    coordinates = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.4, -0.3, 1.2, 0.5], [1.0, 0.6, 0.0, 0.8]])
    for c in coordinates:
        square = c[0] ** 2 - c[1:] @ c[1:]
        hermitian = block(al.point(c)) * np.sqrt(2.0) * 1j                     # omega = -(i / sqrt 2) x^{AA'} pi
        assert np.abs(hermitian - hermitian.conj().T).max() < 1e-13, c       # real events: Hermitian 2x2 matrices
        assert abs(np.linalg.det(hermitian) - square) < 1e-13, c             # det x^{AA'} = x^2

    def plane(event):
        return np.stack([nz(tw.through(event, tw.PI_BASIS[k])) for k in range(2)]
                        + [nz(tw.through(event, (tw.PSEUDOSCALAR * tw.PI_BASIS[k]).cast(al.Even))) for k in range(2)], 1)
    null = al.coordinates(EVENT) + np.array([1.0, 0.6, 0.0, 0.8])
    spacelike = al.coordinates(EVENT) + np.array([0.2, 0.6, 0.0, 0.8])
    assert np.linalg.matrix_rank(np.hstack([plane(EVENT), plane(al.point(null))]), 1e-10) == 6    # meet in C^1
    assert np.linalg.matrix_rank(np.hstack([plane(EVENT), plane(al.point(spacelike))]), 1e-10) == 8


def test_null_twistor_is_a_light_ray_and_others_annihilate_nothing():
    Z = tw.through(EVENT, tw.from_coordinates(random_coordinates()))
    assert abs(tw.norm(Z)) < 1e-14
    assert tw.annihilator(Z).shape == (2, 6)
    generic = tw.from_coordinates(random_coordinates())
    assert abs(tw.norm(generic)) > 1e-3 and tw.annihilator(generic).shape == (0, 6)
    start, direction = tw.ray(Z)
    assert abs(al.scalar(direction.squared())) < 1e-13 and np.abs(al.coordinates(direction)).max() > 1e-3
    for along in (0.0, 1.7, -2.3):
        assert np.abs(nz(tw.incidence(start + along * direction, Z))).max() < 1e-13
    offset = al.coordinates(EVENT) - al.coordinates(start)                  # the defining event is on the ray
    assert np.linalg.matrix_rank(np.stack([offset, al.coordinates(direction)]), 1e-10) == 1


@pytest.mark.parametrize("name, flat", [("frw", True), ("lapse", False)])
def test_local_twistor_curvature_vanishes_exactly_when_conformally_flat(name, flat):
    geometry = {"frw": cartan.friedmann(), "lapse": cartan.static_lapse()}[name]
    strength = cv.curvature(cf.connection(geometry, cf.schouten(geometry)))(X)
    M = tw.matrix(strength(al.DIRECTIONS[0], al.DIRECTIONS[1]))
    assert abs(np.trace(M)) < 1e-12
    assert (np.abs(M).max() < 1e-9) if flat else (np.abs(M).max() > 1e-2)


def test_local_twistor_holonomy_is_su_2_2_and_matches_the_curvature():
    geometry = cartan.static_lapse()
    omega = cf.connection(geometry, cf.schouten(geometry))
    size = 0.05
    holonomy = tw.matrix(tr.holonomy_loop(omega, X, 0, 1, size, steps_per_side=8, scheme="magnus4"))
    assert np.abs(holonomy.conj().T @ HERMITIAN @ holonomy - HERMITIAN).max() < 1e-11
    strength = tw.matrix(cv.curvature(omega)(X)(al.DIRECTIONS[0], al.DIRECTIONS[1]))
    expected = scipy.linalg.expm(-0.5 * size ** 2 * strength)
    assert np.abs(holonomy - expected).max() < 20 * size ** 3 * np.abs(strength).max()


def test_trace_is_the_complex_trace_on_twistor_space():
    operator = al.mv.even(RNG.normal(size=32))
    assert abs(tw.as_complex(tw.trace(operator)) - np.trace(tw.matrix(operator))) < 1e-12
    assert abs(tw.as_complex(tw.trace(tw.OMEGA_PART)) - 2) < 1e-14                 # the omega half is C^2


def test_momentum_of_a_twistor_is_the_flag_pole_of_pi():
    momenta = al.point(np.array([[1.0, 0.6, 0.0, 0.8], [2.0, 0, 0, -2.0], [1.0, 0, 0, 1.0], [1.0, 0, 0, -1.0],
                                 [1.0, 1.0, 0, 0], [1.0, -1.0, 0, 0], [1.3, 0.5, -1.2, 0.0]]))
    Z = tw.from_momentum(momenta)
    assert np.abs(al.coordinates(tw.momentum(Z)) - al.coordinates(momenta)).max() < 1e-13
    assert np.abs(tw.coordinates(Z)[:, :2]).max() < 1e-14                              # through the origin
    assert np.abs(tw.norm(Z)).max() < 1e-14
    # The momentum ignores omega, and translating the twistor (to another event on the ray) keeps it.
    moved = tw.through(EVENT, Z)
    assert np.abs(al.coordinates(tw.momentum(moved)) - al.coordinates(momenta)).max() < 1e-13
    # Phase and scale: e^{I phi} Z has the same momentum, 2 Z four times it.
    turned = ((np.cos(0.7) + np.sin(0.7) * tw.PSEUDOSCALAR) * Z).cast(al.Even)
    assert np.abs(al.coordinates(tw.momentum(turned)) - al.coordinates(momenta)).max() < 1e-13
    assert np.abs(al.coordinates(tw.momentum(2 * Z)) - 4 * al.coordinates(momenta)).max() < 1e-12


def test_flag_is_null_turns_with_the_phase_and_follows_lorentz_transport():
    """The flag of pi is a null bivector k ^ s; e^{I theta} Z turns s by 2 theta about k; and a Lorentz rotor R carries
    the flag direction of Z to that of R Z, as it carries vectors, R s ~R, up to a multiple of the new momentum."""
    momenta = al.point(np.array([[1.0, 0.6, 0.0, 0.8], [2.0, 0, 0, -2.0], [1.0, 0, 0, 1.0], [1.3, 0.5, -1.2, 0.0]]))
    Z = tw.from_momentum(momenta)
    flag, k = tw.flag(Z), np.asarray(al.coordinates(momenta))
    lowered = np.einsum("ac,...cd,db->...ab", al.ETA4, flag, al.ETA4)
    assert np.abs(np.einsum("...ab,...ab->...", flag, lowered)).max() < 1e-14                   # F . F = 0
    wedge = np.einsum("...ab,...c->...abc", flag, k)
    wedge = wedge + np.einsum("...abc->...bca", wedge) + np.einsum("...abc->...cab", wedge)
    assert np.abs(wedge).max() < 1e-14 and np.abs(flag).max() > 0.1                            # k ^ F = 0
    across = tw.flag_direction(Z, al.time)
    read = lambda value: np.asarray(value.kernel)[..., 0]
    assert np.abs(read(across | across) + 1).max() < 1e-14 and np.abs(read(across | momenta)).max() < 1e-14
    assert np.abs(-read(across | tw.flag_direction(tw.with_phase(Z, 0.3), al.time)) - np.cos(0.6)).max() < 1e-14
    t, x, y, z = al.SPACETIME
    rotor = (0.4 * (t ^ x)).exp() * (0.7 * (y ^ z)).exp() * (-0.5 * (t ^ z)).exp()
    moved = (rotor * Z).cast(tw.Twistor)
    difference = np.asarray(al.coordinates(tw.flag_direction(moved, al.time))) - np.asarray(al.coordinates(rotor >> across))
    pole = np.asarray(al.coordinates(tw.momentum(moved)))
    assert np.abs(difference - difference[:, :1] / pole[:, :1] * pole).max() < 1e-13


def test_complex_weyl_invariant_of_kerr():
    """2 tr_omega(F F~) = (C^2 + i C.*C) / 2 = 24 M^2 / (r - i a cos theta)^6 for Kerr, its real part the quadratic
    density; for Schwarzschild it is real."""
    mass, spin = 1.0, 0.6
    geometry = cartan.kerr(mass, spin)
    connection = cf.connection(geometry)                                                # normal: Schouten = 0
    points = al.point(np.array([[0.2, 3.0, 1.0, 2.0], [0.0, 2.5, -1.0, -1.5], [0.0, 4.0, 0.5, 0.3]]))
    value = tw.as_complex(tw.weyl_invariant(geometry, connection)(points))
    radius = np.asarray(cartan.kerr_schild_fields(points, mass, spin)[0].kernel)[..., 0]
    cosine = al.coordinates(points)[:, 3] / radius
    expected = 24 * mass ** 2 / (radius - 1j * spin * cosine) ** 6
    assert np.abs(value / expected - 1).max() < 1e-10
    quadratic = np.asarray(iv.quadratic_invariant(geometry, connection)(points).kernel)[..., 0]
    assert np.abs(value.real - quadratic).max() < 1e-12
    schwarzschild = cartan.schwarzschild(mass)
    value = tw.as_complex(tw.weyl_invariant(schwarzschild, cf.connection(schwarzschild))(points))
    assert np.abs(value.imag).max() < 1e-12 * np.abs(value.real).max()
