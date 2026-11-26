"""Twistors as the spinors of Spin(2,4) = SU(2,2), on extensors.

The even subalgebra of Cl(2,4) is the algebra of complex 4x4 matrices: the pseudoscalar `I`, which squares to -1
and commutes with every even element, is the imaginary unit. Twistor space is a minimal left ideal of it,

    T = Even * P,       P = 1/4 (1 + I x^y)(1 + I t^p),

four complex dimensions (eight real). Any even spinor `Psi` of the gauge theory is four twistors side by side, and
`Psi * P` keeps one of them. Spin(2,4), the connection and the transport act on twistors by left multiplication,
so `covariant_derivative`, `transport` and `holonomy_loop` act on twistors unchanged, and with the normal conformal
connection (`conformal.connection(coframe, conformal.schouten(coframe))`) that action is Penrose's local twistor
transport.

The rest of the structure, all of it in the algebra:
  * dual twistors as the right ideal  P Even,  paired with twistors by  W Z in P Even P,  the complex line of P,
    with no metric;
  * the twistor norm  h(Z1, Z2) = 4 <~Z1 Z2>_{0,6},  a complex number (scalar + I part), Hermitian, signature (2,2):
    the pairing composed with the antilinear duality map  Z -> ~Z  onto the dual twistors;
  * the split  Z = omega + pi  into the null kernels of the point at infinity and the origin,
    omega = 1/2 n_inf n_o Z  (n_inf omega = 0),  pi = 1/2 n_o n_inf Z  (n_o pi = 0);
  * incidence with an event x, its conformal point X = T_x n_o ~T_x (the origin carried there by its translator
    T_x = exp(1/2 x ^ n_inf)):  X Z = 0, equivalently ~T_x Z has no omega half, i.e. Z = T_x pi_0, i.e.
    omega = 1/2 x n_inf pi, the index-free form of Penrose's omega = i x pi (in the coordinates of BASIS,
    omega^A = -(i / sqrt 2) x^{AA'} pi_A' with x^{AA'} a Hermitian 2x2 matrix of determinant x^2);
  * the real vectors V with V Z = 0: a totally null 2-plane, the light ray of Z, when h(Z, Z) = 0, else none.

`P` is its own reverse, so the norm lands in the complex line spanned by P and needs no reference element; the
ideal's elements are still general even multivectors, so `Twistor` is the type `Even`.
"""
from __future__ import annotations

import numpy as np
from numga import stack, where
from numga.extensor import Extensor

from . import algebra as al
from . import cartan
from .cartan import Coframe
from .curvature import curvature
from .fields import Field

Twistor = al.Even
Direction = al.Direction
t, x, y, z, p, n = al.t, al.x, al.y, al.z, al.p, al.n
n_o, n_inf = al.n_o, al.n_inf

PSEUDOSCALAR = t * x * y * z * p * n                                  # [] Pseudoscalar+, squares to -1
IDEMPOTENT = 0.25 * (1 + PSEUDOSCALAR * (x * y)) * (1 + PSEUDOSCALAR * (t * p))   # [] Even+, P P = P = ~P
ZERO = al.mv.scalar([0.0])                                             # [] Scalar+
OMEGA_PART = 0.5 * n_inf * n_o                                         # [] Even+, projector onto omega, along pi
PI_PART = 0.5 * n_o * n_inf                                            # [] Even+, projector onto pi, along omega

# A basis of T adapted to the split: omega_A = OMEGA_PART s_A, pi_A = PI_PART s_A from the two seeds
# s = sqrt(2) P, sqrt(2) x z P, with h(pi_A, omega_B) = delta_AB and both halves null.
_SEEDS = [np.sqrt(2.0) * IDEMPOTENT, np.sqrt(2.0) * (x * z) * IDEMPOTENT]
OMEGA_BASIS = stack([(OMEGA_PART * seed).cast(Twistor) for seed in _SEEDS])   # [2] Even+
PI_BASIS = stack([(PI_PART * seed).cast(Twistor) for seed in _SEEDS])         # [2] Even+
BASIS = stack([*[OMEGA_BASIS[k] for k in range(2)], *[PI_BASIS[k] for k in range(2)]])  # [4] Even+
# Each coordinate of a twistor is its norm against the basis element of the other half.
_READERS = stack([*[PI_BASIS[k] for k in range(2)], *[OMEGA_BASIS[k] for k in range(2)]])  # [4] Even+
# The translation generators along the coordinate directions, s^mu ^ n_inf with s^mu the slice reading d_mu.
TRANSLATIONS = (al.SLICES * n_inf).cast(Twistor)                       # [4] Even+


def twistor(spinor: Extensor) -> Extensor:
    """The twistor of an even spinor: its column in the ideal, `Psi P`."""
    return (spinor * IDEMPOTENT).cast(Twistor)


def pairing(dual: Extensor, twistor_: Extensor) -> Extensor:
    """The pairing of a dual twistor W, in the right ideal P Even, with a twistor Z, in the left ideal Even P:
    W Z lies in P Even P, the complex line of P, and `trace` reads it as a complex number. Complex-linear in both and
    free of any metric: invariant under every invertible even V acting as Z -> V Z, W -> W V^-1, not only Spin(2,4)."""
    return trace(dual * twistor_)


def conjugate(twistor_: Extensor) -> Extensor:
    """The dual twistor of Z under the norm: ~Z = P ~Psi, in the right ideal since P is its own reverse. The reverse
    is antilinear (~I = -I), so this is the duality map Z -> conj(Z) of the Hermitian form, h(Z1, Z2) = pairing(~Z1, Z2)."""
    return ~twistor_


def inner(first: Extensor, second: Extensor) -> Extensor:
    """The twistor norm h(Z1, Z2) = 4 <~Z1 Z2>_{0,6}: a scalar plus a pseudoscalar, the complex number
    `as_complex` reads. Antilinear in the first twistor, linear in the second: the pairing of the dual twistor
    `conjugate(Z1)` with Z2, invariant under the V with ~V V = 1, Spin(2,4) = SU(2,2)."""
    return pairing(conjugate(first), second)


def as_complex(value: Extensor) -> np.ndarray:
    """A scalar-plus-pseudoscalar extensor as complex numbers, I read as i: <c>_0 + i <c I^-1>_0."""
    real = (value.restrict_grade(0) + ZERO).kernel[..., 0]
    imaginary = ((value * -PSEUDOSCALAR).restrict_grade(0) + ZERO).kernel[..., 0]
    return np.asarray(real) + 1j * np.asarray(imaginary)


def real_part(value: Extensor) -> Extensor:
    """The scalar part of a scalar-plus-pseudoscalar extensor, as a scalar extensor."""
    return value.restrict_grade(0) + ZERO


def imaginary_part(value: Extensor) -> Extensor:
    """The I part of a scalar-plus-pseudoscalar extensor, <c I^-1>_0, as a scalar extensor."""
    return (value * -PSEUDOSCALAR).restrict_grade(0) + ZERO


def trace(operator: Extensor) -> Extensor:
    """The complex trace of an even element acting on T, 4 <A>_{0,6}: the even subalgebra is M(4,C), its scalar part
    a quarter of the real part of the trace and its I part a quarter of the imaginary part."""
    return 4 * (operator.restrict_grade(0) + operator.restrict_grade(6))


# The product of the four negative-square directions: h(Z, POSITIVE Z) is the positive-definite Hermitian form of the
# lab frame, sum |c_k|^2 over the coordinates of BASIS on each half (preserved by S(U(2) x U(2)), not by SU(2,2)).
POSITIVE = x * y * z * n                                               # [] Even+


def size(twistor_: Extensor) -> Extensor:
    """The frame's positive norm of a twistor, sqrt(h(Z, x y z n Z)): for omega or pi alone, the root of the sum of
    the squared magnitudes of its two coordinates. The form is clamped at zero, where round-off can take a vanishing
    half (omega of a twistor through its own event) just below it."""
    return real_part(inner(twistor_, (POSITIVE * twistor_).cast(Twistor))).clip(0.0, None).square_root()


def eigenvalues(operator: Extensor) -> np.ndarray:
    """The four complex eigenvalues of an even element acting on twistors, along a new last axis: the roots of its
    characteristic polynomial, whose coefficients follow from the twistor traces of its powers by Newton's identities.
    NaN where the element is not finite. Clustered eigenvalues lose digits (a fourfold root a quarter of them), so for a
    versor near 1 take the spectrum of its logarithm."""
    powers = [operator]
    for _ in range(3):
        powers.append(powers[-1] * operator)
    p1, p2, p3, p4 = (as_complex(trace(power)) for power in powers)
    e2 = (p1 * p1 - p2) / 2
    e3 = (e2 * p1 - p1 * p2 + p3) / 3
    e4 = (e3 * p1 - e2 * p2 + p1 * p3 - p4) / 4
    polynomials = np.stack([np.ones_like(p1), -p1, e2, -e3, e4], axis=-1)
    flat = polynomials.reshape(-1, 5)
    roots = [np.roots(c) if np.isfinite(c).all() else np.full(4, np.nan + 0j) for c in flat]
    return np.array(roots).reshape(polynomials.shape[:-1] + (4,))


def norm(twistor_: Extensor) -> np.ndarray:
    """h(Z, Z), real: zero for a twistor that is a light ray, its sign the sign of the helicity."""
    return as_complex(inner(twistor_, twistor_)).real


def omega(twistor_: Extensor) -> Extensor:
    """The omega half of a twistor, annihilated by the point at infinity."""
    return (OMEGA_PART * twistor_).cast(Twistor)


def pi(twistor_: Extensor) -> Extensor:
    """The pi half of a twistor, annihilated by the origin: its momentum spinor."""
    return (PI_PART * twistor_).cast(Twistor)


def momentum(twistor_: Extensor) -> Extensor:
    """The momentum of a twistor, the moment map of the translations: the direction with coordinates
    p^mu = -Im h(Z, (s^mu ^ n_inf) Z). For a twistor with pi != 0 it is null and future pointing, the flag pole of pi,
    Penrose's p = pi conj(pi); it does not see omega. Computed from one product: the scalar part is cyclic and I is
    central in the even subalgebra, so -Im h(Z, T Z) = 4 <T Z ~Z I>_0, and Z ~Z, its own reverse, has only grades 0
    and 4, of which the grade-4 part pairs with the bivector T through I."""
    density = (twistor_ * ~twistor_).restrict_grade(4) * PSEUDOSCALAR                    # [...] Bivector
    paired = (TRANSLATIONS * density[..., None]).restrict_grade(0) + ZERO                 # [..., 4]
    return ((4 * paired) * al.DIRECTIONS).sum(axis=-1).cast(Direction)                  # [...] Direction-


def from_momentum(momentum_: Extensor) -> Extensor:
    """The twistor of a light ray through the origin with momentum `momentum_`, a future-pointing null direction:
    omega = 0 and pi = n_o k s for a seed s, scaled so that `momentum` returns the given momentum. Both seeds are tried
    and the one giving the larger pi kept (each one alone fails for a single direction)."""
    vector = as_vector(momentum_)
    candidates = [(n_o * vector * seed).cast(Twistor) for seed in _SEEDS]
    weights = [al.t & momentum(candidate) for candidate in candidates]                  # time components, >= 0
    larger = weights[0] > weights[1]
    chosen, weight = where(larger, candidates[0], candidates[1]), where(larger, weights[0], weights[1])
    # The scale on the left: an even multivector times a batch of scalars on the right compiles very slowly on the GPU.
    return (((al.t & momentum_) / weight).square_root() * chosen).cast(Twistor)


def coordinates(twistor_: Extensor) -> np.ndarray:
    """The complex coordinates (omega^0, omega^1, pi_0, pi_1) along a new last axis: Z = sum_k c_k BASIS[k]. In them
    the norm is h(Z, Z) = 2 Re(conj(omega) . pi)."""
    return as_complex(inner(_READERS, twistor_[..., None]))


def from_coordinates(values) -> Extensor:
    """The twistor with complex coordinates `values` (..., 4)."""
    values = np.asarray(values, dtype=complex)
    real = al.mv.scalar(values.real[..., None])
    imaginary = al.mv.scalar(values.imag[..., None])
    return ((real + imaginary * PSEUDOSCALAR) * BASIS).sum(axis=-1).cast(Twistor)


def matrix(operator: Extensor) -> np.ndarray:
    """The complex 4x4 matrix of an even element acting on twistors by left multiplication, in BASIS. For a versor
    it lies in SU(2,2); for a bivector, in su(2,2)."""
    return np.moveaxis(coordinates((operator[..., None] * BASIS).cast(Twistor)), -1, -2)


def as_vector(event: Extensor) -> Extensor:
    """The extrinsic spacetime vector of an event or direction, the one its coordinates multiply t, x, y, z by."""
    return event.dual_inverse()


def incidence(event: Extensor, twistor_: Extensor) -> Extensor:
    """X Z with X = T_x n_o ~T_x, an odd multivector that vanishes exactly when the twistor's alpha-plane contains the
    event: X Z = T_x n_o (~T_x Z) is zero when the twistor, translated back by ~T_x, has no omega half."""
    return al.conformal_point(event) * twistor_


def through(event: Extensor, momentum: Extensor) -> Extensor:
    """The twistor incident with `event` whose pi half is `momentum`: the twistor of the parallel ray through the
    origin, pi, translated to the event, T_x pi (so omega = 1/2 x n_inf pi). Over the momenta it sweeps the event's
    alpha-plane, a complex 2-plane of T."""
    return (al.translator(event) * pi(momentum)).cast(Twistor)


def annihilator(twistor_: Extensor, tolerance: float = 1e-9) -> np.ndarray:
    """The real vectors V of R^{2,4} with V Z = 0, as rows of blade coefficients: none for a twistor with
    h(Z, Z) != 0, a totally null 2-plane, the twistor's light ray, for h(Z, Z) = 0."""
    products = np.asarray((al.Vector * twistor_).kernel)                 # Odd <- Vector, as (odd blades, 6)
    _, singular, rows = np.linalg.svd(products)
    rank = int((singular > tolerance * max(singular.max(), 1.0)).sum())
    return rows[rank:]


def ray(twistor_: Extensor) -> tuple[Extensor, Extensor]:
    """An event on the light ray of a null twistor and the ray's null direction, as intrinsic directions.
    The annihilator plane holds the conformal points X of the ray's events, normalized by X . n_inf = 1,
    and its one vector orthogonal to n_inf, the ray's direction. Undefined for pi = 0, the ray at infinity."""
    plane = al.mv.vector(al.CTX.xp.asarray(annihilator(twistor_)))     # [2] Vector+
    towards = np.asarray((plane | n_inf).kernel)[..., 0]                # X . n_inf of each spanning vector
    at = int(np.argmax(np.abs(towards)))
    point = plane[at] / towards[at]
    direction = towards[1 - at] * plane[at] - towards[at] * plane[1 - at]
    return _event(point), _event(direction)


def _event(vector: Extensor) -> Extensor:
    """The spacetime direction whose coordinates a six-vector has along t, x, y, z, read by the slices."""
    return al.point(np.asarray((vector | al.SLICES).kernel)[..., 0])


# --- the null flag of pi ----------------------------------------------------------------------------
# The conjugation of the spinor form: t y z is odd, so it anticommutes with I and turns the Hermitian pairing of pi
# with its reverse into a bilinear one. n_inf pi C ~pi is the complex bivector epsilon_AB pi^A pi^B: a Lorentz
# bivector in grade 2 and its I-dual in grade 4.
CONJUGATION = t * y * z                                                # [] Odd+


def with_phase(twistor_: Extensor, angle) -> Extensor:
    """e^{I angle} Z: the same light ray and momentum, its flag turned by twice the angle about the momentum."""
    angle = np.asarray(angle, dtype=np.float64)
    turn = al.mv.scalar(np.cos(angle)[..., None]) + al.mv.scalar(np.sin(angle)[..., None]) * PSEUDOSCALAR
    return (turn * twistor_).cast(Twistor)


def flag(twistor_: Extensor) -> Extensor:
    """The null flag of pi, a real Lorentz bivector: the real part of the self-dual bivector epsilon_AB pi^A pi^B,
    the grade-2 part of sqrt(8) n_inf pi C ~pi with C the conjugation t y z. It is k ^ s, with k the momentum and s a
    unit spacelike direction across the ray fixed up to multiples of k; unlike the momentum it sees the phase of pi (it
    is quadratic, not Hermitian, in pi), which turns s about k."""
    spinor = pi(twistor_)
    return al.sector(np.sqrt(8.0) * (n_inf * spinor * CONJUGATION * ~spinor).restrict[2], "M")   # [...] Bivector+


def flag_direction(twistor_: Extensor, observer: Extensor) -> Extensor:
    """The flag's unit direction across the ray that the observer (a timelike frame direction) sees as orthogonal to
    itself: u . F / (u . k), orthogonal to both u and the momentum k. Local twistor transport turns it like the
    parallel transported screen."""
    u, k = as_vector(observer), as_vector(momentum(twistor_))
    across = (u * flag(twistor_)).restrict[1] / (u | k)                                  # [...] Vector+
    return (across / (-(across | across)).square_root()).dual().cast(Direction)        # [...] Direction-


# --- the Weyl curvature seen by twistors ----------------------------------------------------------
def weyl_invariant(coframe: Coframe, omega_: Field) -> Field:
    """x -> 2 tr_omega(F F~) contracted over the frame: the twistor trace of the curvature squared restricted to the
    omega half, a complex number (scalar plus I part). Lorentz bivectors keep the split, and on the omega half they act
    by their self-dual part, so this is half the complex square of the Weyl tensor, (C^2 + i C.*C) / 2, for the normal
    conformal connection: its real part is the existing quadratic density, its imaginary part half the Chern-Pontryagin
    density. For Kerr it is 24 M^2 / (r - i a cos theta)^6, sign of the dual as in the Kerr formula with spin along +z."""
    strength = curvature(omega_)

    def invariant(x: Extensor) -> Extensor:
        frame = cartan.frame_at(coframe, x)
        framed = strength(x)(frame, frame)                                                # Bivector <- (a, b)
        squared = OMEGA_PART * framed * ~framed
        restricted = squared.restrict_grade(0) + squared.restrict_grade(6)
        return 2 * restricted.contract(1, 3).contract(1, 2)                               # [] Scalar + Pseudoscalar

    return invariant
