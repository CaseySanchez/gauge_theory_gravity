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
  * the twistor norm  h(Z1, Z2) = 4 <~Z1 Z2>_{0,6},  a complex number (scalar + I part), Hermitian, signature (2,2);
  * the split  Z = omega + pi  into the null kernels of the point at infinity and the origin,
    omega = 1/2 n_inf n_o Z  (n_inf omega = 0),  pi = 1/2 n_o n_inf Z  (n_o pi = 0);
  * incidence with an event x, its conformal point X = n_o + x - 1/2 x^2 n_inf:  X Z = 0, equivalently
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


def inner(first: Extensor, second: Extensor) -> Extensor:
    """The twistor norm h(Z1, Z2) = 4 <~Z1 Z2>_{0,6}: a scalar plus a pseudoscalar, the complex number
    `as_complex` reads. Antilinear in the first twistor, linear in the second."""
    product = ~first * second
    return 4 * (product.restrict_grade(0) + product.restrict_grade(6))


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


def conformal_point(event: Extensor) -> Extensor:
    """X = n_o + x - 1/2 x^2 n_inf, the null vector of an event: the origin translated to x."""
    vector = as_vector(event)
    return n_o + vector - 0.5 * vector.squared() * n_inf


def incidence(event: Extensor, twistor_: Extensor) -> Extensor:
    """X Z, an odd multivector that vanishes exactly when the twistor's alpha-plane contains the event."""
    return conformal_point(event) * twistor_


def through(event: Extensor, momentum: Extensor) -> Extensor:
    """The twistor incident with `event` whose pi half is `momentum`: omega = 1/2 x n_inf pi. Over the momenta it
    sweeps the event's alpha-plane, a complex 2-plane of T."""
    momentum = pi(momentum)
    return (momentum + 0.5 * (as_vector(event) * n_inf) * momentum).cast(Twistor)


def annihilator(twistor_: Extensor, tolerance: float = 1e-9) -> np.ndarray:
    """The real vectors V of R^{2,4} with V Z = 0, as rows of blade coefficients: none for a twistor with
    h(Z, Z) != 0, a totally null 2-plane, the twistor's light ray, for h(Z, Z) = 0."""
    products = np.asarray((al.Vector * twistor_).kernel)                 # Odd <- Vector, as (odd blades, 6)
    _, singular, rows = np.linalg.svd(products)
    rank = int((singular > tolerance * max(singular.max(), 1.0)).sum())
    return rows[rank:]


def ray(twistor_: Extensor) -> tuple[Extensor, Extensor]:
    """An event on the light ray of a null twistor and the ray's null direction, as intrinsic directions.
    The annihilator plane holds the conformal points X of the ray's events, normalised by X . n_inf = 1,
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
# The Lorentz bivectors e_a ^ e_b keep the omega/pi split; their 2x2 blocks on pi, and the spinor form epsilon_AB.
_PAIRS = [(a, b) for a in range(4) for b in range(a + 1, 4)]
_LORENTZ = np.stack([matrix((al.SPACETIME[a] ^ al.SPACETIME[b]).cast(Twistor))[2:, 2:] for a, b in _PAIRS])
_EPSILON = np.array([[0.0, 1.0], [-1.0, 0.0]])


def with_phase(twistor_: Extensor, angle) -> Extensor:
    """e^{I angle} Z: the same light ray and momentum, its flag turned by twice the angle about the momentum."""
    angle = np.asarray(angle, dtype=np.float64)
    turn = al.mv.scalar(np.cos(angle)[..., None]) + al.mv.scalar(np.sin(angle)[..., None]) * PSEUDOSCALAR
    return (turn * twistor_).cast(Twistor)


def flag(twistor_: Extensor) -> np.ndarray:
    """The null flag of pi as the components F^{ab} of a real bivector over the frame (t, x, y, z), along two new last
    axes: the real part of the self-dual bivector epsilon_AB pi^A pi^B, read against each Lorentz generator e_a ^ e_b.
    It is k ^ s, with k the momentum and s a unit spacelike direction across the ray fixed up to multiples of k; unlike
    the momentum it sees the phase of pi (it is quadratic, not Hermitian, in pi), which turns s about k. Not a product
    in the even subalgebra (I is central there, so no element conjugates a twistor), hence computed in coordinates."""
    spinor = coordinates(twistor_)[..., 2:]                                              # [..., 2] pi_A
    paired = np.einsum("...i,ij,gjk,...k->...g", spinor, _EPSILON, _LORENTZ, spinor).real    # [..., 6]
    components = np.zeros(paired.shape[:-1] + (4, 4))
    for g, (a, b) in enumerate(_PAIRS):
        # <(e_a ^ e_b)(e_a ^ e_b)> = -eta_aa eta_bb: the reciprocal generator turns the reading into a component.
        components[..., a, b] = paired[..., g] / (-al.ETA4[a, a] * al.ETA4[b, b])
        components[..., b, a] = -components[..., a, b]
    return components


def flag_direction(twistor_: Extensor, observer: Extensor) -> Extensor:
    """The flag's unit direction across the ray that the observer (a timelike frame direction) sees as orthogonal to
    itself: u . F / (u . k), orthogonal to both u and the momentum k. Local twistor transport turns it like the
    parallel transported screen."""
    u = np.broadcast_to(np.asarray(al.coordinates(observer)), coordinates(twistor_).shape[:-1] + (4,)) @ al.ETA4
    k = np.asarray(al.coordinates(momentum(twistor_)))
    across = np.einsum("...a,...ab->...b", u, flag(twistor_)) / (u * k).sum(-1, keepdims=True)
    return al.point(across / np.sqrt(-np.einsum("...a,ab,...b->...", across, al.ETA4, across))[..., None])


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
