"""Dirac spinors as twistor fields: the conventions of the GTG Dirac action in the CSTA (docs/PLAN_DIRAC.md, phase 0).

A Dirac spinor is a twistor, an element of the left ideal Even P (`twistor.py`): eight real components, acted on by
Spin(2,4) from the left, with the central pseudoscalar I as the imaginary unit. The Dirac algebra acts on it through

    Gamma_a = I e_a p,      p = (n_o + n_inf) / sqrt 2 = (X + I_inf) / sqrt 2,

which satisfy {Gamma_a, Gamma_b} = 2 eta_ab and Gamma_a Gamma_b = e_a e_b for a != b, so the Lorentz sector of the
connection acts on twistors exactly as the rotation gauge field acts on GTG's spinor, psi -> 1/2 Omega(a) psi. Built from
the origin and the infinity twistor (the compensators of `first_order`), p is not a fixed frame choice: it is the
direction of X + I_inf. With this choice the twistor norm h is the Dirac bar pairing, h(Z1, Z2) = bar(psi1) psi2.

The rest of the conformal algebra acts on the Dirac spinor algebraically:
  * the dilation generator N = n_o ^ n_inf is the chirality, N = I Gamma_0 Gamma_1 Gamma_2 Gamma_3 (gamma_5), and the
    omega and pi halves of a twistor are its Weyl spinors (N = -1 and +1);
  * a translation is Gamma times a chiral projector, P_a = -sqrt 2 I Gamma_a Pi_+, and a special conformal
    transformation the other one, K_a = -sqrt 2 I Gamma_a Pi_-, with Pi_+- = (1 +- N) / 2.

The map from Lasenby, Doran and Gull's spinor (an even element psi of the spacetime algebra, the subalgebra of t, x,
y, z here, whose signature is theirs) is Z = psi REFERENCE, with REFERENCE the unit twistor fixed by Gamma_0 Z = Z and
e_y e_x Z = I Z. It turns their operations into these: gamma_mu psi gamma_0 -> Gamma_mu Z, psi I sigma_3 -> I Z,
<psi ~psi>_0 -> h(Z, Z), psi gamma_0 ~psi -> the current h(Z, Gamma_a Z), psi gamma_3 ~psi -> -h(Z, Gamma_a N Z),
and their kinetic form <gamma^a phi I gamma_3 ~psi>_0 -> Re h(Z_psi, I Gamma^a Z_phi).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numga import stack
from numga.extensor import Extensor

from . import algebra as al
from . import twistor as tw
from .curvature import curvature
from .fields import derivative
from .first_order import _moved_connection
from .variation import _euler, _monomial, _number, determinant

Twistor = tw.Twistor
I = tw.PSEUDOSCALAR
SPACETIME = al.SPACETIME
SQUARES = np.diag(al.ETA4)                                              # eta_aa, the frame raises by these

P_AXIS = (al.n_o + al.n_inf) / np.sqrt(2.0)                             # [] Vector+, = p, square +1
GAMMA = stack([(I * e * P_AXIS).cast(al.Even) for e in SPACETIME])     # [4] Even+, Gamma_a
GAMMA_UP = stack([(I * e * P_AXIS * s).cast(al.Even) for e, s in zip(SPACETIME, SQUARES)])   # [4] Even+, Gamma^a
CHIRALITY = (al.n_o ^ al.n_inf).cast(al.Even)                           # [] Even+, N = gamma_5, square +1
RIGHT = (0.5 * (1 + CHIRALITY)).cast(al.Even)                           # Pi_+, onto the pi half
LEFT = (0.5 * (1 - CHIRALITY)).cast(al.Even)                            # Pi_-, onto the omega half

_SEED = (0.25 * (1 + GAMMA[0]) * (1 - I * (al.y * al.x)) * (np.sqrt(2.0) * tw.IDEMPOTENT)).cast(Twistor)
REFERENCE = _SEED / float(np.sqrt(tw.norm(_SEED)))                      # [] Twistor, h(REFERENCE, REFERENCE) = 1


def from_sta(spinor: Extensor) -> Extensor:
    """The twistor of Lasenby, Doran and Gull's spinor, an even element of the spacetime subalgebra: psi REFERENCE."""
    return (spinor * REFERENCE).cast(Twistor)


def bar(first: Extensor, second: Extensor) -> Extensor:
    """The Dirac pairing bar(psi1) psi2 = h(Z1, Z2), a scalar plus a pseudoscalar (complex)."""
    return tw.inner(first, second)


def act(operator: Extensor, spinor: Extensor) -> Extensor:
    """An even element acting on a twistor from the left, kept in the ideal."""
    return (operator * spinor).cast(Twistor)


def scalar_density(spinor: Extensor) -> Extensor:
    """sigma = bar(psi) psi = h(Z, Z), real: Lasenby, Doran and Gull's <psi ~psi>_0."""
    return tw.real_part(bar(spinor, spinor))


def pseudoscalar_density(spinor: Extensor) -> Extensor:
    """omega = h(Z, I N Z) = bar(psi) i gamma_5 psi, real: minus their <psi ~psi I>_0."""
    return tw.real_part(bar(spinor, act(I * CHIRALITY, spinor)))


def current(spinor: Extensor) -> Extensor:
    """J_a = h(Z, Gamma_a Z), the frame components of the Dirac current psi gamma_0 ~psi, along a new last axis."""
    return tw.real_part(bar(spinor[..., None], act(GAMMA, spinor[..., None])))


def axial_current(spinor: Extensor) -> Extensor:
    """A_a = h(Z, Gamma_a N Z) = bar(psi) gamma_a gamma_5 psi: minus the frame components of the spin vector
    psi gamma_3 ~psi, along a new last axis. The spin trivector is S = 1/2 psi I gamma_3 ~psi = -1/2 I A."""
    return tw.real_part(bar(spinor[..., None], act(GAMMA * CHIRALITY, spinor[..., None])))


# --- phase 1: the Dirac density of a connection, broken by the compensators X = n_o and I_inf = n_inf -------------
ORIGIN, INFINITY = al.n_o, al.n_inf
# The eight real units of a twistor, Z = sum_k c_k UNITS[k]: the basis twistors and I times them.
UNITS = [tw.BASIS[k] for k in range(4)] + [(I * tw.BASIS[k]).cast(Twistor) for k in range(4)]
CONNECTIONS = ("full", "lorentz")


def coframe_of(connection: Extensor) -> Extensor:
    """The coframe a connection carries, e : Direction <- Direction: DX = Omega x X with its components along the
    compensators removed, v - (v . I_inf) X - (v . X) I_inf, read as a direction. The dilation sector, which moves
    X along itself, drops out."""
    moved = connection.commutator(ORIGIN)                                              # Vector <- Direction
    across = moved - (moved | INFINITY) * ORIGIN - (moved | ORIGIN) * INFINITY
    return across.dual().cast(al.Direction)


def transport_part(connection: Extensor, which: str) -> Extensor:
    """The part of Omega that moves the spinor: all of it ('full', the twistor transport), or the Lorentz sector, the
    stabilizer of both compensators ('lorentz', Lasenby, Doran and Gull's rotation gauge field)."""
    if which == "full":
        return connection
    if which == "lorentz":
        return al.sector(connection, "M")
    raise ValueError(f"connection must be one of {CONNECTIONS}")


def dirac_operator(omega, spinor, x: Extensor, which: str = "full") -> Extensor:
    """Gamma^a D_{E_a} Z at x: the frame vectors E_a = e^-1(d_a), D_v Z = v . grad Z + 1/2 Omega(v) Z with Omega the
    `transport_part`."""
    connection = omega(x)
    frame = coframe_of(connection).inverse()(al.DIRECTIONS)                          # [4] Direction, E_a
    moving = transport_part(connection, which)
    covariant = derivative(spinor)(x)(frame) + 0.5 * (moving(frame) * spinor(x))     # [4] Even, D_{E_a} Z
    return (GAMMA_UP * covariant).sum(axis=0).cast(Twistor)


def density(omega, spinor, x: Extensor, mass: float = 0.0, which: str = "full"):
    """|det e| (Re h(Z, I Gamma^a D_a Z) - m h(Z, Z)) at x as a component of d^4x: Lasenby, Doran and Gull's
    det(h)^-1 <h(d_a) a.D psi I gamma_3 ~psi - m psi ~psi> (their 3.56, with A = 0) through `from_sta`, when
    `which` is 'lorentz'."""
    Z = spinor(x)
    volume = jnp.abs(_number(determinant(coframe_of(omega(x)))))
    kinetic = tw.real_part(bar(Z, act(I, dirac_operator(omega, spinor, x, which))))
    return volume * (_number(kinetic) - mass * _number(scalar_density(Z)))


def _moved_spinor(spinor, alpha, x, amplitude):
    def moved(y: Extensor) -> Extensor:
        weight = _monomial(y - x, alpha)
        shift = sum((UNITS[k] * (amplitude[k] * weight) for k in range(8)), 0 * UNITS[0])
        return (spinor(y) + shift).cast(Twistor)
    return moved


def euler_lagrange_spinor(omega, spinor, x4, mass: float = 0.0, which: str = "full") -> np.ndarray:
    """delta S / delta c_k, [8], for Z = sum_k c_k UNITS[k]: the Dirac equation, first order in Z."""
    return _euler(lambda alpha, x, amplitude: density(omega, _moved_spinor(spinor, alpha, x, amplitude), x, mass, which),
                  (8,), x4, order=1)


def euler_lagrange_connection(omega, spinor, x4, mass: float = 0.0, which: str = "full") -> np.ndarray:
    """delta S / delta Omega^k_mu, [15, 4], of the Dirac density: the matter sources of the gravitational equations."""
    return _euler(lambda alpha, x, amplitude: density(_moved_connection(omega, alpha, x, amplitude), spinor, x, mass,
                                                     which), (15, 4), x4, order=1)


# --- phase 3: the spinor field equation in closed form -------------------------------------------------------------
def frame_at(connection: Extensor) -> Extensor:
    """[4] Direction: the frame vectors E_a = e^-1(d_a) of the coframe the connection carries."""
    return coframe_of(connection).inverse()(al.DIRECTIONS)


def torsion_trace(omega, x: Extensor) -> Extensor:
    """tau = -e^b T^a_{ab}, a spacetime vector: the trace of the torsion T(a, b) = -n_o _| F_P(a, b), read from the
    translation sector of the curvature of the Lorentz and translation parts of Omega on the frame (the dilation part
    would add b ^ e to F_P, and the Dirac field does not see it). It is Lasenby, Doran and Gull's
    det(h) D_mu (h_bar(e^mu) det(h)^-1), (3.65), whose vanishing makes the minimally coupled action give the minimally
    coupled equation."""
    def poincare(y: Extensor) -> Extensor:
        connection = omega(y)
        return al.sector(connection, "M") + al.sector(connection, "P")

    strength = al.sector(curvature(poincare)(x), "P")
    torsion = -(ORIGIN | strength)                                                     # Vector <- (a, b)
    frame = frame_at(omega(x))
    reciprocal = [e * s for e, s in zip(SPACETIME, SQUARES)]                           # e^a
    total = 0 * SPACETIME[0]
    for b in range(4):
        trace = sum(((torsion(frame[a], frame[b]) * reciprocal[a]).restrict_grade(0) for a in range(4)), 0 * I)
        total = total - (trace * reciprocal[b]).restrict_grade(1)
    return total.cast(al.Vector)


def spinor_equation(omega, spinor, x: Extensor, mass: float = 0.0, which: str = "full") -> Extensor:
    """W, a twistor, with delta S / delta c_k = 2 |det e| Re h(UNITS[k], W):

        W = I Gamma^a D_a Z + 1/2 I Gamma(tau) Z - m Z + 1/2 (O + ~O) Z,      O = 1/2 I Gamma^a (Omega - Omega_M)(E_a),

    D the Lorentz transport and tau the `torsion_trace`. The last term is the h-Hermitian part (the reverse is the
    h-adjoint) of what the other sectors add under the full transport; it is absent for 'lorentz'. With 'lorentz' this
    is Lasenby, Doran and Gull's coupled Dirac equation (3.62)."""
    connection = omega(x)
    frame = frame_at(connection)
    Z = spinor(x)
    lorentz = al.sector(connection, "M")
    covariant = derivative(spinor)(x)(frame) + 0.5 * (lorentz(frame) * Z)
    kinetic = I * (GAMMA_UP * covariant).sum(axis=0)
    tau = torsion_trace(omega, x)
    gamma_tau = (I * tau * P_AXIS).cast(al.Even)
    equation = kinetic + 0.5 * (I * gamma_tau * Z) - mass * Z
    if which == "full":
        extra = connection - lorentz
        operator = (0.5 * I * (GAMMA_UP * extra(frame)).sum(axis=0)).cast(al.Even)
        equation = equation + 0.5 * ((operator + ~operator) * Z)
    elif which != "lorentz":
        raise ValueError(f"connection must be one of {CONNECTIONS}")
    return equation.cast(Twistor)


# --- phase 4: the matter sources of the gravitational equations ----------------------------------------------------
STA_PSEUDOSCALAR = SPACETIME[0] * SPACETIME[1] * SPACETIME[2] * SPACETIME[3]      # [] Pseudoscalar of t, x, y, z
RECIPROCAL_FRAME = [e * s for e, s in zip(SPACETIME, SQUARES)]                     # e^a


def spin_trivector(spinor: Extensor) -> Extensor:
    """S = 1/2 psi I gamma_3 ~psi = -1/2 I_sta A, A = A_a e^a the axial current (Lasenby, Doran and Gull's 4.34). The
    Lorentz rows of the connection equation are |det e| S . g^mu, g^mu = E^mu_a e^a: their spin tensor S . h_bar(a)."""
    axial = current_vector(axial_current(spinor))
    return (-0.5 * (STA_PSEUDOSCALAR * axial)).restrict_grade(3)


def current_vector(components: Extensor) -> Extensor:
    """sum_a c_a e^a from frame components along the last axis (lowered index)."""
    return sum((components[..., a] * RECIPROCAL_FRAME[a] for a in range(4)), 0 * SPACETIME[0])


def reciprocal_coordinate_frame(connection: Extensor) -> list[Extensor]:
    """[4] vectors g^mu = E^mu_a e^a, the reciprocal of g_mu = e(d_mu): Lasenby, Doran and Gull's h_bar(e^mu)."""
    frame = frame_at(connection)                                                       # [4] Direction, E_a
    readings = al.SLICES[:, None] & frame[None, :]                                     # [mu, a] Scalar, E^mu_a
    return [sum((readings[mu, a] * RECIPROCAL_FRAME[a] for a in range(4)), 0 * SPACETIME[0]) for mu in range(4)]


def _stress_transport(connection: Extensor, which: str) -> Extensor:
    """The transport that the stress-energy sees: under the full transport, every sector but the translations, whose
    action on the spinor, 1/2 Gamma^a e(E_a) ^ n_inf, does not depend on the coframe."""
    if which == "full":
        return connection - al.sector(connection, "P")
    return transport_part(connection, which)


def stress_energy(omega, spinor, x: Extensor, mass: float = 0.0, which: str = "full") -> Extensor:
    """T^b_a = Re h(Z, I Gamma^b D_{E_a} Z) - delta^b_a L, a [4 a, 4 b] array of scalars, L the Lagrangian of `density`
    without the volume: Lasenby, Doran and Gull's T(a) = <a.D psi I gamma_3 ~psi>_1 - a L (4.20) in the frame. The
    translation rows of the connection equation are -|det e| E^mu_b T^b_a."""
    connection = omega(x)
    frame = frame_at(connection)
    Z = spinor(x)
    covariant = derivative(spinor)(x)(frame) + 0.5 * (_stress_transport(connection, which)(frame) * Z)   # [4] D_{E_a} Z
    kinetic = tw.real_part(bar(Z, act(I * GAMMA_UP[None, :], covariant[:, None].cast(Twistor))))           # [a, b]
    lagrangian = density(omega, spinor, x, mass, which) / jnp.abs(_number(determinant(coframe_of(connection))))
    return kinetic - lagrangian * np.eye(4)


SPECIAL_GENERATORS = stack([(e ^ ORIGIN).cast(al.Even) for e in SPACETIME])          # [4] Even, K_a


def special_conformal_current(spinor: Extensor) -> Extensor:
    """C^b_a = Re h(Z, 1/2 I Gamma^b K_a Z), [4 a, 4 b]: what the special-conformal rows of the connection equation
    receive under the full transport, |det e| E^mu_b C^b_a. Its symmetric part (lowered) is (sqrt 2 / 4) sigma eta, the
    rest the tensor bilinear bar(psi) sigma_ab (1 - gamma_5) psi."""
    return tw.real_part(bar(spinor, act(0.5 * I * GAMMA_UP[None, :] * SPECIAL_GENERATORS[:, None], spinor)))
