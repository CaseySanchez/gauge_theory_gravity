"""Cartan geometry from a tetrad, on extensors.

The coframe is the extensor  e : Direction(frame) <- Direction(coordinate),  e(d_nu) = e^a_nu e_a, with events
and directions intrinsic antivectors (see `algebra`).  The metric is the flat duality map eta pulled back across the
coframe by its adjugate, g = e^adj eta e : Slice <- Direction, read as the form g(v) & w (equal to e | e).  The frame
vectors are E = e^-1, the Christoffel symbols come from the derivative of g, and the Lorentz connection is the bivector-valued map  Omega : Bivector <- Direction  with
1/2 [Omega(d_mu), v] = (nabla_mu v)^a e_a,  so that the curvature of Omega is the Riemann bivector map (checked
against an independent index implementation in tests).

Geometries are written the way gauge theory gravity writes them, as the identity plus dyads: the
Painleve-Gullstrand coframe is `Direction + shift * (t & Direction)`, a position-dependent direction paired with the
time slice.  Each geometry carries its frame field E in closed form: for an identity-plus-null-dyad coframe, where the
dyad squares to zero, the inverse is the identity minus the dyad.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from .fields import derivative, permute_inputs

Direction = al.Direction
Slice = al.Slice
Scalar = al.Scalar
CoframeType = al.ALG.gatype((Direction, Direction))      # Direction- <- Direction-

# The lab's unit time direction, the time slice that reads it, and the projectors on the time axis and on the space
# across it.
time = al.time                                           # [] Direction-
TimePart = time * (al.t & Direction)                     # Direction- <- Direction-
SpacePart = Direction - TimePart                         # Direction- <- Direction-


@dataclass(frozen=True)
class Geometry:
    """A coframe field `coframe(x)` together with its frame field `frame(x) = coframe(x)^-1`, both
    position -> (Direction <- Direction).  Calling a geometry evaluates its coframe."""
    coframe: Callable[[Extensor], Extensor]
    frame: Callable[[Extensor], Extensor]

    def __call__(self, x: Extensor) -> Extensor:
        return self.coframe(x)


Coframe = Geometry | Callable[[Extensor], Extensor]      # a geometry, or a bare coframe field


def frame_at(coframe: Coframe, x: Extensor) -> Extensor:
    """The frame vectors E = e^-1 at x: the geometry's closed form when it has one, else the numerical inverse."""
    if isinstance(coframe, Geometry):
        return coframe.frame(x)
    return coframe(x).inverse()


def coordinate(position: Extensor, axis: int) -> Extensor:
    """The coordinate of a position along the basis direction `axis` (t, x, y, z = 0..3), a scalar."""
    return al.SLICES[axis] & position


def spatial_radius(position: Extensor) -> Extensor:
    """The Cartesian radius of an event, a scalar."""
    return (-SpacePart(position).squared()).square_root()


# --- geometry from a coframe ---------------------------------------------------------------------
def metric_map(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> g = e^adj eta e : Slice <- Direction, the flat duality map pulled back across the coframe by congruence.
    The adjugate carries a slice back across e with no metric and no inverse, `c & e(v) == e.adjugate()(c) & v`, so
    `g(v) & w == eta(e(v)) & e(w)`; it is symmetric in the metric-free sense, `g == g.adjugate()`."""
    def duality(x: Extensor) -> Extensor:
        e = coframe(x)
        return e.adjugate()(al.FLAT_METRIC(e))
    return duality


def metric(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> the metric as a form, g(v) & w : Scalar <- (Direction, Direction); equal to e | e."""
    duality = metric_map(coframe)

    def form(x: Extensor) -> Extensor:
        return duality(x) & Direction
    return form


def christoffel_lower(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> Gamma_{lam mu nu} = 1/2 (d_mu g_{lam nu} + d_nu g_{lam mu} - d_lam g_{mu nu}) as a trilinear form."""
    gradient = derivative(metric(coframe))               # inputs (alpha, beta, step gamma): d_gamma g_{alpha beta}

    def symbols(x: Extensor) -> Extensor:
        d = gradient(x)
        return 0.5 * (permute_inputs(d, (0, 2, 1)) + d - permute_inputs(d, (2, 0, 1)))

    return symbols


def frame_vectors(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> E = e^-1 : Direction(coordinate) <- Direction(frame)."""
    def inverse(x: Extensor) -> Extensor:
        return frame_at(coframe, x)
    return inverse


def lorentz_connection(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> Omega(x): Bivector <- Direction.
    W(c, mu, b) = Gamma_{c mu b} + e_c . (d_mu E_b)  are the lowered spin-connection coefficients in the frame,
    and the bivector is  Omega(d_mu) = 1/2 W(d_k, mu, d_l) s^k ^ s^l  over the coordinate slices s^k that read the
    frame directions d_k (the two traces pair each frame input of W with the slice that reads it)."""
    symbols = christoffel_lower(coframe)
    gradient = derivative(frame_vectors(coframe))        # coordinate direction <- (frame b, step mu)

    def connection(x: Extensor) -> Extensor:
        e = coframe(x)
        E = frame_at(coframe, x)
        W = symbols(x)(E, Direction, E) + permute_inputs(Direction | e(gradient(x)), (0, 2, 1))   # Scalar <- (c, mu, b)
        bivectors = (W * (Slice ^ Slice)).trace(1, 4).trace(2, 3)                                 # Bivector <- mu
        return (0.5 * bivectors).cast(al.Bivector)

    return connection


# --- geometries ----------------------------------------------------------------------------------
def null_dyad(profile: Extensor, null: Extensor) -> Extensor:
    """profile * null (null . a): a dyad that squares to zero whenever `null` is orthogonal to itself in the flat
    metric, so that identity plus the dyad is inverted by identity minus it.  Direction <- Direction."""
    return profile * null * (null.dual_inverse() & Direction)


def painleve_gullstrand_shift(x: Extensor, mass: Extensor | float) -> Extensor:
    """The PG shift beta = sqrt(2M/r) x/r, a spatial direction."""
    spatial = SpacePart(x)                                                      # [] Direction, squares to -r^2
    return spatial * (2 * mass / spatial_radius(x) ** 3).square_root()          # [] Direction-


def painleve_gullstrand(x: Extensor, mass: Extensor | float) -> Extensor:
    """Schwarzschild in Painleve-Gullstrand Cartesian coordinates (t,x,y,z):
    e^0 = dt,  e^i = dx^i + beta^i dt,  beta = sqrt(2M/r) x/r.   ds^2 = dt^2 - sum (e^i)^2.
    As the identity plus a dyad: the shift, a direction, paired with the time slice. `mass` may be a scalar field
    of the position: with M(t) varying this is a valid torsion-free geometry with matter (a toy, not Vaidya)."""
    return Direction + painleve_gullstrand_shift(x, mass) * (al.t & Direction)  # Direction- <- Direction-


def painleve_gullstrand_frame(x: Extensor, mass: Extensor | float) -> Extensor:
    """The inverse of the PG coframe: the shift is spatial and the time slice does not read it, so the dyad squares
    to zero and the inverse is the identity minus the dyad, E_0 = d_t - beta."""
    return Direction - painleve_gullstrand_shift(x, mass) * (al.t & Direction)  # Direction- <- Direction-


def schwarzschild(mass: float = 1.0) -> Geometry:
    """The Painleve-Gullstrand coframe of a hole of constant mass: vacuum, regular through r = 2M."""
    def coframe(x: Extensor) -> Extensor:
        return painleve_gullstrand(x, mass)

    def frame(x: Extensor) -> Extensor:
        return painleve_gullstrand_frame(x, mass)

    return Geometry(coframe, frame)


def growing_mass(mass: float = 1.0, growth: float = 1.0, at: float = 0.0, width: float = 2.0) -> Geometry:
    """Painleve-Gullstrand form with M(t) = mass + growth/2 (1 + tanh((t - at)/width)): a toy with matter
    (Ricci != 0), so the special-conformal (Cotton) sector is switched on while the mass changes."""
    def mass_at(x: Extensor) -> Extensor:
        rising = 1 + ((coordinate(x, 0) - at) / width).tanh()                   # [] Scalar+
        return mass + 0.5 * growth * rising

    def coframe(x: Extensor) -> Extensor:
        return painleve_gullstrand(x, mass_at(x))

    def frame(x: Extensor) -> Extensor:
        return painleve_gullstrand_frame(x, mass_at(x))

    return Geometry(coframe, frame)


def friedmann(exponent: float = 2.0 / 3.0) -> Geometry:
    """Spatially flat FRW: e^0 = dt, e^i = a(t) dx^i with a = t^exponent (conformally flat, with matter)."""
    def scale(x: Extensor) -> Extensor:
        return (coordinate(x, 0).log() * exponent).exp()                         # [] Scalar+

    def coframe(x: Extensor) -> Extensor:
        return TimePart + scale(x) * SpacePart                                   # Direction- <- Direction-

    def frame(x: Extensor) -> Extensor:
        return TimePart + SpacePart / scale(x)                                   # Direction- <- Direction-

    return Geometry(coframe, frame)


def de_sitter(hubble: float = 0.5) -> Geometry:
    """de Sitter in flat slicing, e^0 = dt, e^i = e^{H t} dx^i: Einstein with cosmological constant 3 H^2 and
    conformally flat."""
    def scale(x: Extensor) -> Extensor:
        return (hubble * coordinate(x, 0)).exp()                                 # [] Scalar+

    def coframe(x: Extensor) -> Extensor:
        return TimePart + scale(x) * SpacePart                                   # Direction- <- Direction-

    def frame(x: Extensor) -> Extensor:
        return TimePart + SpacePart / scale(x)                                   # Direction- <- Direction-

    return Geometry(coframe, frame)


def static_lapse() -> Geometry:
    """e^0 = N(x) dt, e^i = dx^i with a smooth lapse N: generically not conformally flat, with matter."""
    def lapse(x: Extensor) -> Extensor:
        return (1 + 0.3 * coordinate(x, 1).sin() * coordinate(x, 2).cos()
                + 0.1 * coordinate(x, 3) ** 2)                                   # [] Scalar+

    def coframe(x: Extensor) -> Extensor:
        return lapse(x) * TimePart + SpacePart                                   # Direction- <- Direction-

    def frame(x: Extensor) -> Extensor:
        return TimePart / lapse(x) + SpacePart                                   # Direction- <- Direction-

    return Geometry(coframe, frame)


def boost(velocity: float) -> Extensor:
    """The Lorentz rotor of a boost along +x with the given speed, as a sandwich on spacetime directions."""
    return ((al.t ^ al.x) * (np.arctanh(velocity) / 2)).exp()                   # [] Rotor+


def boosted(coframe: Coframe, velocity: float) -> Geometry:
    """The same geometry moving along +x with the given speed: lab coordinates x, rest-frame coordinates
    y = L x (a boost in the t-x plane), coframe e'(x) = e(L x) L, with `L = R >> Direction` the boost map, and frame
    E'(x) = L^-1 E(L x), the reverse sandwich of the output."""
    rotor = boost(velocity)
    carry = rotor >> Direction                                                  # Direction- <- Direction-

    def moving(x: Extensor) -> Extensor:
        return coframe(rotor >> x)(carry)

    def frame(x: Extensor) -> Extensor:
        return rotor << frame_at(coframe, rotor >> x)

    return Geometry(moving, frame)


# --- Kerr ----------------------------------------------------------------------------------------
def kerr_schild_fields(x: Extensor, mass: float, spin: float) -> tuple[Extensor, Extensor, Extensor]:
    """Oblate radius r, profile f = M r^3 / (r^4 + a^2 z^2) and ingoing null direction l = d_t - n of the Cartesian
    Kerr-Schild form, for a hole of spin a per unit mass turning in the xy plane, as `(t ^ x ^ y ...)`. The outgoing
    spatial direction n = (r x + a y, r y - a x, (r^2 + a^2) z / r) / (r^2 + a^2) is the position turned against the
    hole's sense by its commutator with the turning plane, plus a height term along the spin axis."""
    spatial = SpacePart(x)                                                      # [] Direction-
    turning = al.x ^ al.y                                                       # [] Bivector+
    height = coordinate(x, 3)                                                   # [] Scalar+
    offset = -spatial.squared() - spin ** 2                                     # [] Scalar+, rho^2 - a^2
    radius_squared = (offset + (offset ** 2 + 4 * spin ** 2 * height ** 2).square_root()) / 2
    radius = radius_squared.square_root()                                       # [] Scalar+
    # x d_x + y d_y turned a quarter turn about z: (x, y) -> (y, -x), the position's commutator with the plane.
    twist = spatial.commutator(turning)                                         # [] Antivector-
    outward = (radius * spatial + spin * twist.cast(Direction)
               + (spin ** 2 / radius) * height * al.DIRECTIONS[3]) / (radius_squared + spin ** 2)
    profile = mass * radius ** 3 / (radius_squared ** 2 + spin ** 2 * height ** 2)
    return radius, profile, time - outward


def kerr(mass: float = 1.0, spin: float = 0.6) -> Geometry:
    """Kerr in ingoing Cartesian Kerr-Schild coordinates as an identity plus a null dyad, gauge theory gravity's
    position gauge field:  e = Direction - f l (l . Direction),  g = eta - 2 f l l  (signature + - - -), regular
    across the future horizon.  l is null in the flat metric, so the frame field is  E = Direction + f l (l . ).
    With spin 0 it is Schwarzschild in Eddington-Finkelstein form."""
    def coframe(x: Extensor) -> Extensor:
        _, profile, null = kerr_schild_fields(x, mass, spin)
        return Direction - null_dyad(profile, null)                             # Direction- <- Direction-

    def frame(x: Extensor) -> Extensor:
        _, profile, null = kerr_schild_fields(x, mass, spin)
        return Direction + null_dyad(profile, null)                             # Direction- <- Direction-

    return Geometry(coframe, frame)


def kerr_kretschmann(x: Extensor, mass: float, spin: float) -> Extensor:
    """The Kretschmann scalar of Kerr, 48 M^2 (r^6 - 15 r^4 u + 15 r^2 u^2 - u^3) / (r^2 + u)^6 with u = a^2 cos^2 theta
    = a^2 z^2 / r^2 in the oblate radius r."""
    radius, _, _ = kerr_schild_fields(x, mass, spin)
    r2 = radius ** 2
    u = spin ** 2 * coordinate(x, 3) ** 2 / r2
    return 48 * mass ** 2 * (r2 ** 3 - 15 * r2 ** 2 * u + 15 * r2 * u ** 2 - u ** 3) / (r2 + u) ** 6
