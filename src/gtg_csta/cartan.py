"""Cartan geometry from a tetrad, on extensors.

The coframe is the extensor  e : Spacetime(frame) <- Spacetime(coordinate),  e(d_nu) = e^a_nu e_a.
The metric is the form  g = e | e  (the algebra's own metric eta, pulled back), the frame vectors are
E = e.inverse(), the Christoffel symbols come from the derivative of g, and the Lorentz connection is the
bivector-valued map  Omega : Bivector <- Spacetime  with  1/2 [Omega(d_mu), v] = (nabla_mu v)^a e_a,
so that the curvature of Omega is the Riemann bivector map (checked against an independent
index implementation in tests).

Geometries are written the way gauge theory gravity writes them, as the identity plus dyads: the
Painleve-Gullstrand coframe is `Spacetime + shift * (t | Spacetime)`, a position-dependent vector paired with
the time form.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from .fields import derivative, permute_inputs

Spacetime = al.Spacetime
Scalar = al.Scalar
Coframe = Callable[[Extensor], Extensor]                 # position -> (Spacetime <- Spacetime)
CoframeType = al.ALG.gatype((Spacetime, Spacetime))      # Spacetime <- Spacetime

# The lab's time direction, and the projectors on the time axis and on the space across it.
time = al.t                                              # [] Vector+
TimePart = time * (time | Spacetime)                     # Spacetime <- Spacetime
SpacePart = Spacetime - TimePart                         # Spacetime <- Spacetime


def coordinate(position: Extensor, axis: int) -> Extensor:
    """The coordinate of a position along the basis direction `axis` (t, x, y, z = 0..3), a scalar."""
    return (al.SPACETIME[axis] | position) * al.ETA4[axis, axis]


# --- geometry from a coframe ---------------------------------------------------------------------
def metric(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> g = e | e : Scalar <- (Spacetime, Spacetime)."""
    def form(x: Extensor) -> Extensor:
        return coframe(x) | coframe(x)
    return form


def christoffel_lower(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> Gamma_{lam mu nu} = 1/2 (d_mu g_{lam nu} + d_nu g_{lam mu} - d_lam g_{mu nu}) as a trilinear form."""
    gradient = derivative(metric(coframe))               # inputs (alpha, beta, step gamma): d_gamma g_{alpha beta}

    def symbols(x: Extensor) -> Extensor:
        d = gradient(x)
        return 0.5 * (permute_inputs(d, (0, 2, 1)) + d - permute_inputs(d, (2, 0, 1)))

    return symbols


def frame_vectors(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> E = e^-1 : Spacetime(coordinate) <- Spacetime(frame)."""
    def inverse(x: Extensor) -> Extensor:
        return coframe(x).inverse()
    return inverse


def lorentz_connection(coframe: Coframe) -> Callable[[Extensor], Extensor]:
    """x -> Omega(x): Bivector <- Spacetime.
    W(c, mu, b) = Gamma_{c mu b} + e_c . (d_mu E_b)  are the lowered spin-connection coefficients in the frame,
    and the bivector is  Omega(d_mu) = 1/2 W(e_k, mu, e_l) e^k ^ e^l  (two metric contractions)."""
    symbols = christoffel_lower(coframe)
    gradient = derivative(frame_vectors(coframe))        # coordinate vector <- (frame b, step mu)

    def connection(x: Extensor) -> Extensor:
        e = coframe(x)
        E = e.inverse()
        W = symbols(x)(E, Spacetime, E) + permute_inputs(Spacetime | e(gradient(x)), (0, 2, 1))   # Scalar <- (c, mu, b)
        bivectors = (W * (Spacetime ^ Spacetime)).contract(1, 4).contract(2, 3)                    # Bivector <- mu
        return (0.5 * bivectors).cast(al.Bivector)

    return connection


# --- geometries ----------------------------------------------------------------------------------
def painleve_gullstrand(x: Extensor, mass: Extensor | float) -> Extensor:
    """Schwarzschild in Painleve-Gullstrand Cartesian coordinates (t,x,y,z):
    e^0 = dt,  e^i = dx^i + beta^i dt,  beta = sqrt(2M/r) x/r.   ds^2 = dt^2 - sum (e^i)^2.
    As the identity plus a dyad: the shift, a vector, paired with the time form. `mass` may be a scalar field
    of the position: with M(t) varying this is a valid torsion-free geometry with matter (a toy, not Vaidya)."""
    spatial = SpacePart(x)                                                      # [] Spacetime, squares to -r^2
    radius = (-spatial.squared()).square_root()                                 # [] Scalar+
    shift = spatial * (2 * mass / radius**3).square_root()                       # [] Spacetime
    return Spacetime + shift * (time | Spacetime)                               # Spacetime <- Spacetime


def schwarzschild(mass: float = 1.0) -> Coframe:
    """The Painleve-Gullstrand coframe of a hole of constant mass: vacuum, regular through r = 2M."""
    def coframe(x: Extensor) -> Extensor:
        return painleve_gullstrand(x, mass)
    return coframe


def growing_mass(mass: float = 1.0, growth: float = 1.0, at: float = 0.0, width: float = 2.0) -> Coframe:
    """Painleve-Gullstrand form with M(t) = mass + growth/2 (1 + tanh((t - at)/width)): a toy with matter
    (Ricci != 0), so the special-conformal (Cotton) sector is switched on while the mass changes."""
    def coframe(x: Extensor) -> Extensor:
        rising = 1 + ((coordinate(x, 0) - at) / width).tanh()                   # [] Scalar+
        return painleve_gullstrand(x, mass + 0.5 * growth * rising)
    return coframe


def friedmann(exponent: float = 2.0 / 3.0) -> Coframe:
    """Spatially flat FRW: e^0 = dt, e^i = a(t) dx^i with a = t^exponent (conformally flat, with matter)."""
    def coframe(x: Extensor) -> Extensor:
        scale = (coordinate(x, 0).log() * exponent).exp()                        # [] Scalar+
        return TimePart + scale * SpacePart                                      # Spacetime <- Spacetime
    return coframe


def static_lapse() -> Coframe:
    """e^0 = N(x) dt, e^i = dx^i with a smooth lapse N: generically not conformally flat, with matter."""
    def coframe(x: Extensor) -> Extensor:
        lapse = (1 + 0.3 * coordinate(x, 1).sin() * coordinate(x, 2).cos()
                 + 0.1 * coordinate(x, 3) ** 2)                                  # [] Scalar+
        return lapse * TimePart + SpacePart                                      # Spacetime <- Spacetime
    return coframe


def boost(velocity: float) -> Extensor:
    """The Lorentz rotor of a boost along +x with the given speed, as a sandwich on spacetime vectors."""
    return ((al.t ^ al.x) * (np.arctanh(velocity) / 2)).exp()                   # [] Rotor+


def boosted(coframe: Coframe, velocity: float) -> Coframe:
    """The same geometry moving along +x with the given speed: lab coordinates x, rest-frame coordinates
    y = L x (a boost in the t-x plane), coframe e'(x) = e(L x) L, with `L = R >> Spacetime` the boost map."""
    rotor = boost(velocity)
    carry = rotor >> Spacetime                                                  # Spacetime <- Spacetime

    def moving(x: Extensor) -> Extensor:
        return coframe(rotor >> x)(carry)
    return moving
