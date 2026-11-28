"""The particle-like Einstein-Dirac solutions of Finster, Smoller and Yau (Phys. Rev. D 59 (1999) 104020,
gr-qc/9801079, "FSY") in the spacetime algebra, for docs/PLAN_DIRAC_SOLITON.md. Test-only: scipy, the shooting and the
Chebyshev fits stay out of the library.

Two fermions in the singlet, static and spherically symmetric, units G = c = hbar = 1. FSY (2.11), (3.4), (3.6), (5.4)-(5.7):

    ds^2 = dt^2 / T^2 - dr^2 / A - r^2 dOmega^2,
    Psi_a = e^{-i omega t} (u1 e_a, sigma_r u2 e_a)  (Dirac representation),   u1 = sqrt T alpha / r,  u2 = i sqrt T beta / r,
    sqrt A alpha' = alpha / r - (omega T + m) beta,
    sqrt A beta'  = (omega T - m) alpha - beta / r,
    r A' = 1 - A - 16 pi omega T^2 (alpha^2 + beta^2),
    2 r A T' / T = A - 1 - 16 pi omega T^2 (alpha^2 + beta^2) + 32 pi T alpha beta / r + 16 pi m T (alpha^2 - beta^2).

They are regular at r = 0 (alpha ~ alpha_1 r, beta ~ r^2) and found by shooting in omega with T(0) = 1 and m = 1 (FSY
section 6; rescaling then gives T(oo) = 1 and the normalization, but the unscaled solution already solves the field
equations, which is all a pointwise check needs). The field equations are local, so on an interval [r1, r2] away from
the origin the solution is fitted by Chebyshev series and every field is a jax-traceable function of x4 = (t, x, y, z).

In Cartesian coordinates FSY's Dirac matrices G^j (2.14)-(2.17) are the frame E_0 = T d_t, E_k = d_k + (sqrt A - 1) n_k n.d
(n = x / r) against the vacuum gamma^a: the coframe is e^0 = dt / T, e^k = dx^k + (1/sqrt A - 1) n^k n.dx, identity plus a
radial dyad. Their Dirac-representation spinors map to the STA as in Doran and Lasenby: an upper Pauli spinor
(a0 + i a3, -a2 + i a1) is a0 + a_k I sigma_k, the column (phi, eta) is phi + eta sigma_3, i is right multiplication by
I sigma_3, and gamma_mu acts as gamma_mu psi gamma_0 (checked in `test_fsy_solution`).

The same fields in the CSTA (`geometry`, `csta_spinor`, `csta_connection`): the coframe as identity plus the radial
dyad (1/sqrt A - 1) P and (1/T - 1) on the time part, P(v) = x (x.v)/r^2 = -x (x . v)_eta / r^2 the radial projector,
with its closed-form inverse; Z_a = dirac.from_sta(psi_a).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev
from scipy.integrate import solve_ivp

from gauge_theory_gravity import algebra as al, cartan, conformal as cf, dirac as dr

mv = al.mv
GAMMA = al.SPACETIME
I_STA = GAMMA[0] * GAMMA[1] * GAMMA[2] * GAMMA[3]
SIGMA = [GAMMA[k] * GAMMA[0] for k in range(4)]                  # sigma_k = gamma_k gamma_0, k = 1..3
KAPPA = 8 * np.pi
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


# --- FSY's ODEs (5.4)-(5.7) and the shooting ----------------------------------------------------------------------
def rhs(r, y, omega, mass, coupling=None):
    """FSY (5.4)-(5.7); with `coupling` = beta, the CSTA's full transport (PLAN_DIRAC_SOLITON.md Phase 2): the mass
    m0 = m - sqrt 2 (1 + 1/beta) - kappa T (alpha^2 - beta^2) / (3 beta r^2), the radial term 1/r + delta with
    delta = -kappa T alpha beta / (beta r^2), and the contact density
    L_c = kappa sigma^2 / (12 beta) + kappa K^2 / (8 beta), sigma = 2 T (alpha^2 - beta^2) / r^2, K = 4 T alpha beta / r^2,
    in the Einstein equations."""
    alpha, beta, A, T = y
    root = np.sqrt(max(A, 1e-300))
    density = alpha ** 2 + beta ** 2
    m, radial, contact = mass, 1 / r, 0.0
    if coupling is not None:
        sigma, K = 2 * T * (alpha ** 2 - beta ** 2) / r ** 2, 4 * T * alpha * beta / r ** 2
        m = mass - np.sqrt(2) * (1 + 1 / coupling) - KAPPA * sigma / (6 * coupling)
        radial = 1 / r - KAPPA * K / (4 * coupling)
        contact = KAPPA * sigma ** 2 / (12 * coupling) + KAPPA * K ** 2 / (8 * coupling)
    return [(radial * alpha - (omega * T + m) * beta) / root,
            ((omega * T - m) * alpha - radial * beta) / root,
            (1 - A - 16 * np.pi * omega * T ** 2 * density - 8 * np.pi * r ** 2 * contact) / r,
            T * (A - 1 - 16 * np.pi * omega * T ** 2 * density + 32 * np.pi * T * radial * alpha * beta
                 + 16 * np.pi * m * T * (alpha ** 2 - beta ** 2) + 8 * np.pi * r ** 2 * contact) / (2 * r * A)]


def regular_start(r, alpha1, omega, mass, T0=1.0):
    """The Taylor series at the origin to second order, from the ODEs themselves."""
    return [alpha1 * r, (omega * T0 - mass) * alpha1 * r ** 2 / 3,
            1 - 16 * np.pi / 3 * omega * T0 ** 2 * alpha1 ** 2 * r ** 2,
            T0 + 4 * np.pi / 3 * alpha1 ** 2 * T0 ** 2 * (mass - 2 * omega * T0) * r ** 2]


def _diverges(r, y, omega, mass, coupling=None):
    return min(5.0 - abs(y[0]) - abs(y[1]), y[2] - 1e-3)


_diverges.terminal = True


def integrate(alpha1, omega, mass=1.0, outer=3000.0, inner=1e-5, coupling=None):
    return solve_ivp(rhs, (inner, outer), regular_start(inner, alpha1, omega, mass), args=(omega, mass, coupling),
                     method="DOP853", rtol=1e-13, atol=1e-15, events=_diverges, dense_output=True)


def shoot(alpha1, low, high, mass=1.0, steps=55, coupling=None):
    """The omega between low and high at which the sign of the diverging spinor flips: a bound state."""
    side = lambda omega: np.sign(integrate(alpha1, omega, mass, coupling=coupling).y[0, -1])  # noqa: E731
    s_low = side(low)
    assert side(high) != s_low
    for _ in range(steps):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if side(middle) == s_low else (low, middle)
    return 0.5 * (low + high)


# --- Dirac representation -> STA ----------------------------------------------------------------------------------
def pauli_to_sta(c0, c1):
    """(a0 + i a3, -a2 + i a1) -> a0 + a_k I sigma_k."""
    a0, a3, a2, a1 = jnp.real(c0), jnp.imag(c0), -jnp.real(c1), jnp.imag(c1)
    return a0 + a1 * (I_STA * SIGMA[1]) + a2 * (I_STA * SIGMA[2]) + a3 * (I_STA * SIGMA[3])


def column_to_sta(column):
    """(phi, eta) -> phi + eta sigma_3, an even multivector."""
    return (pauli_to_sta(column[0], column[1]) + pauli_to_sta(column[2], column[3]) * SIGMA[3]).cast(al.Even)


def dirac_matrices():
    """gamma_mu (lower index) in FSY's Dirac representation (2.13): gamma_0 = diag(1, -1), gamma_k = -gamma^k."""
    zero, one = np.zeros((2, 2)), np.eye(2)
    gammas = [np.block([[one, zero], [zero, -one]]).astype(complex)]
    gammas += [-np.block([[zero, s], [-s, zero]]) for s in PAULI]
    return gammas


# --- Chebyshev fits as jax-traceable functions of r ----------------------------------------------------------------
def _clenshaw(coefficients, u):
    b1, b2 = 0.0 * u, 0.0 * u
    for c in coefficients[:0:-1]:
        b1, b2 = 2 * u * b1 - b2 + c, b1
    return u * b1 - b2 + coefficients[0]


@dataclass(frozen=True)
class Soliton:
    """The ground state for alpha_1 with omega shot between `low` and `high`, fitted on [inner, outer]."""
    alpha1: float = 0.1
    low: float = 1.5
    high: float = 1.55
    mass: float = 1.0
    inner: float = 0.4
    outer: float = 6.0
    degree: int = 90
    coupling: float | None = None      # beta of the CSTA under the full transport; None for GTG's Lorentz coupling
    given_omega: float | None = None   # use this omega instead of shooting: a local solution, not a bound state

    @cached_property
    def omega(self):
        if self.given_omega is not None:
            return self.given_omega
        return shoot(self.alpha1, self.low, self.high, self.mass, coupling=self.coupling)

    @cached_property
    def effective_mass(self):
        """m0 = m - sqrt 2 (1 + 1/beta) under the full transport, the mass of the oracle's effective action."""
        return self.mass if self.coupling is None else self.mass - np.sqrt(2) * (1 + 1 / self.coupling)

    @cached_property
    def solution(self):
        return integrate(self.alpha1, self.omega, self.mass, coupling=self.coupling)

    @cached_property
    def coefficients(self):
        """[4, degree + 1]: alpha, beta, A, T in Chebyshev series on [inner, outer]."""
        nodes = np.cos(np.pi * (np.arange(self.degree + 1) + 0.5) / (self.degree + 1))
        radii = self.inner + (nodes + 1) * (self.outer - self.inner) / 2
        values = self.solution.sol(radii)
        return np.stack([chebyshev.chebfit(nodes, values[k], self.degree) for k in range(4)])

    def profile(self, r):
        """(alpha, beta, A, T) at r, from the fits; jax-traceable."""
        u = (2 * r - self.inner - self.outer) / (self.outer - self.inner)
        return [_clenshaw(self.coefficients[k], u) for k in range(4)]

    def fit_residual(self, samples=200):
        """max |fit' - rhs(fit)| over [inner, outer]: the floor for every later check."""
        r = np.linspace(self.inner, self.outer, samples)
        u = (2 * r - self.inner - self.outer) / (self.outer - self.inner)
        values = np.stack([chebyshev.chebval(u, c) for c in self.coefficients])
        slopes = np.stack([chebyshev.chebval(u, chebyshev.chebder(c)) * 2 / (self.outer - self.inner)
                           for c in self.coefficients])
        exact = np.stack([rhs(ri, values[:, i], self.omega, self.mass, self.coupling) for i, ri in enumerate(r)],
                         axis=1)
        return float(np.abs(slopes - exact).max())

    # --- fields of x4 ---------------------------------------------------------------------------------------------
    @staticmethod
    def _radial(x4):
        r = jnp.sqrt(x4[1] ** 2 + x4[2] ** 2 + x4[3] ** 2)
        return r, jnp.stack([x4[1], x4[2], x4[3]]) / r

    def coframe(self, x4):
        """e[a, mu]: e^0 = dt / T, e^k = dx^k + (1/sqrt A - 1) n^k n.dx."""
        r, n = self._radial(x4)
        _, _, A, T = self.profile(r)
        spatial = jnp.eye(3) + (1 / jnp.sqrt(A) - 1) * jnp.outer(n, n)
        return jnp.zeros((4, 4)).at[0, 0].set(1 / T).at[1:, 1:].set(spatial)

    def columns(self, x4):
        """[2, 4] complex: FSY's Psi_1, Psi_2 (3.4) in the Dirac representation."""
        r, n = self._radial(x4)
        alpha, beta, _, T = self.profile(r)
        u1, u2 = jnp.sqrt(T) * alpha / r, 1j * jnp.sqrt(T) * beta / r
        sigma_r = sum(n[k] * PAULI[k] for k in range(3))
        phase = jnp.exp(-1j * self.omega * x4[0])
        basis = jnp.eye(2)
        return jnp.stack([phase * jnp.concatenate([u1 * basis[a], u2 * (sigma_r @ basis[a])]) for a in range(2)])

    def spinor_multivectors(self, x4):
        return [column_to_sta(c) for c in self.columns(x4)]

    def spinors(self, x4):
        """[2, 32]: the two STA spinors' kernels."""
        return jnp.stack([psi.kernel for psi in self.spinor_multivectors(x4)])

    # --- the same fields in the CSTA, functions of a position --------------------------------------------------------
    @staticmethod
    def _x4(position):
        return jnp.stack([cartan.coordinate(position, m).kernel.reshape(()) for m in range(4)])

    def _parts(self, position):
        """(A, T, P): the profile at the position's radius and the radial projector Direction <- Direction."""
        spatial = cartan.SpacePart(position)                                     # [] Direction, squares to -r^2
        r = jnp.sqrt(-spatial.squared().kernel.reshape(()))
        _, _, A, T = self.profile(r)
        projector = (-1 / r ** 2) * spatial * (spatial.dual_inverse() & al.Direction)
        return A, T, projector

    def geometry(self):
        def coframe(position):
            A, T, P = self._parts(position)
            return al.Direction + (1 / T - 1) * cartan.TimePart + (1 / jnp.sqrt(A) - 1) * P

        def frame(position):
            A, T, P = self._parts(position)
            return al.Direction + (T - 1) * cartan.TimePart + (jnp.sqrt(A) - 1) * P
        return cartan.Geometry(coframe, frame)

    def csta_spinor(self, k):
        """position -> Z_k = dirac.from_sta(psi_k), k = 0, 1."""
        return lambda position: dr.from_sta(self.spinor_multivectors(self._x4(position))[k])

    def csta_connection(self, special):
        """Levi-Civita, the coframe and the special-conformal field: the singlet's spin vanishes, so no contorsion."""
        return cf.connection(self.geometry(), special)

    def shifted_special(self, beta):
        """The special-conformal field that solves the K rows under the full transport, from the singlet's total
        current C = C_1 + C_2: F^b_a = (1/beta)(1 + kappa tr C / 3) delta^b_a - (kappa/beta) C^b_a, f = F e."""
        geometry = self.geometry()

        def special(position):
            current = sum(dr.special_conformal_current(self.csta_spinor(k)(position)) for k in range(2))  # [a, b] = C^b_a
            trace = current[0, 0] + current[1, 1] + current[2, 2] + current[3, 3]
            coframe = geometry(position)
            total = 0 * coframe
            for a in range(4):
                for b in range(4):
                    weight = (1 + KAPPA * trace / 3) / beta * float(a == b) - KAPPA / beta * current[b, a]
                    total = total + weight * (al.DIRECTIONS[a] * (al.SLICES[b] & coframe))
            return total
        return special
