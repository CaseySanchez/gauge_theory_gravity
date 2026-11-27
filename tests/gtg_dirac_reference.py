"""An independent oracle for Gauge Theory Gravity with Dirac matter, in Lasenby, Doran and Gull's own formulation
(gr-qc/0405033, "LDG"): the spacetime algebra, a position-gauge field h_bar, the rotation gauge field Omega_mu, and a
spinor psi, an even element of the spacetime algebra acted on from the right by I sigma_3 and gamma_0.

Test-only: it uses numpy matrices for h_bar and its inverse, and none of the CSTA machinery (no conformal vectors, no
twistors, no coframe extensors). The spacetime algebra is the subalgebra of t, x, y, z of `algebra.ALG`, whose
signature (+ - - -) is LDG's. Fields are functions of a coordinate array x4 (t, x, y, z):

  coframe(x4) -> e[a, mu], with g_mu = h^-1(e_mu) = e^a_mu gamma_a  (so det(h)^-1 = det e)
  spinor(x4)  -> the 32 even-blade coefficients of psi (jax-traceable)
  rotation(x4) -> [4] bivectors Omega_mu = Omega(e_mu)

Formulas used, by LDG equation number:
  (3.56)  L = det(h)^-1 < g^mu (d_mu + 1/2 Omega_mu) psi I gamma_3 ~psi - m psi ~psi >
  (4.2)   R_mu nu = d_mu Omega_nu - d_nu Omega_mu + Omega_mu x Omega_nu, R(B) = R(h(B)), (4.11-4.12)
  (4.49)  H(a) = h_bar(grad' ^ h_bar'^-1(a)),   (4.53)  omega(a) = -H(a) + 1/2 a . (d_b ^ H(b)),  Omega(b) = omega(h^-1(b))
  (3.65)  tau = det(h) D_mu (h_bar(e^mu) det(h)^-1), D_mu M = d_mu M + Omega_mu x M  (the torsion trace)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gauge_theory_gravity import algebra as al

mv = al.mv
GAMMA = al.SPACETIME                                                 # gamma_0..gamma_3
ETA = np.diag(al.ETA4)
I_STA = GAMMA[0] * GAMMA[1] * GAMMA[2] * GAMMA[3]
EVEN_ZERO = mv.even(np.zeros(32))


def vector(components):
    """sum_a c_a gamma_a."""
    return sum((components[a] * GAMMA[a] for a in range(4)), 0 * GAMMA[0])


def coefficients(v) -> np.ndarray:
    """c_a with v = sum c_a gamma_a, read with the reciprocal frame gamma^a = eta^aa gamma_a."""
    return np.array([scalar(v * (ETA[a] * GAMMA[a])) for a in range(4)])


def scalar(m) -> float:
    part = m.restrict_grade(0)
    return float(np.asarray(part.kernel).reshape(-1)[0]) if part.kernel.size else 0.0


def even(kernel):
    return mv.even(kernel)


def commutator(a, b):
    return 0.5 * (a * b - b * a)


# --- the position-gauge field --------------------------------------------------------------------------------------
def h_bar_matrix(e: np.ndarray) -> np.ndarray:
    """The matrix of h_bar in the gamma_a basis: h_bar(e^mu) = g^mu, the reciprocal of g_mu = e^a_mu gamma_a.
    Columns act on components of a vector in the gamma_a basis."""
    E = np.linalg.inv(e)                                             # E[mu, a]: g^mu = E[mu, a] gamma^a
    g_up = np.stack([E[mu] * ETA for mu in range(4)])                # g^mu in gamma_a components
    e_up = np.diag(ETA)                                              # e^mu = eta^mu mu gamma_mu
    return g_up.T @ np.linalg.inv(e_up)                              # h_bar @ e^mu = g^mu


def reciprocal_frame(e: np.ndarray):
    """[4] vectors g^mu = h_bar(e^mu)."""
    E = np.linalg.inv(e)
    return [vector(E[mu] * ETA) for mu in range(4)]


# --- the rotation gauge field of a torsion-free h (LDG 4.49, 4.53) ---------------------------------------------------
def rotation_gauge(coframe, x4) -> list:
    """Omega_mu = omega(g_mu) from LDG's closed solution of the wedge equation, derivatives by jax."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)

    def inverse_h_bar(y):
        return jnp.linalg.inv(_h_bar_jnp(coframe(y)))

    Hb = np.asarray(_h_bar_jnp(coframe(x4)))
    d_inverse = np.asarray(jax.jacfwd(inverse_h_bar)(x4))           # [i, j, mu]
    e_up = [ETA[mu] * GAMMA[mu] for mu in range(4)]

    def h_bar(v):
        return vector(Hb @ coefficients(v))

    def H(a_components):
        """H(a) = h_bar(sum_mu e^mu ^ d_mu(h_bar^-1)(a)), h_bar applied as an outermorphism."""
        total = 0 * (GAMMA[0] ^ GAMMA[1])
        for mu in range(4):
            moved = vector(d_inverse[:, :, mu] @ a_components)
            total = total + (h_bar(e_up[mu]) ^ h_bar(moved))
        return total

    frame_basis = np.eye(4)
    Hs = [H(frame_basis[b]) for b in range(4)]
    trivector = sum((e_up[b] ^ Hs[b] for b in range(4)), 0 * (GAMMA[0] ^ GAMMA[1] ^ GAMMA[2]))

    def omega(a_components):
        a = vector(a_components)
        return -H(a_components) + 0.5 * (a * trivector).restrict_grade(2)

    e = np.asarray(coframe(x4))
    return [omega(e[:, mu]) for mu in range(4)]                        # Omega_mu = omega(g_mu)


def _h_bar_jnp(e):
    E = jnp.linalg.inv(e)
    g_up = E * jnp.asarray(ETA)[None, :]
    return g_up.T @ jnp.diag(1.0 / jnp.asarray(ETA))


# --- curvature (LDG 4.2, 4.9, 4.11, 4.12) --------------------------------------------------------------------------
def ricci_scalar(coframe, rotation_kernels, x4) -> float:
    """R from Omega_mu given as a jax-traceable function x4 -> [4, 15] bivector blade coefficients."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)
    values = np.asarray(rotation_kernels(x4))
    d = np.asarray(jax.jacfwd(rotation_kernels)(x4))                 # [nu, blade, mu]
    Om = [mv.bivector(values[mu]) for mu in range(4)]
    dOm = [[mv.bivector(d[nu, :, mu]) for nu in range(4)] for mu in range(4)]   # dOm[mu][nu] = d_mu Omega_nu
    g_up = reciprocal_frame(np.asarray(coframe(x4)))
    total = 0.0
    for mu in range(4):
        for nu in range(4):
            R = dOm[mu][nu] - dOm[nu][mu] + commutator(Om[mu], Om[nu])
            # R(B) = R(h(B)): with B = g^nu ^ g^mu ... the contraction d_a . R(a ^ b) summed is <(g^nu ^ g^mu) R_mu nu>
            total += scalar((g_up[nu] ^ g_up[mu]) * R)
    return total


# --- the Dirac action (LDG 3.56) -----------------------------------------------------------------------------------
def dirac_density(coframe, rotation, spinor, x4, mass: float = 0.0) -> float:
    """det(h)^-1 < g^mu (d_mu psi + 1/2 Omega_mu psi) I gamma_3 ~psi - m psi ~psi > at x4."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)
    e = np.asarray(coframe(x4))
    psi = even(spinor(x4))
    d = np.asarray(jax.jacfwd(spinor)(x4))                           # [32, mu]
    Om = rotation(x4)
    g_up = reciprocal_frame(e)
    kinetic = 0.0
    for mu in range(4):
        covariant = even(d[:, mu]) + 0.5 * (Om[mu] * psi)
        kinetic += scalar(g_up[mu] * covariant * I_STA * GAMMA[3] * ~psi)
    return abs(np.linalg.det(e)) * (kinetic - mass * scalar(psi * ~psi))


# --- the Dirac equation (LDG 3.57-3.62) ----------------------------------------------------------------------------
I_SIGMA3 = I_STA * GAMMA[3] * GAMMA[0]                               # I sigma_3


def torsion_trace(coframe, rotation_kernels, x4):
    """tau = det(h) D_mu (g^mu det(h)^-1) = (1/det e) [d_mu (det e g^mu) + Omega_mu x (det e g^mu)], a vector (3.65)."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)

    def weighted(y):                                                 # det e g^mu in gamma_a components, [mu, a]
        e = coframe(y)
        return jnp.abs(jnp.linalg.det(e)) * jnp.linalg.inv(e) * jnp.asarray(ETA)[None, :]

    values, d = np.asarray(weighted(x4)), np.asarray(jax.jacfwd(weighted)(x4))      # d[mu, a, nu]
    Om = [mv.bivector(k) for k in np.asarray(rotation_kernels(x4))]
    total = 0 * GAMMA[0]
    for mu in range(4):
        total = total + vector(d[mu, :, mu]) + commutator(Om[mu], vector(values[mu])).restrict_grade(1)
    return total / abs(np.linalg.det(np.asarray(coframe(x4))))


def dirac_residual(coframe, rotation_kernels, spinor, x4, mass: float = 0.0):
    """D psi I sigma_3 - m psi gamma_0 + 1/2 tau psi I sigma_3, which (3.62) sets to zero; D psi = g^mu D_mu psi."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)
    e = np.asarray(coframe(x4))
    psi = even(spinor(x4))
    d = np.asarray(jax.jacfwd(spinor)(x4))
    Om = [mv.bivector(k) for k in np.asarray(rotation_kernels(x4))]
    g_up = reciprocal_frame(e)
    operator = sum((g_up[mu] * (even(d[:, mu]) + 0.5 * (Om[mu] * psi)) for mu in range(4)), 0 * psi)
    tau = torsion_trace(coframe, rotation_kernels, x4)
    return operator * I_SIGMA3 - mass * psi * GAMMA[0] + 0.5 * tau * psi * I_SIGMA3


def euler_lagrange(coframe, rotation_kernels, spinor, x4, mass: float = 0.0) -> np.ndarray:
    """delta S / delta psi_A over the 32 even blades of the oracle's density: d L / d psi_A - d_mu (d L / d(d_mu psi_A)),
    the second derivatives of psi by jax."""
    x4 = jnp.asarray(x4, dtype=jnp.float64)

    def local(y, psi_kernel, d_psi):
        e = coframe(y)
        Om_k = rotation_kernels(y)
        E = jnp.linalg.inv(e) * jnp.asarray(ETA)[None, :]
        psi = even(psi_kernel)
        kinetic = 0.0
        for mu in range(4):
            g = sum((E[mu, a] * GAMMA[a] for a in range(4)), 0 * GAMMA[0])
            covariant = even(d_psi[:, mu]) + 0.5 * (mv.bivector(Om_k[mu]) * psi)
            kinetic = kinetic + (g * covariant * I_STA * GAMMA[3] * ~psi).restrict_grade(0).kernel.reshape(())
        return jnp.abs(jnp.linalg.det(e)) * (kinetic - mass * (psi * ~psi).restrict_grade(0).kernel.reshape(()))

    def momenta(y):
        return jax.grad(local, argnums=2)(y, spinor(y), jax.jacfwd(spinor)(y))       # [32, mu]

    direct = jax.grad(local, argnums=1)(x4, spinor(x4), jax.jacfwd(spinor)(x4))
    divergence = jnp.einsum("amm->a", jax.jacfwd(momenta)(x4))
    return np.asarray(direct - divergence)


def spin_trivector(spinor, x4):
    """S = 1/2 psi I gamma_3 ~psi (4.34)."""
    psi = even(spinor(jnp.asarray(x4, dtype=jnp.float64)))
    return 0.5 * psi * I_STA * GAMMA[3] * ~psi
