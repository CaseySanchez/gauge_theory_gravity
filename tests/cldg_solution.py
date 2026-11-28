"""The self-consistent Dirac cosmology of Challinor, Lasenby, Doran and Gull (Gen. Rel. Grav. 29 (1997) 1527), with the
cosmological constant of gr-qc/0311016 section 2.4, in the spacetime algebra and Lasenby, Doran and Gull's conventions:
a reference solution for docs/PLAN_DIRAC_SOLUTION.md. Test-only.

Spatially flat, Newtonian gauge, units G = 1 (kappa = 8 pi):

    h_bar(b) = b + H(t) (b . x) gamma_0,    x = x gamma_1 + y gamma_2 + z gamma_3,
    omega(b) = H b ^ gamma_0 + (kappa/2) b . S,    Omega_mu = omega(g_mu),
    psi = sqrt(rho) e^{I beta / 2} e^{-I sigma_3 chi} L,    S = 1/2 psi I gamma_3 ~psi,

with cos beta = +1 (particle) or -1 (antiparticle), L = exp(xi/2 (sin eta sigma_1 + cos eta sigma_2)) a constant boost
across gamma_3, rho' = -3 rho H, chi' = m cos beta + 3 pi rho, and the constraint
3 H^2 - 12 pi^2 rho^2 - 8 pi m rho cos beta - Lambda = 0. The singular branches (the thesis's regions (i) and (ii)),
with alpha = sqrt(3 Lambda) / 2 and s, c = sinh, cosh (alpha t):

    rho = alpha^2 / (6 pi s (alpha c +- m s)),
    H   = alpha (alpha cosh 2 alpha t +- m sinh 2 alpha t) / (3 s (alpha c +- m s)),
    chi = +- m t - 1/2 ln(alpha coth(alpha t) +- m)      (derived here: chi' = +- m + 3 pi rho).

Everything is a jax-traceable function of the coordinates x4 = (t, x, y, z), for the oracle's autodiff.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gauge_theory_gravity import algebra as al

mv = al.mv
GAMMA = al.SPACETIME
ETA = np.diag(al.ETA4)
I_STA = GAMMA[0] * GAMMA[1] * GAMMA[2] * GAMMA[3]
KAPPA = 8 * np.pi


def sigma(k):
    return GAMMA[k] * GAMMA[0]


@dataclass(frozen=True)
class Cosmology:
    mass: float = 1.0
    cosmological: float = 0.3          # Lambda
    sign: int = 1                      # cos beta: +1 particle, -1 antiparticle
    xi: float = 0.0                    # the constant boost L
    eta: float = 0.0

    @property
    def alpha(self):
        return np.sqrt(3 * self.cosmological) / 2

    def _sc(self, t):
        a = self.alpha
        return jnp.sinh(a * t), jnp.cosh(a * t)

    def density(self, t):
        a, m, s = self.alpha, self.mass, self.sign
        sh, ch = self._sc(t)
        return a ** 2 / (6 * np.pi * sh * (a * ch + s * m * sh))

    def hubble(self, t):
        a, m, s = self.alpha, self.mass, self.sign
        sh, ch = self._sc(t)
        return a * (a * jnp.cosh(2 * a * t) + s * m * jnp.sinh(2 * a * t)) / (3 * sh * (a * ch + s * m * sh))

    def phase(self, t):
        a, m, s = self.alpha, self.mass, self.sign
        return s * m * t - 0.5 * jnp.log(a / jnp.tanh(a * t) + s * m)

    # --- fields of x4 ---------------------------------------------------------------------------------------------
    def coframe(self, x4):
        """e[a, mu] with g_mu = e^a_mu gamma_a, the reciprocal of g^mu = h_bar(e^mu)."""
        H = self.hubble(x4[0])
        g_up = jnp.zeros((4, 4)).at[0, 0].set(1.0)                            # g^mu in gamma_a components
        for i in range(1, 4):
            g_up = g_up.at[i, 0].set(H * x4[i]).at[i, i].set(-1.0)
        return jnp.linalg.inv(g_up * jnp.asarray(ETA)[None, :])

    def spinor_multivector(self, x4):
        t = x4[0]
        rho, chi = self.density(t), self.phase(t)
        sign_part = 1 + 0 * GAMMA[0] * GAMMA[0] if self.sign > 0 else I_STA                  # e^{I beta / 2}
        rotation = jnp.cos(chi) - jnp.sin(chi) * (I_STA * sigma(3))
        boost = np.cosh(self.xi / 2) + np.sinh(self.xi / 2) * (np.sin(self.eta) * sigma(1) + np.cos(self.eta) * sigma(2))
        return (jnp.sqrt(rho) * sign_part * rotation * boost).cast(al.Even)

    def spinor(self, x4):
        return self.spinor_multivector(x4).kernel

    def spin(self, x4):
        psi = self.spinor_multivector(x4)
        return 0.5 * psi * I_STA * GAMMA[3] * ~psi

    def rotation_kernels(self, x4, torsion: bool = True):
        """[4, 15]: Omega_mu = H g_mu ^ gamma_0 + (kappa/2) g_mu . S."""
        e = self.coframe(x4)
        H = self.hubble(x4[0])
        S = self.spin(x4)
        rows = []
        for mu in range(4):
            leg = sum((e[a, mu] * GAMMA[a] for a in range(4)), 0 * GAMMA[0])
            value = H * (leg ^ GAMMA[0])
            if torsion:
                value = value + (KAPPA / 2) * (leg * S).restrict_grade(2)
            rows.append(value.cast(al.Bivector).kernel)
        return jnp.stack(rows)
