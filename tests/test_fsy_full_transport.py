"""Phase 2 of docs/PLAN_DIRAC_SOLITON.md: the Einstein-Dirac soliton under the CSTA's full twistor transport, the target
coupling, where the Dirac-shifted special-conformal field varies in space.

  * The special-conformal current in spacetime-algebra bilinears: C_ab = (sqrt 2 / 4)(sigma eta_ab + K_ab), with
    K_ab = <(gamma_a ^ gamma_b) I S>, S = psi I sigma_3 ~psi: the dual of the spin bivector (exact).
  * On the singlet the total current has the trace sqrt 2 sigma, sigma = 2 T (alpha^2 - beta^2) / r^2, and a magnetic
    tensor part orthogonal to the radius, K = 4 T alpha beta / r^2.
  * f enters the action algebraically (S1 = -2 sqrt g R + 12 sqrt g tr F, S2 and the Dirac coupling), so eliminating it
    gives, everywhere, GTG with the mass m0 = m - sqrt 2 (1 + 1/beta) and the contact term
    (kappa / 2 beta)(sigma^2 / 6 + K_ab K^ab / 8) in the matter Lagrangian. On the singlet this modifies FSY's equations
    (`fsy_solution.rhs`): the mass m0 - kappa sigma / (6 beta), the radial term 1/r - kappa K / (4 beta), and the contact
    density in Einstein's equations.
  * A local solution of the modified equations solves LDG's equations with the contact term (oracle), and not without
    it. The magnetic K fixes the contraction that the cosmology's electric K could not.
  * The soliton survives: shooting finds a nodeless, bound ground state (beta = 4, m0 = 1), which solves the effective
    oracle and every row of the CSTA under the full transport, with the bare mass and f shifted by the total C (slow).
    The contact term moves FSY's spectrum by a few per cent at these couplings.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, dirac as dr, first_order as fo
import fsy_solution as fs
import gtg_dirac_reference as ref
from test_fsy_solution import EVENTS, STRONG

BETA = 4.0
LOCAL = fs.Soliton(alpha1=0.1, coupling=BETA, given_omega=1.2, inner=0.4, outer=2.0)   # diverges at r = 2.37
# The ground state with m0 = 1, bare m = 1 + sqrt 2 (1 + 1/beta) = 2.768 (the equations scale like FSY's with m0).
BOUND = fs.Soliton(alpha1=0.1, low=1.47, high=1.52, mass=1 + np.sqrt(2) * (1 + 1 / BETA), coupling=BETA)
G = al.SPACETIME
I_STA = G[0] * G[1] * G[2] * G[3]
ETA = np.diag(al.ETA4)


def test_current_in_spacetime_algebra_bilinears():
    rng = np.random.default_rng(3)
    blades = [1 + 0 * G[0] * G[0], G[0] * G[1], G[0] * G[2], G[0] * G[3], G[1] * G[2], G[1] * G[3], G[2] * G[3], I_STA]
    for _ in range(3):
        k = rng.normal(size=8)
        psi = sum((k[i] * blades[i] for i in range(8)), 0 * blades[0]).cast(al.Even)
        current = np.asarray(dr.special_conformal_current(dr.from_sta(psi)).kernel).reshape(4, 4) * ETA[None, :]
        sigma = ref.scalar(psi * ~psi)
        spin = (psi * I_STA * G[3] * G[0] * ~psi).restrict_grade(2)
        for a in range(4):
            for b in range(4):
                K = ref.scalar((G[a] ^ G[b]) * I_STA * spin)
                assert abs(current[a, b] - np.sqrt(2) / 4 * (sigma * ETA[a] * (a == b) + K)) < 1e-14


@pytest.mark.parametrize("x4", [[0.3, 1.2, -0.8, 0.9], [0.0, 0.0, 0.0, 1.7]])
def test_singlet_current_is_scalar_plus_magnetic(x4):
    x4 = jnp.asarray(x4)
    r = float(jnp.linalg.norm(x4[1:]))
    alpha, beta, _, T = [float(v) for v in STRONG.profile(r)]
    current = sum(np.asarray(dr.special_conformal_current(dr.from_sta(p)).kernel).reshape(4, 4)
                  for p in STRONG.spinor_multivectors(x4))
    sigma, K = 2 * T * (alpha ** 2 - beta ** 2) / r ** 2, 4 * T * alpha * beta / r ** 2
    n = np.asarray(x4[1:]) / r
    expected = np.sqrt(2) / 4 * sigma * np.eye(4)
    expected[1:, 1:] += np.sqrt(2) / 4 * K * np.array([[0, n[2], -n[1]], [-n[2], 0, n[0]], [n[1], -n[0], 0]])
    assert abs(K) > 1e-4
    assert np.abs(current - expected).max() < 1e-15


def test_modified_equations_solve_the_effective_oracle():
    assert LOCAL.fit_residual() < 1e-9
    x4 = jnp.asarray(EVENTS[0])
    rotation = ref.rotation_kernels(LOCAL.coframe)
    spinor, coframe, connection = ref.field_equations(LOCAL.coframe, rotation, LOCAL.spinors, x4, LOCAL.effective_mass,
                                                      fs.KAPPA, 0.0, contact=fs.KAPPA / (2 * BETA))
    assert np.abs(spinor).max() < 1e-9 and np.abs(coframe).max() < 1e-8 and np.abs(connection).max() < 1e-14
    spinor, coframe, _ = ref.field_equations(LOCAL.coframe, rotation, LOCAL.spinors, x4, LOCAL.effective_mass,
                                             fs.KAPPA, 0.0)
    assert np.abs(spinor).max() > 0.1 and np.abs(coframe).max() > 0.01


def test_full_transport_ground_state():
    from test_fsy_solution import test_shooting_finds_fsy_ground_states
    test_shooting_finds_fsy_ground_states(BOUND)                      # nodeless, decaying, bound, fit floor
    assert abs(BOUND.omega - 1.4976695441) < 1e-9
    assert abs(BOUND.omega - STRONG.omega) > 1e-3                     # the contact term moves FSY's omega (1.51348)
    x4 = jnp.asarray(EVENTS[0])
    spinor, coframe, connection = ref.field_equations(BOUND.coframe, ref.rotation_kernels(BOUND.coframe),
                                                      BOUND.spinors, x4, BOUND.effective_mass, fs.KAPPA, 0.0,
                                                      contact=fs.KAPPA / (2 * BETA))
    assert np.abs(spinor).max() < 1e-10 and np.abs(coframe).max() < 1e-9 and np.abs(connection).max() < 1e-14


@pytest.mark.slow
def test_ground_state_solves_every_csta_row_under_the_full_transport():
    x4 = EVENTS[0]
    LOCAL = BOUND                                                      # noqa: N806
    omega = LOCAL.csta_connection(LOCAL.shifted_special(BETA))
    gravity = fo.euler_lagrange_connection(omega, x4)
    total = gravity[0] + BETA * gravity[1] + gravity[2] / BETA
    for k in range(2):
        matter = 4 * fs.KAPPA * dr.euler_lagrange_connection(omega, LOCAL.csta_spinor(k), x4, mass=LOCAL.mass,
                                                             which="full")
        assert np.abs(matter[al.SECTORS["K"]]).max() > 1e-3
        total = total + matter
        spinor = dr.euler_lagrange_spinor(omega, LOCAL.csta_spinor(k), x4, mass=LOCAL.mass, which="full")
        assert np.abs(spinor).max() < 1e-10
    assert np.abs(total).max() < 1e-8
