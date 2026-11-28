"""Phase 0 of docs/PLAN_DIRAC_SOLITON.md: the Einstein-Dirac solitons of Finster, Smoller and Yau (`fsy_solution`) in
Lasenby, Doran and Gull's oracle.

  * The Doran-Lasenby map from FSY's Dirac representation to the STA is exact: gamma_mu acts as gamma_mu psi gamma_0, i as
    right multiplication by I sigma_3, and bar(Psi) Psi is <psi ~psi>.
  * Shooting reproduces FSY's ground states: no node, decaying, bound after their scaling (omega~ < m~), and with
    ADM mass below the rest mass 2 m~ at small coupling (their figures 7 and 10).
  * The singlet's total spin vanishes (FSY's sum_a bar(Psi_a) gamma^5 G^p Psi_a = 0, section 4), while each fermion's
    does not. So LDG's contorsion is zero, and FSY's torsion-free solution is a GTG solution: with kappa = 8 pi and the
    Levi-Civita rotation field, LDG's spinor, h and Omega equations (two fermions, their Dirac densities summed) vanish
    to the Chebyshev fit's floor. A wrong kappa, a wrong mass, or one fermion alone fails them.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al
import fsy_solution as fs
import gtg_dirac_reference as ref

STRONG = fs.Soliton(alpha1=0.1, low=1.5, high=1.55)                 # A falls to 0.61, T to 0.52
WEAK = fs.Soliton(alpha1=0.02, low=1.07, high=1.08, inner=1.0, outer=20.0)
EVENTS = [[0.3, 1.2, -0.8, 0.9], [-0.7, 0.4, 2.1, -1.5], [1.1, -0.5, -0.3, 0.6]]     # r = 1.7, 2.6, 0.84


def size(m):
    return float(np.abs(np.asarray(m.kernel)).max())


def test_dirac_representation_map():
    rng = np.random.default_rng(0)
    column = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = fs.column_to_sta(jnp.asarray(column))
    for mu, gamma in enumerate(fs.dirac_matrices()):
        assert size(fs.column_to_sta(jnp.asarray(gamma @ column)) - fs.GAMMA[mu] * psi * fs.GAMMA[0]) < 1e-15
    assert size(fs.column_to_sta(jnp.asarray(1j * column)) - psi * fs.I_STA * fs.SIGMA[3]) < 1e-15
    bar = (np.conj(column) @ fs.dirac_matrices()[0] @ column).real
    assert abs(bar - ref.scalar(psi * ~psi)) < 1e-14


@pytest.mark.parametrize("soliton", [WEAK, STRONG])
def test_shooting_finds_fsy_ground_states(soliton):
    solution = soliton.solution
    radii = np.linspace(1e-4, solution.t[-1], 20000)
    alpha, beta, A, T = solution.sol(radii)
    norm = np.abs(alpha) + np.abs(beta)
    peak = int(np.argmax(np.diff(norm) < 0))                                               # the first maximum
    cut = peak + int(np.argmin(norm[peak:]))                                               # before the divergent tail
    assert norm[cut] < 1e-3 * norm[peak]
    assert np.all(alpha[1:cut] > 0)                                                        # no node: the ground state
    assert np.all(A[:cut] > 0) and np.all(np.diff(T[:cut]) < 0)                            # T decreasing (FSY section 7)
    R = radii[cut]
    integrand = lambda r: (lambda z: (z[0] ** 2 + z[1] ** 2) * z[3] / np.sqrt(z[2]))(solution.sol(r))  # noqa: E731
    lam = np.sqrt(4 * np.pi * quad(integrand, 1e-5, R, limit=500)[0])                       # FSY section 6
    tau = T[cut]
    mass, omega, adm = lam * soliton.mass, lam * soliton.omega * tau, R / 2 * (1 - A[cut]) / lam
    assert omega < mass                                                                    # bound
    assert adm < 2 * mass                                                                  # energy gained (FSY fig. 10)
    assert soliton.fit_residual() < 1e-9


@pytest.mark.parametrize("x4", EVENTS[:2])
def test_singlet_spin_vanishes(x4):
    spins = [ref.spin_trivector(lambda y, k=k: STRONG.spinors(y)[k], jnp.asarray(x4)) for k in range(2)]
    assert min(size(s) for s in spins) > 1e-4
    assert size(spins[0] + spins[1]) < 1e-15


def test_traceable_rotation_field_is_the_closed_form():
    x4 = jnp.asarray(EVENTS[0])
    closed = ref.rotation_gauge(STRONG.coframe, x4)
    traced = np.asarray(ref.rotation_kernels(STRONG.coframe)(x4))
    assert max(size(c) for c in closed) > 0.05
    assert max(np.abs(np.asarray(closed[mu].cast(al.Bivector).kernel) - traced[mu]).max() for mu in range(4)) < 1e-15


@pytest.mark.parametrize("x4", EVENTS)
def test_ldg_field_equations_vanish(x4):
    spinor, coframe, connection = ref.field_equations(STRONG.coframe, ref.rotation_kernels(STRONG.coframe),
                                                      STRONG.spinors, x4, STRONG.mass, fs.KAPPA, 0.0)
    assert np.abs(spinor).max() < 1e-10
    assert np.abs(coframe).max() < 1e-9
    assert np.abs(connection).max() < 1e-14


def test_wrong_coupling_mass_or_one_fermion_fails():
    x4 = EVENTS[0]
    rotation = ref.rotation_kernels(STRONG.coframe)
    _, coframe, _ = ref.field_equations(STRONG.coframe, rotation, STRONG.spinors, x4, STRONG.mass, fs.KAPPA / 2, 0.0)
    assert np.abs(coframe).max() > 0.05
    spinor, _, _ = ref.field_equations(STRONG.coframe, rotation, STRONG.spinors, x4, 1.05, fs.KAPPA, 0.0)
    assert np.abs(spinor).max() > 1e-3
    one = lambda y: STRONG.spinors(y)[0]                                                  # noqa: E731
    _, coframe, connection = ref.field_equations(STRONG.coframe, rotation, one, x4, STRONG.mass, fs.KAPPA, 0.0)
    assert np.abs(coframe).max() > 0.05 and np.abs(connection).max() > 1e-3
