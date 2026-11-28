# Plan: a localized self-gravitating Dirac solution in the CSTA

Phase 3 of docs/PLAN_DIRAC_SOLUTION.md. The cosmology (Phases 0–2 there) solves every CSTA equation, but it is
homogeneous: nothing depends on position, so gradients of the special-conformal field f never enter. A static,
spherically symmetric soliton has gradients everywhere. The target is again the full twistor transport, with GTG's
Lorentz coupling as the reference that ties it to a known solution.

## The known solution (reference)

Finster, Smoller and Yau, *Particlelike solutions of the Einstein–Dirac equations*, Phys. Rev. D 59 (1999) 104020,
arXiv:gr-qc/9801079 ("FSY"). Einstein gravity with no torsion, and two fermions in the singlet state (j = 1/2,
opposite angular momenta), the smallest number that keeps the total stress-energy spherically symmetric.

- Metric `ds² = dt²/T(r)² − dr²/A(r) − r² dΩ²`. Spinors `e^{−iωt} T^{1/2} r^{-1} (α(r), β(r))` in FSY's two-component
  radial form.
- The ODEs, as recalled. Phase 0 transcribes them from the paper and settles every sign in the oracle:
  - `√A α' = α/r − (ωT + m) β`
  - `√A β' = (ωT − m) α − β/r`
  - `r A' = 1 − A − 16πω T² (α² + β²)`
  - `2rA T'/T = A − 1 − 16πω T² (α² + β²) + 32π T αβ/r + 16π m T (α² − β²)`
  - normalization `4π ∫ (α² + β²) T/√A dr = 1`, with ω the eigenvalue
- Boundary conditions: regular at r = 0 (`α ~ α₁ r`, `β ~ r²`) and asymptotically flat (`A, T → 1`). The solutions
  are found by shooting in α₁ and ω. They exist for ω below m in a band of masses near the Planck scale. FSY tabulate
  ω, the ADM mass and the binding energy.

Two facts make the pointwise checks easier than the eigenvalue problem:
- **Local solutions suffice.** The field equations are local. Any solution of the ODEs on an interval satisfies them
  there, normalizable or not. Shooting is needed only to name *the* soliton, not to check the equations.
- **No torsion is expected.** In each fermion's state the spin density is not spherically symmetric. Summed over the
  singlet, it must be built from r̂ and γ₀ alone. A radial axial vector and an axial time component both change sign
  under parity, and the FSY states have definite parity. So the total axial current vanishes, so does the spin
  trivector S, and LDG's contorsion is zero. If this holds, FSY's torsion-free solution is a GTG solution as it
  stands. If it fails, see Risks.

## Phase 0: the solution in the oracle (GTG, STA)

- **Transcription.** Take the ODEs, boundary conditions and one tabulated solution from FSY (`tests/fsy_solution.py`,
  test-only). Integrate with scipy (`solve_ivp`, DOP853, rtol 1e-13) and shoot for α₁ and ω. Reproduce one table entry
  (ω, ADM mass) to the paper's printed digits.
- **jax-traceable fields.** On an interval [r₁, r₂] away from the origin (r₁ ≈ 0.3 in FSY units), fit α, β, A and T
  with Chebyshev series. They are smooth, so the fit error is near machine precision. The oracle's autodiff then sees
  ordinary functions of x4. The fit residual sets the tolerance for every later check, about 1e-9 rather than 1e-12;
  record the measured floor.
- **Cartesian STA form.**
  - The coframe is diagonal in Schwarzschild-like coordinates. In the repo's Cartesian coordinates it is
    `e^0 = dt/T` plus `e^i = dx^i + (1/√A − 1) r̂^i (r̂·dx)`: identity plus a radial dyad. The dyad is not nilpotent,
    but the inverse is still closed form: `E = T ∂_t` plus the identity plus `(√A − 1) r̂ r̂·`.
  - The spinors: the j = 1/2 states in the STA, after Doran and Lasenby's treatment of the Dirac hydrogen atom.
    Something like `ψ₁ = (u(r) + v(r) I σ_r ...) e^{−Iσ₃ωt}`, with ψ₂ the opposite angular momentum, e.g. ψ₁ times a
    fixed rotor on the right.
  - Settle the exact form, and its map to FSY's (α, β), by the oracle: each spinor must satisfy LDG's Dirac
    equation (3.62).
- **The spin.** Check that the summed spin trivector S₁ + S₂ vanishes at sample events, and that each fermion's
  does not.
- **Field equations.** With two fermions the oracle's action density is `½R − κ(L₁ + L₂) + Λ` with Λ = 0. Check that
  the spinor, h and Ω equations vanish at several events (different r, and rotated directions for spherical symmetry)
  to the fit floor. As a negative check, a slightly wrong ω or m must make them fail.

## Phase 1: the reference in the CSTA (Lorentz coupling)

- Build the CSTA connection: the radial-dyad coframe as a `cartan.Geometry`, its Levi-Civita connection, and no
  contorsion (S = 0). Take `f = e/β`, `γ = 1/β` (Λ = 0), Z₁, Z₂ = `from_sta(ψ₁, ψ₂)`, and `which = "lorentz"`.
- The matter terms are sums over the two fermions. `dirac.euler_lagrange_connection` and `euler_lagrange_spinor` are
  linear in the density, so they are called once per fermion and added, in a test-only wrapper.
- Check that all 60 connection rows and both sets of 8 spinor rows vanish to the fit floor, at events across
  [r₁, r₂] and in several directions. The matter rows must be O(1) in the M and P sectors, and the M rows must vanish
  with no contorsion: S = 0 seen from the CSTA side.
- This is the first localized self-gravitating solution in the repo, and the first with gradients of the matter
  fields in every direction.

## Phase 2: the target, full transport

What differs from the cosmology:
- **f varies in r.** The total special-conformal current `C = C₁ + C₂` is static and spherically symmetric:
  - a trace `√2σ(r)`, from the scalar density summed;
  - a tensor part. Parity allows the electric, radial `0r` component (a polar vector) and forbids the magnetic one. The
    `(1 − γ₅)` structure of C mixes the two, so the surviving pattern is measured, not assumed.
  The K rows are algebraic, so `F = (1/β)(1 + κ trC/3) − (κ/β)Cᵀ` still solves them pointwise. But F now depends on r.
- **Gradients.** Whether `D f` enters the M and P rows at α = 0 is open. In the cosmology it could not. PLAN_DIRAC
  Phase 6 found `e ∧ D f` only through the MacDowell–Mansouri term, which this action leaves out, but S₁'s own
  dependence on the K sector off the homogeneous state has not been isolated. Settle this first.

Steps:
1. **C and the spin.** Compute C₁ + C₂ and S₁ + S₂ on the Phase 0 solution. Record the surviving components of C, and
   whether the full transport induces torsion: the M rows with LDG's contorsion, here zero.
2. **Reduce to ODEs.** At a radius r the rows are affine in the first derivatives `(α', β', A', T')`, given the values
   `(α, β, A, T)` and ω. f is determined algebraically, so its r-derivative is a function of these. Read the
   coefficients by evaluating the rows on trial values of a local Taylor ansatz, as in PLAN_DIRAC_SOLUTION Phase 2.
   Check that the reduction does not depend on direction (spherical symmetry) and is consistent across radii.
3. **Identify the corrections in closed form.** Expected:
   - the local mass `m → m − √2(1 + ¼ tr F(r))` and the tensor coupling, as in the cosmology but now with r-dependent
     F;
   - the contact term `(κ/2β)((trC)²/3 − C:C)` in the Einstein equations, here with the 0r tensor component it did
     not have in the cosmology;
   - any gradient term from step 0.
   This is the test of the contact-term formula off the homogeneous state, which RESULTS.md lists as open.
4. **Solve.** Integrate the modified system with the same shooting and Chebyshev pipeline. Check that all rows
   vanish along it to the fit floor, and that FSY's solution fails them. Compare ω, the ADM mass and the binding energy
   with FSY's as functions of β.
5. **The scale question again.** FSY's solitons have m near the Planck mass, the same order as the compensator mass
   `√2(1 + 1/β)` in these units. So m₀ and the contact terms are O(1) changes, not perturbations. Record:
   - for which β the soliton family survives;
   - whether β = −1 (zero compensator mass at f = e/β) is special here, as it was not in the cosmology, where only m₀
     appeared;
   - whether the band of masses that allows solitons moves.

## Phase 3: documentation

RESULTS.md (the self-gravitating section), DEEP_DIVE §5.9 and §11, the LaTeX formulation (§13.4 and open questions),
README: the soliton under both couplings, and what the full transport's gradients and contact terms change.

## Risks

- **The STA form of the singlet.** FSY's two-component radial reduction hides the spinor's angular structure, and the
  Cartesian STA form has to be rebuilt. Mitigated by Phase 0: each spinor must solve LDG's Dirac equation in the
  oracle before anything else.
- **Nonzero spin.** If S₁ + S₂ does not vanish (the parity argument fails for the actual states), GTG's torsion
  modifies FSY. In ECSK form, the axial contact term `∝ κ A²` enters. Phase 1 would then need its own reduction and
  solve, like Phase 2, before the full transport.
- **Precision.** Checks are at the fit floor, about 1e-9, not 1e-12. Negative checks (wrong ω, no contact term) must
  fail by many orders more than that.
- **The origin.** The coordinates are singular at r = 0 and the Cartesian dyad has r̂ in it. Sample only r ≥ r₁.
- **Cost.** Each event costs about 1–3 minutes, and two fermions double the matter part. The reduction in Phase 2
  needs about 10 evaluations per radius, so mark the larger checks slow.
- **Existence under the full transport.** The modified shooting problem may have no normalizable solution for some
  β. That is a result to record, not a failure. The local checks hold on any local solution.
- **Strictness.** scipy, the shooting and the Chebyshev fits stay in tests, not in the library's GA code.

## Status

- Phase 0: done (`tests/fsy_solution.py`; `gtg_dirac_reference.rotation_kernels`, and `action_density` and
  `field_equations` for several fermions; `tests/test_fsy_solution.py`, 10 tests, about 2 minutes).
  - **The paper.** FSY's (5.4)–(5.7) are the ODEs recalled above. It prints no tables, only figures. Its section 4
    states the key fact directly: for the singlet, `Σ_a Ψ̄_a γ⁵ G^p Ψ_a = 0`.
  - **Shooting.** With T₀ = 1 and m = 1 (FSY section 6), the ground states are at ω₀ = 1.0711161774331 for α₁ = 0.02
    and ω₀ = 1.5134807004016 for α₁ = 0.1. Both have no node, decay and have T decreasing. After FSY's scaling:
    - α₁ = 0.02: m̃ = 0.534, ω̃ = 0.502, ρ̃ − 2m̃ = −0.021;
    - α₁ = 0.1: m̃ = 0.588, ω̃ = 0.459, ρ̃ − 2m̃ = −0.031.
    So both are bound and gain energy, as in FSY's figures 7 and 10. α₁ = 0.1 is the strong-field case (A down to
    0.61, T to 0.52). The first sign flip in ω is the ground state. Later flips are excited states, and flips with
    ωτ > m are continuum artifacts.
  - **The fit floor.** Chebyshev degree 90 on [0.4, 6] matches the ODEs to 4e-11.
  - **The Dirac-representation map.** It is exact: γ_μ acts as `γ_μ ψ γ₀`, i as `·Iσ₃`, and Ψ̄Ψ is `⟨ψψ̃⟩`.
  - **No torsion.** Each fermion's spin trivector is about 1.6e-3, and their sum is 2e-19. The parity argument holds.
  - **The rotation field.** The traceable rotation field equals LDG's closed form (4.49, 4.53) to 1e-17.
  - **LDG's equations.** With κ = 8π, Levi-Civita and no contorsion, the spinor, h and Ω equations of the two-fermion
    action vanish at three events (r = 0.84, 1.7, 2.6): spinor 1e-12, h 1.2e-11 (the fit floor), Ω 1e-16. So FSY's
    torsion-free solution is a GTG solution as it stands, and their conventions and normalization match LDG's.
    κ = 4π, m = 1.05, or one fermion alone fail by 0.05 to 0.5.
