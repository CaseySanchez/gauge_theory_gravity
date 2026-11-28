# Plan: a self-gravitating Dirac solution in the CSTA

Goal: one exact solution of the full coupled system: gravity, spin–torsion and the Dirac field, all at once. So far
(docs/PLAN_DIRAC.md) the equations are checked pointwise on given backgrounds; no configuration solves all of them.
The target is the CSTA with the full twistor transport. GTG's Lorentz coupling is the reference that ties it to a
known solution.

## The known solution (reference)

Challinor, Lasenby, Doran and Gull, *Massive, non-ghost solutions for the Dirac field coupled self-consistently to
gravity*, Gen. Rel. Grav. 29 (1997) 1527 ("CLDG"), extended with Λ in gr-qc/0311016 §2.4 ("thesis"). Spatially flat
FRW with a homogeneous Dirac field; homogeneity of the observables forces k = 0. Units G = 1; the thesis's κ² is LDG's
κ = 8π.

- Newtonian gauge, thesis (2.35), (2.37): `h̄(b) = b + rH(t) b·e_r e_t`, `ω(b) = H b∧e_t + (κ/2) b·S`. The position
  gauge field is identity plus a dyad, like the repo's Painlevé–Gullstrand and de Sitter coframes.
- Spinor, thesis (2.48), (2.82): `ψ = √ρ e^{Iβ/2} e^{−Iσ₃χ(t)} L`, with `cos β = ±1` (particle or antiparticle), L a
  constant boost across γ₃, and `χ̇ = m cos β + 3πρ`. The spin is along γ₃, `S = ½ψIγ₃ψ̃`.
- Equations, thesis (2.41)–(2.45), (2.52)–(2.54): `ρ̇ = −3ρH` and `3H² − 12π²ρ² − 8πmρ cos β − Λ = 0`, so
  `ẏ² = 1 ± 4my + 3Λy²` with `y = 1/(6πρ)`. The `12π²ρ²` is the spin–torsion term; it dominates while the Compton
  wavelength exceeds the Hubble radius.
- Closed forms, thesis (2.56)–(2.65), (2.77)–(2.81), with `α = √(3Λ)/2`. Region (i)/(ii):
  `ρ = α²/(6π sinh αt (α cosh αt ± m sinh αt))`, `H = α(α cosh 2αt ± m sinh 2αt)/(3 sinh αt (α cosh αt ± m sinh αt))`.
  Region (iii) (antiparticle, nonsingular bounce): (2.58), (2.59). Extremal `4m² = 3Λ`: (2.60)–(2.65). χ(t) in
  closed form for each.
- Stress-energy, thesis (2.66): `T(e_t) = ρ(2πρ ± m)e_t`, `T(γ₃) = 0`, and an antisymmetric part in the γ₁γ₂ plane.
  The solution is isotropic in its metric but not in its gauge fields.

## Phase 0: the solution in the oracle (GTG, STA)

- Transcribe h̄, ω (with the contorsion), ψ(t), ρ(t), H(t), χ(t) for regions (i), (ii) and (iii), and for the
  extremal case, into `tests/gtg_dirac_reference.py` style: test-only, in the thesis's own formulas.
- Check against the oracle's own equations at sample events (t, x): the Dirac residual (3.62), the Ω equation (4.37)
  `D∧h̄(a) = κS·h̄(a)`, and the h equation `G(a) − Λa = κT(a)`, with T from the oracle's `coframe_derivative`. This
  also pins the signs (particle/antiparticle, S, β) before anything touches the CSTA.
- Risk: the thesis's typography (the PDF extraction loses minus signs and braces). Settle every sign by the oracle,
  not by reading.

## Phase 1: the reference solution in the CSTA (Lorentz coupling)

- Build the CSTA connection: the Newtonian-gauge coframe (closed-form frame, as `cartan` geometries carry), its
  Levi-Civita connection, the LDG contorsion `½κ g_μ·S` from `dirac.spin_trivector` of `Z = from_sta(ψ)`, and
  `f = e/β`.
- Action `S₁ + βS₂ + γS₃ + 4κ|det e|L_D` with `Λ = 6(γ − 1/β)` and `which = "lorentz"`. Check that all 60 connection
  rows and the 8 spinor rows vanish at sample events, in regions (i), (ii) and (iii) and in the extremal case.
- This is the first configuration in the repo that solves the coupled gravity–Dirac system. It also checks the
  Phase 5 coupling normalization (`4κ`) and the Λ identification against an independent, self-consistent solution.

## Phase 2: the target, full transport

Under the full transport the system changes in three ways (PLAN_DIRAC phases 0–6):
- a compensator mass, `m → m − √2(1 + ¼ tr F)`;
- the Dirac-shifted special-conformal field, `F = (1/β)(1 + κ tr C/3) − (κ/β)Cᵀ`, `tr C = √2σ`;
- the tensor bilinear `T_ab` coupling through the antisymmetric part of F.

Steps:
1. **Ansatz.** Keep the symmetry: Newtonian gauge, `ψ(t)` of the same form with unknown ρ, β, χ, and spin along γ₃.
   Compute `C(t)` for it and check what it does to homogeneity: whether `T_ab` for this ψ has a fixed frame
   orientation (spin along γ₃ suggests the γ₁γ₂ plane) or rotates with χ.
2. **Reduce to ODEs** with the pointwise machinery. The rows at a sample event are affine in the time derivatives of
   the unknowns (ρ̇, β̇, χ̇, Ḣ). Read the coefficients by evaluating the rows on a few trial values, and check that the
   reduction is independent of the event (homogeneity) and of the radius (the gauge).
3. **Compare with the reference.** At `κ → 0` in the f-shift and with the mass shift absorbed, the ODEs must reduce to
   the thesis's. Find the extra terms in closed form: a `(ψ̄ψ)²` contact term from the compensator mass through
   tr F, and `T_ab T^ab` from the antisymmetric part.
4. **Solve.** Closed form if the modified constraint stays quadratic in y (the thesis's structure suggests it might);
   otherwise integrate the ODEs and check that all rows vanish along the numerical solution to integrator tolerance.
5. **The scale question.** The compensator mass `√2(1 + 1/β)` is measured in units of the length that the translation
   sector carries. It vanishes at `β = −1` (then `Λ = 6(γ + 1)`). Record what the solution says about β: whether a
   physical fermion mass forces `β ≈ −1`, or the shift is absorbed into m.

## Phase 3 (stretch): a localized solution

The static, spherically symmetric Einstein–Dirac solitons of Finster, Smoller and Yau need two fermions in a singlet
for spherical symmetry, and a numerical boundary-value solve. They test the same equations with gradients, where
the CSTA's `e∧Dφ` terms (PLAN_DIRAC phase 6) matter. Planned in docs/PLAN_DIRAC_SOLITON.md.

## Phase 4: documentation

RESULTS.md, DEEP_DIVE §5.9, the LaTeX formulation (§13), README: the first self-gravitating solution, and what the full
transport changes in it.

## Risks

- **Transcription signs.** Mitigated by Phase 0, where everything is checked inside the oracle first.
- **Torsion sizes.** The torsion is large near the singularities (`κS ~ 4πρ`). Sample events away from `t = 0` and
  from the region (ii) recollapse; region (iii) is singularity-free.
- **Homogeneity under the full transport.** If `T_ab` rotates with χ, the homogeneous ansatz may not close, and the
  full-transport solution would need a less symmetric ansatz.
- **Strictness.** Any ODE integration stays in tests or examples, not in the library's GA code.

## Status

- Phase 0: done (`tests/cldg_solution.py`; `gtg_dirac_reference.action_density`, `field_equations`;
  `tests/test_cldg_solution.py`, 10 tests).
  - The closed forms satisfy `ρ̇ = −3ρH`, `χ̇ = m cos β + 3πρ` and the constraint (1e-12), for particle and antiparticle
    (regions (i) and (ii)). The thesis's χ came through the extraction garbled, so it is derived here:
    `χ = ±mt − ½ ln(α coth αt ± m)`.
  - The Newtonian-gauge rotation gauge field `H b∧γ₀` is LDG's closed form (4.53) for that h̄ (exact).
  - The oracle's field equations are autodiff of LDG's action density in ψ, e and Ω_μ. With the contorsion
    `(κ/2) b·S` all three vanish (1e-12), with and without the boost L. Without the contorsion, or with its sign
    flipped, the h- and Ω-equations fail.
  - The sign of Λ: with this R (de Sitter `R = −4Λ`), a repulsive Λ (`3H² = Λ` for de Sitter) enters the density as
    `+Λ det(h)⁻¹`. LDG print `−Λ det(h)⁻¹` (4.36); the solution satisfies the h-equation only with `+Λ` (with `−Λ` it
    misses by 2Λ). Phase 1 has to carry this into `Λ = 6(γ − 1/β)`.
  - Not yet transcribed: region (iii) (the antiparticle bounce) and the extremal case `4m² = 3Λ`.
- Phase 1: done (`Cosmology.geometry`, `csta_spinor`, `csta_connection` in `tests/cldg_solution.py`;
  `tests/test_cldg_csta.py`, 5 tests, about 5 minutes).
  - The Newtonian-gauge coframe is `e^0 = dt`, `e^i = dx^i − H x^i dt`: identity plus the dyad `−H x (t & Direction)`,
    inverted in closed form like Painlevé–Gullstrand, and equal to the oracle's.
  - With `f = e/β`, `γ = Λ/6 + 1/β`, κ = 8π, the Lorentz coupling, and Levi-Civita plus LDG's contorsion, all 60
    connection rows and the 8 spinor rows of `S₁ + βS₂ + γS₃ + 4κ|det e|L_D` vanish (1e-13; 1e-15 seen). This holds
    for the particle (β = 4), the antiparticle (β = 2) and the boosted particle (β = −1), at different events. The
    matter rows are O(1–15) in the M and P sectors, and zero in K and D (the Lorentz coupling does not source f).
  - Without the contorsion the M and P rows fail by O(5) and the spinor rows by O(0.4).
  - So both normalizations hold against an independent self-consistent solution: the `4κ` coupling, and
    `Λ = 6(γ − 1/β)` with Λ repulsive.
  - Region (iii) and the extremal case are still not transcribed (Phase 0), so they are not checked here.
- Phase 2: done (`Cosmology(coupling=beta)`, `shifted_special` in `tests/cldg_solution.py`;
  `tests/test_cldg_full_transport.py`, 5 tests; the reduced-equation test covers the full-transport closed forms).
  - **Homogeneity holds.** For this spinor `C/ρ` is constant: `(√2/4)σ` on the diagonal plus a tensor part in the
    γ₀γ₃ plane, not γ₁γ₂. χ rotates γ₁γ₂, so it leaves C alone. With the boost, part of C rotates with χ but stays a
    function of t alone. σ = ±ρ and the pseudoscalar density vanishes.
  - **Spinor equation.** At the shifted f the full transport adds a real multiple of Z: the compensator mass
    `√2(1 + tr F/4)` and the tensor coupling `−F₀₃/√2`. Together the mass is `m − √2(1 + 1/β) + κσ/(12β)`, for both
    branches.
  - **Reduction.** On a local Taylor ansatz (H, ρ, χ and their first derivatives at t₀, with f shifted exactly), only
    the P_t row along `E^μ₀` survives: the Friedmann constraint. LDG's contorsion still solves the Lorentz rows, the
    shifted f solves the K rows, and the D rows vanish. The residual equals
    `4(3H² − Λ − 12π²aρ² − 8πm₀ρ cos β)` at 10 shifted (H, ρ) states for both branches (1e-14), with
    `a = 1 + 2/(9β)` and `m₀ = m − √2(1 + 1/β)`.
  - **Why.** Eliminating f from `12 tr F − 2β((tr F)² − F:F) + 4κ C·F` leaves 24/β (inside `Λ = 6(γ − 1/β)`),
    `4κ√2σ/β` (inside the mass) and the contact term `(κ/2β)((tr C)²/3 − C:C) = −κσ²/(24β)`. As a potential
    `V = m₀σ + κσ²/(24β)`, its energy `V` gives the constraint and `V'` gives the mass.
  - **Closed form.** The constraint stays quadratic in `y = 1/(6πρ)`: `ẏ² = a + 4m₀ cos β y + 3Λy²`. So the thesis's
    formulas carry over with m₀ and `√a α` in place of m and α, and `χ = ±m₀t − (√a/2) ln(√a α coth αt ± m₀)`. All 60
    connection rows and 8 spinor rows vanish (1e-12): particle β = 4, antiparticle β = 2, boosted particle β = −1. The
    Lorentz-coupled solution fails the full-transport P and spinor rows.
  - **The scale question.** The rows see only m₀, so the compensator mass is absorbed into the bare m, and a physical
    mass does not force β ≈ −1. What β does fix is a, the strength of the effective spin–torsion term: a = 7/9 at
    β = −1, and the term cancels at β = −2/9. A real `√a` needs β > 0 or β < −2/9. For a ≤ 0 the closed form changes
    (not covered).
- Phase 3 (stretch, the localized Finster–Smoller–Yau solution): planned in docs/PLAN_DIRAC_SOLITON.md.
- Phase 4: done. RESULTS.md has a new section, "A self-gravitating Dirac solution", and DEEP_DIVE §5.9, §10 and §11, the
  README and the LaTeX formulation (§13.4, open questions, the CLDG and Leong citations) are updated. Counts: 342 fast
  tests and 21 slow.
