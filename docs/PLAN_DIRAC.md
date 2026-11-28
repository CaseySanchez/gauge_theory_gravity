# Plan: the GTG Dirac action in CSTA first-order form

Goal: recover the dynamics of Gauge Theory Gravity with Dirac matter, as Lasenby, Doran and Gull give it in the STA
([GRAVITY, GAUGE THEORIES AND GEOMETRIC ALGEBRA.pdf](GRAVITY,%20GAUGE%20THEORIES%20AND%20GEOMETRIC%20ALGEBRA.pdf),
gr-qc/0405033, cited below as LDG), in the first-order CSTA form of `first_order.py`. Vacuum Einstein recovery cannot
tell GTG from GR; spin–torsion with Dirac matter can.

## Target (LDG)

- Action, (3.56) with (4.14): `S = ∫ det(h)⁻¹ [ ½R − κ⟨h̄(∂_a)(a·∇ + ½Ω(a))ψIγ₃ψ̃ − mψψ̃⟩ ]`.
- Dirac equation from it, (3.62); minimally coupled form (3.63) only when `det(h) Ḋ_μ(h̄(e^μ)det(h)⁻¹) = 0`, (3.65).
- h-equation, (4.21): `G(a) = κ T(a)`, `T` from (4.20), not assumed symmetric.
- Ω-equation, (4.37): `D∧h̄(a) = κ S·h̄(a)`, spin trivector `S = ½ψIγ₃ψ̃` (4.34); connection with spin (4.41, 4.53):
  Levi-Civita plus `½κ a·S`.
- Consistency loop (4.32–4.35): the Dirac spin tensor has zero contraction, so the minimally coupled action gives the
  minimally coupled equation. LDG state this holds only for the Ricci-scalar action, not for R + R².

## Representation (checked before Phase 0)

- The Dirac spinor is a twistor field `Z ∈ Even·P` (8 real components), as in `twistor.py`.
- `Γ_a = I e_a p`, with `I` the pseudoscalar and `p = (n_o + n_inf)/√2 = (X + 𝓘)/√2`, satisfies
  `{Γ_a, Γ_b} = 2η_ab` on T (2e-16). Covariantly, `Γ(a) = I·DX(a)·(X + 𝓘)/√2`: built from the compensators.
- With it the twistor norm `h` is the Dirac bar pairing: `h(Z, Γ₀Z) > 0`, `J_a = h(Z, Γ_a Z)` timelike, and
  `½[Γ_a, Γ_b] = e_a e_b`, so the Lorentz sector acts on Z as LDG's `½Ω(a)ψ`.
- The alternative `e_a n` also satisfies the Clifford relation, but its current is indefinite (axial). Rejected.

## Phase 0: conventions and one decision (gate)

1. Bilinears: `σ = h(Z,Z)`, `J`, the axial current, `S_abc`; Fierz `J² = σ² + ω²`.
2. Explicit map `Z ↔ ψ` (STA even element): `γ^μψγ₀ ↔ Γ^μ Z`, `ψIσ₃ ↔ I Z`, and the sign map to numga's `x+y+z+t-`
   STA (`sta_embedding.py`).
3. Normalization of S₁ against LDG's `½R`, which fixes κ.
4. Decision: the connection in the Dirac operator.
   - (A) Full twistor transport. The extra sectors act algebraically: P as `Γ_a` times a chiral projector, K through
     `f` (`= e/β` on shell), D as an axial coupling `γ₅ b·Γ`. Hypothesis: on shell the P and K pieces combine into a
     parity-even mass, at least at some β, so the Dirac mass comes from the compensators.
   - (B) The covariant derivative of the stabilizer of X and 𝓘 (the Lorentz sector), built from the compensators,
     with all 60 components of Ω still varied. GTG lies inside B by construction.
   - **Decided:** A is the target theory, the Dirac field in the full CSTA. B is built first, as the reference that
     ties the construction to LDG through the oracle. The A − B difference (mass shift and contact terms, Phase 0
     findings) is then pinned exactly by tests: it is what the CSTA adds to GTG, not a deviation to be removed.

## Phase 1: `dirac.py`

- Density: volume from `DX∧DX∧DX∧DX∧X∧𝓘` (as S₃); kinetic `Re[I·h(Z, Γ(∂_a) D_a Z)]` with the frame from
  `cartan.frame_at`; mass `m·h(Z,Z)`. Later as a top form `⟨DX∧DX∧DX ∧ J(Z, DZ) ∧ X∧𝓘⟩`.
- `euler_lagrange_spinor`: units `twistor.BASIS[k]`, `I·BASIS[k]`, through `variation._euler` at order 1.
- `euler_lagrange_connection` takes the combined density S₁ + βS₂ + γS₃ + κL_D unchanged.
- All of it in numga; the STA oracle below is test-only.

## Phase 2: independent GTG oracle (test-only)

`tests/gtg_dirac_reference.py`: LDG (3.56), (3.62), (4.20), (4.27) in numga's STA, with h̄ and Ω given. Geometries:
flat, Painlevé–Gullstrand Schwarzschild, Kerr–Schild Kerr, and a random h̄, Ω (with torsion).

## Phase 3: matter equations

- Flat: the Z equation is Dirac's; a plane wave solves it with `J ∝ p`.
- Curved: the CSTA Z equation maps to LDG (3.62), extra term included; it vanishes exactly when the torsion
  contraction does (3.64).

## Phase 4: sources

- Spin: the Dirac part of the Ω equation, M sector, is `S(a) = S·a` with `S = ½ψIγ₃ψ̃` through the map.
- Stress-energy: the Dirac part of the P-sector equation is LDG's `T(a)` (4.20).
- New CSTA content: what Dirac feeds into the K and D equations (zero under B, nonzero under A). Record either way.

## Phase 5: spin–torsion

For given `(e, Z)`, Levi-Civita plus contorsion `½κ a·S` solves the combined M-sector equations; the torsion contraction
vanishes; the minimally coupled Dirac equation then holds. The first test that separates GTG from GR.

## Phase 6: consistency loop and α

Add `α·⟨F∧F∧X∧𝓘⟩` (Weyl squared). LDG predict a nonzero torsion trace in the Ω equation, so the minimally coupled
Dirac equation fails. If confirmed, Dirac matter fixes α = 0, closing an open item.

## Phase 7: documentation

RESULTS.md (claims, tests, tolerances), DEEP_DIVE §11, the LaTeX formulation (a Dirac matter section), README.

## Risks

- Complex structure: the twistor's `I` is central and acts on the left; LDG's `Iσ₃` acts on the right. Phase 0.
- Conformal weight of the Dirac field (−3/2) and Weyl rescaling of Z; the mass term breaks Weyl invariance.
- Autodiff cost: 60 + 8 unknowns; spin tests on Kerr likely `slow`.
- No exact self-gravitating Dirac solution: the checks establish the equations and their sources, not solutions.

## Status

- Phase 0: done except the decision (`dirac.py`, `tests/test_dirac_conventions.py`, 10 tests). Findings:
  - The map `Z = ψ·REFERENCE`, with `REFERENCE` the unit twistor that has `Γ₀Z = Z` and `e_y e_x Z = I Z`, is a real
    isomorphism. It carries `γ_μψγ₀ → Γ_μ Z`, `ψIσ₃ → I Z`, `⟨ψψ̃⟩ → h(Z,Z)`, `ψγ₀ψ̃ → h(Z,Γ_a Z)`,
    `ψγ₃ψ̃ → −h(Z,Γ_a N Z)`, and the kinetic form `⟨γ^a φ Iγ₃ ψ̃⟩ → Re h(Z_ψ, I Γ^a Z_φ)`. So LDG's Dirac Lagrangian
    is `Re h(Z, I Γ(∂_a) D_a Z) − m h(Z,Z)`.
  - The dilation generator `N = n_o∧n_inf` is γ₅; the ω and π halves are the Weyl spinors.
    `P_a = −√2 I Γ_a Π₊`, `K_a = −√2 I Γ_a Π₋`.
  - Fierz holds: `J² = σ² + ω² = −A²`, `J·A = 0`.
  - Normalization: `S₁ = −2√g R + 12√g tr(f e⁻¹)`, `S₂ = −24k²√g`, `S₃ = −24√g` at `f = k e`. So LDG's `½R` is
    `−¼S₁` at `f = 0`, provided their R has the repo's sign (both give `R = −4Λ` for de Sitter from `G − Λ = 0`);
    the Phase 2 oracle confirms.
  - Decision data, option A (full twistor transport): in the action, the P sector adds `√2 σ` for any coframe. The K
    sector adds `(√2/4)(tr f σ + f^{ab} T_ab)`, with f in frame indices and `T_ab` the antisymmetric tensor bilinear
    `ψ̄σ_ab(1−γ₅)ψ`; a symmetric trace-free f does not couple. The D sector adds exactly zero. So A equals LDG plus:
    (i) a parity-even mass shift `m → m − √2(1 + ¼ tr f)`, made of the compensators, `√2(1 + 1/β)` on shell;
    (ii) Dirac sources in the K-sector equation, `σ` in the trace and `T_ab` in the antisymmetric part, so
    `f = e/β + O(κ ψ̄ψ)`. Eliminating f leaves `(ψ̄ψ)²` and `T_ab T^ab` contact terms that LDG do not have.
    Option B removes all of these.
- Phase 1: done (`dirac.density`, `dirac_operator`, `euler_lagrange_spinor`, `euler_lagrange_connection`;
  `tests/test_dirac_action.py`, 7 tests). The coframe comes from `DX` with its compensator components removed, and
  `which = "full" | "lorentz"` picks A or B. In flat space the spinor equation is Dirac's and plane waves solve it.
  On Schwarzschild with `f = k e` and a dilation field with curl, A equals B with `m → m − √2(1 + k)`, for the density
  and for the spinor equation. Both densities are invariant under local Lorentz transformations. First look at the
  connection variation (flat): the M-sector source is totally antisymmetric, built from ½ the axial current (LDG's
  `S·a`); A adds `√2σ` on the P diagonal and `(√2/4)(σ η + T)` in K; D gets nothing. Pinned in Phase 4.
- Phase 2: done (`tests/gtg_dirac_reference.py`, `tests/test_gtg_dirac_reference.py`, 16 tests). The oracle uses
  LDG's formulas in the STA with numpy matrices for h̄, and no CSTA machinery. Checks:
  - LDG's closed-form rotation gauge field (4.49, 4.53) is the repo's Levi-Civita connection to 1e-14, on
    Painlevé–Gullstrand, Kerr–Schild Kerr, the static lapse and de Sitter.
  - The Ricci scalar has the repo's sign (de Sitter `R = −4Λ`), so LDG's `½R` is `−¼S₁` at `f = 0`.
  - The oracle's Euler–Lagrange derivative of (3.56) is `2|det e|⟨b_A γ₀ R̃⟩`, with R the residual of LDG's (3.62)
    including the torsion-trace term. Holds on Kerr with τ = 0 and with a torsion that gives τ ≠ 0.
  - The oracle density equals `dirac.density(..., "lorentz")` through `from_sta`, to 1e-13, on PG and Kerr, with and
    without torsion and mass.
- Phase 3: done (`dirac.torsion_trace`, `dirac.spinor_equation`; `tests/test_dirac_equation.py`, 11 tests).
  - With the Lorentz coupling, the CSTA spinor equation is the oracle's blade by blade (1e-13), on PG and Kerr, with
    and without torsion. That is LDG's (3.62).
  - Closed form: `I Γ^a D_a Z + ½ I Γ(τ) Z = m Z`, with `δS/δc_k = 2|det e| Re h(U_k, W)`. The torsion trace
    `τ_b = −T^a_{ab}` uses `T(a,b) = −n_o⌋F_P(a,b)`, the curvature of the Lorentz and translation parts of Ω. It equals
    LDG's τ (3.65). The dilation part has to be left out: `[D, P] ∝ P` puts `b∧e` into `F_P`, and the Dirac field does
    not see b. Without torsion τ = 0 for any b, so the equation is minimally coupled.
  - Full transport: W gains `½(O + Õ)Z`, the h-Hermitian part (the reverse is the h-adjoint) of
    `O = ½ I Γ^a (Ω − Ω_M)(E_a)`. Exact for general f (antisymmetric part included), b with curl, and torsion. For
    `f = k e` it is the compensator mass `√2(1 + k) Z`.
- Phase 4: done (`dirac.spin_trivector`, `stress_energy`, `special_conformal_current`,
  `reciprocal_coordinate_frame`; `tests/test_dirac_sources.py`, 10 tests). The Dirac part of `δS/δΩ^k_μ`, sector by
  sector, on Kerr with torsion, general f and b with curl:
  - M (spin): `|det e| S·g^μ`, with `S = ½ψIγ₃ψ̃ = −½ I_sta A` and `g^μ = h̄(e^μ)`. This is LDG's spin tensor
    `S(a) = S·h̄(a)` (4.27, 4.37). The native S equals the oracle's, and the rows are the same under both transports.
  - P (stress-energy): with the Lorentz coupling, the oracle's `∂(det(h)⁻¹L)/∂e` (4.16, 4.20). In closed form
    `−|det e| E^μ_b T^b_a`, with `T(a) = ⟨a·Dψ Iγ₃ψ̃⟩₁ − aL`. Under the full transport D omits the P sector (its
    action `½Γ^a e(E_a)∧n_inf` does not depend on e) and L carries the compensator terms.
  - K: zero with the Lorentz coupling. With the full transport `|det e| E^μ_b C^b_a`, where C is the
    special-conformal current: its symmetric part is `(√2/4)σ η`, and the antisymmetric part is the tensor bilinear.
  - D: zero under both.
- Phase 5: done (`tests/test_spin_torsion.py`, 9 tests). The action is `S₁ + βS₂ + γS₃ + 4κ|det e|L_D`, which is
  LDG's `det(h)⁻¹(½R − κL_D)` times −4.
  - Levi-Civita plus LDG's contorsion `Ω(d_μ) += ½κ g_μ·S` (4.41, 4.53) solves the Lorentz rows exactly (1e-12), on PG
    and Kerr, under both transports. The opposite sign misses by more than 1.
  - The torsion is the spin density: `T(a,b) = −n_o⌋F_P(a,b) = κ(a∧b)·S` on the frame (LDG 4.37). It is traceless, so
    the spinor equation is minimally coupled (the consistency loop, 4.32–4.35).
  - Beyond GTG (full transport): `S₂ = −2√g((tr F)² − F^a_b F^b_a)` with `F = f e⁻¹`, and the special-conformal
    equation `12δ − 4β tr F δ + 4β Fᵀ + 4κC = 0` is solved exactly by
    `F^b_a = (1/β)(1 + κ tr C/3)δ^b_a − (κ/β)C^b_a`, with `tr C = √2σ`. So the Dirac field shifts f away from e/β by
    ψ̄ψ and by the tensor bilinear, and through f it shifts the compensator mass, giving a `(ψ̄ψ)²` contact term.
- Phase 6: done (`tests/test_consistency_loop.py`, 2 tests plus 1 slow). `⟨F∧F∧X∧𝓘⟩` sees only `F_M = R + (e∧f)`.
  Its `R∧R` part is the Euler density, so the Lorentz equation gains `α ε(T∧f − e∧Df)`.
  - Lorentz coupling: f ∝ e, so the term only rescales S₁'s. A rescaled LDG contorsion solves the Lorentz rows for
    α = 0.01 and 0.1 (1e-11), and the torsion stays traceless: the loop holds for every α. LDG's "only R" claim does
    not carry over to this action with their coupling.
  - Full transport: with the Dirac-shifted f of Phase 5 (exact to first order in α), the 24 Lorentz rows are solved
    for a constant contorsion at x on Kerr. τ = 6e-16 at α = 0, and τ/α = (−0.003, 0.287, 0.046, 0.215) at α = 1e-3,
    unchanged to 3% at 1e-2 and bending at 0.1. So the minimally coupled Dirac equation fails unless α = 0: in the
    full CSTA, Dirac matter fixes the Weyl-squared admixture to zero.
- Phase 7: done (RESULTS.md, DEEP_DIVE §5.9, §10, §11, README, the LaTeX formulation §13 with the PDF rebuilt).
  Next: a self-gravitating solution, docs/PLAN_DIRAC_SOLUTION.md.
