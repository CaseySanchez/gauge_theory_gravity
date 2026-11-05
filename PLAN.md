# Plan: Simulating Gauge Theory Gravity in Conformal Spacetime Algebra with `numga`

Sources: `gauge_theory_gravity_conformal_spacetime_algebra.md` (the "spec") and the matching PDF.
Library: `numga/` (vendored in this directory, v2.0.0a0, NumPy/JAX/PyTorch backends).

## 0. Goal and scope

Build a numerical test-bed for the kinematic structure in the spec: the Cl(2,4;R) algebra, the 15-component bivector connection Ω(a,x), versor gauge transformations, covariant derivative, curvature, and path-ordered parallel transport. Then probe the open questions (spec §9–§11): Lorentz-sector restriction, translation sector as coframe, Einstein-gravity recovery.

The spec says plainly that it is not yet a complete gravitational theory: there is no action or field equations. So the plan has two tiers:

- **Tier A (kinematics, fully specified):** implement and numerically verify every identity the spec states.
- **Tier B (dynamics, exploratory):** experiments on the open questions. Results are labelled "numerical evidence", not "recovery of GTG".

## 1. Findings from the initial survey

1. **Environment.** System `numpy` is broken (`undefined symbol: cblas_sgemm`). `numga` is not installed. Use a fresh venv and `pip install -e ./numga[linalg,jax,examples,test]`. Python 3.12 is present and numga needs ≥3.11.
2. **numga idioms** (from `numga/CLAUDE.md`, `docs/`):
   - Algebra from a signature string, e.g. `Algebra("x+y+z+t-")`.
   - Everything is an `Extensor`. Multivectors come from `NumpyContext(alg).multivector`.
   - Sandwich is `>>`, reverse is `~`, commutator is `.commutator`.
   - Orientation-aware types (`Rotor`, `Bivector`, ...) with `intrinsic`/`extrinsic` tracking. This is a source of friction for the conventions below.
   - `.field()` fields over sites and `jax` `derivative()` give a route to `a·∇`.
3. **Signature mismatch to resolve.** The spec uses η = diag(+,−,−,−,+,−), i.e. `e_t²=+1`, `e_{x,y,z}²=−1`, `e_+²=+1`, `e_-²=−1`. numga's own `STA` is `x+y+z+t-` (the opposite overall sign for spacetime). Cl(2,4) and Cl(4,2) are not isomorphic as real algebras, so the sign flip is not cosmetic. Decision needed in Phase 1 (see §6).
4. **Spec spot-checks already done by hand** (to be re-verified numerically in Phase 2):
   - n_o² = n_∞² = 0 and n_o·n_∞ = 1 for the given null basis. ✓
   - Gauge covariance D'Ψ' = V DΨ follows from Ω' = VΩṼ − 2(a·∇V)Ṽ with D = a·∇ + ½Ω. ✓
   - [D_a, D_b]Ψ = ½F(a,b)Ψ requires F to carry the ½[Ω(a),Ω(b)] term as written. ✓
5. **Exp/log in Cl(2,4).** `numga/extensions/invariant_decomposition.py` states that its closed-form exp/log covers fewer than 6D. For algebras with ≥2 positive and ≥2 negative directions the decomposition has complex parts, and numga falls back to the generic methods (`logexp.py`: quadratic/linear/Padé). A general 15-component bivector exp in 6D is therefore not closed-form. Plan: generic series/scaling-and-squaring, with a unit-norm check on the result.
6. **Spec subtleties to carry into code:**
   - Ω is a vector-argument-linear map a ↦ bivector, so it should be modelled as an extensor `Bivector <- Vector` (4 spacetime args → 15 conformal bivectors), not 4 loose bivectors.
   - The spec warns that a general connection is not pure gauge, so curvature-zero tests must use genuinely pure-gauge Ω = −2(∂V)Ṽ and a separate non-flat case.
   - The curvature formula only holds for constant, commuting a, b. For coordinate fields, use basis vectors e_μ, where the extra term −Ω([a,b]) vanishes.
   - The "translation" and "special conformal" sectors depend on which null vector gets which role. Fix the convention once, test it, and document it.

## 2. Project layout

```
gauge_theory_gravity/
  PLAN.md
  numga/                      # vendored library (unchanged)
  gtg_csta/
    __init__.py
    algebra.py                # Cl(2,4) construction, bases, null basis, generators
    connection.py             # Ω(a,x): storage, sector split/assemble, field on a grid
    versors.py                # Spin(2,4) elements, random/analytic versors, V·Ṽ=1 checks
    gauge.py                  # gauge transformation of Ω, of Ψ, of X (adjoint)
    covariant.py              # D_a Ψ, finite-difference and JAX derivative backends
    curvature.py              # F(a,b), sector decomposition, Bianchi identity
    transport.py              # path-ordered exponential, holonomy, pure-gauge test
    sta_embedding.py          # Cl(1,3) ⊂ Cl(2,4), Lorentz-sector restriction
    cartan.py                 # (Tier B) translation sector ↔ coframe, torsion
    dynamics.py               # (Tier B) candidate actions, field-equation residuals
    examples/                 # runnable scenarios (python -m ...)
  tests/
    test_algebra.py  test_gauge.py  test_curvature.py  test_transport.py ...
  notebooks_or_scripts/       # visual checks, plots
  RESULTS.md                  # running log of what was verified / what failed
```

Follow numga's pattern for examples: `core.py` (maths), `scenarios.py` (runnable), `render.py` (plots).

## 3. Phases

### Phase 1: Environment and algebra (kinematic base)

1. Create a venv, install numga editable with `linalg,jax,examples,test`. Run a targeted numga test (e.g. `tests/test_core_logexp.py`) to confirm the install. Never run the full suite without the flag documented in `numga/CLAUDE.md`.
2. In `algebra.py` build `Algebra("t+x-y-z-p+n-")` (the spec's Cl(2,4); decided in §6). Verify:
   - basis squares equal diag(+,−,−,−,+,−)
   - dim 64, bivector subspace dim 15
3. Build the null basis n_o, n_∞ and assert n_o²=n_∞²=0, n_o·n_∞=1.
4. Build the 15 generators: 6 `M_ab`, 4 `P_a = e_a∧n_∞`, 4 `K_a = e_a∧n_o`, 1 `D = n_o∧n_∞`. Assert they span the full bivector space (rank 15).
5. Compute the structure constants from commutators and check the so(2,4) relations by sector:
   - [M,M]⊂M, [D,P]∝P, [D,K]∝K, [P,K]⊂M⊕D, [P,P]=[K,K]=0
   - This pins down the P/K/D roles and normalisation (spec §3, §4 normalisation note).

**Exit:** all algebra facts in spec §2–§3 pass as unit tests.

### Phase 2: Versors and gauge transformations

1. `versors.py`: construct V = exp(−½B) or product of exponentials from random 15-component bivectors B. Use a generic exp (scaling-and-squaring or numga `exp`), not the <6D closed forms. Check:
   - V Ṽ = 1 to tolerance
   - V X Ṽ preserves grade for bivectors
   - V X Ṽ preserves the inner product
2. Check Spin(2,4) covering: V and −V give the same adjoint action.
3. `gauge.py`: implement
   - Ω'(a) = V Ω(a) Ṽ − 2 (a·∇V) Ṽ
   - Ψ' = VΨ
   - X' = V X Ṽ
4. Test that Ω' is still a pure bivector (grade-2 projection residual ≈ 0). This is the spec's "preserves bivector-valued nature" claim.

**Exit:** verified V Ṽ=1 and Ω' ∈ Cl², random V(x) fields on a grid.

### Phase 3: Derivative infrastructure

Pick the representation of fields once, because everything later depends on it.

- Option 1 (**default**): fields on a regular 4D lattice as numga batch/field axes; `a·∇` via central finite differences (second-order, optionally spectral for smooth periodic fields).
- Option 2: JAX backend with `numga.backend.jax.derivative` for exact autodiff of analytic fields (V(x), Ω(x) given in closed form). Use this for convergence-clean identity tests.

Deliver a single `partial(field, mu)` interface that both backends satisfy, plus tests against analytic derivatives.

### Phase 4: Covariant derivative and curvature

1. `covariant.py`: D_a Ψ = a·∇Ψ + ½ Ω(a)Ψ. Test gauge covariance D'_aΨ' = V D_aΨ with random V(x), Ω(x), Ψ(x). Expected error: O(h²) for finite differences, ~1e-12 for JAX.
2. `curvature.py`: F(a,b) = a·∇Ω(b) − b·∇Ω(a) + ½[Ω(a),Ω(b)], evaluated on basis vectors e_μ, e_ν (6 independent components F_μν, each a 15-component bivector).
3. Tests:
   - [D_μ,D_ν]Ψ = ½F_μν Ψ on random Ψ (spec §7.1)
   - F is a bivector (grade-2 closure)
   - covariance: F' = V F Ṽ under gauge transformations
   - F ≡ 0 for pure-gauge Ω = −2(∂V)Ṽ (flatness)
   - F ≠ 0 for a hand-made non-flat Ω
   - Bianchi identity D_[λ F_μν] = 0 in the adjoint sense
4. Sector decomposition (§7.2): project F onto Lorentz / translation / special-conformal / dilation using the Phase 1 generator basis. Record which sector curvatures mix under the Cartan commutator structure (e.g. F_translation looks like torsion; F_Lorentz picks up e∧f terms).

**Exit:** all of spec §5–§7 verified numerically, including the ½-factor consistency between D and F.

### Phase 5: Parallel transport

1. `transport.py`: V(x1,x2) = 𝒫 exp(−½∫_γ Ω) by stepwise product of short-step exponentials (ordered product, midpoint or Magnus-2 per step). Convergence study in step size.
2. Tests:
   - V Ṽ=1 along paths (bivector connection ⇒ spin group) (spec §8)
   - pure-gauge Ω ⇒ V(x1,x2) = V(x1)⁻¹…-type path independence (to fix the exact order/sign against the −½ convention)
   - non-flat Ω ⇒ nonzero holonomy around small loops, with holonomy ≈ exp(−½ F·Area) to leading order
   - reparametrisation invariance
3. Verify the spec statement that a general connection is not representable by one global V field: show non-trivial holonomy for the same Ω that fails the pure-gauge test.

### Phase 6: Recovering the STA/GTG sector (spec §9.1)

1. `sta_embedding.py`: embed Cl(1,3) as the sub-algebra generated by the four spacetime vectors. Restrict Ω to M_ab only (e=f=b=0).
2. Verify:
   - this sector closes under gauge transformations by Lorentz rotors
   - F reduces to the standard GTG Riemann bivector map, R(a∧b) = a·∇ω(b) − b·∇ω(a) + ω(a)×ω(b) with GTG's normalisation (convert ½ factors and signs between conventions)
   - non-closure: general commutators [M, P] etc. leak out of the sector, which shows the restriction alone is not Einstein gravity (spec §9.1 caveat)
3. Reproduce a known GTG solution as a sanity check. Choose **Schwarzschild in Newtonian/Doran–Lasenby gauge** (h(a)=a + (…) (a·e_r)e_r, with the rotor field L). Compute the Riemann bivector R(B) and verify vacuum Ricci = 0 numerically. This needs the GTG displacement field h(a), which does not exist in the spec's Ω alone; it is the bridge to Phase 7.

### Phase 7 (Tier B): Translation sector, coframe and dynamics

This phase is exploratory. Each subsection ends in a recorded outcome, including negative ones.

1. **Soldering/torsion.** Interpret the translation components e^a(a,x) as a coframe; compute F_translation = torsion T = de + ω∧e + (b-terms). Test whether GTG's torsion-free condition can be imposed as F_P = 0 and what it implies for ω.
2. **Displacement field relation.** Test the identification h(a) ↔ e^a(a,x) (spec §9.2, §11.2). Specify precisely what is imposed: normalisation, gauge for dilation and special-conformal parts (e.g. b=0, f solved from F constraints à la standard conformal Cartan "normal" condition, f = Schouten tensor).
3. **Normal-connection constraint.** Impose the normal conformal Cartan conditions (torsion-free, trace-free curvature). Verify that f^a becomes the Schouten tensor of the metric g = e·e, and that the curvature reduces to the Weyl tensor. This is the standard result and confirms that the numerics are correct.
4. **Action experiments.** Following spec §10:
   - quadratic action ∫ ⟨F F̃⟩ → Weyl-squared; check numerically that its field-equation residual vanishes on Schwarzschild (conformally flat is not needed, Bach-flat suffices) and on FRW
   - add a compensator (scalar φ with weight 1, fixed by gauge φ=const) and the Einstein–Hilbert-type term φ² R; check whether Schwarzschild solves the resulting equations
   - report that "Einstein recovered" only if both the action and the constraints are stated explicitly
5. Discretised field-equation residual evaluators via `numga.backend.jax.derivative` of a lattice action, so variational derivatives need no hand-derived EOMs.

### Phase 8: Demonstrations and write-up

Runnable scenarios (numga style, `python -m gtg_csta.examples.<name>`):

1. **Gauge orbits:** random V(x); show Ω → Ω' while F → V F Ṽ; invariant scalar ⟨F F̃⟩ constant.
2. **Pure-gauge vs curved connection:** holonomy around loops, heat map of ‖holonomy − 1‖ vs loop area.
3. **Sector decomposition plot:** F split into M, P, K, D parts for a curved example.
4. **Conformal transformations as versors:** show V in Spin(2,4) acting on null-vector/point images, including dilation (e^{λD}) and special conformal (e^{K_a}) maps. This checks the geometric meaning of the generator labels.
5. **Schwarzschild test** (Phase 6/7): curvature profile, geodesic or light-ray deflection using covariant derivative, compared with the analytic result.

`RESULTS.md` collects: every identity checked, tolerance achieved, and each open question with status ("verified", "contradicted", "needs an additional assumption").

## 4. Testing strategy

- pytest with fixed RNG seeds; tolerances 1e-12 (JAX float64 analytic) and ~h² convergence slope checks for finite differences.
- Each spec equation gets one test named for its section (`test_spec_5_1_gauge_transform_of_connection` ...).
- Property tests on random bivectors in all 15 components.
- Convergence plots: error vs grid spacing, error vs path step.
- No test relies on numga defaults for signs: the convention fixtures live in `algebra.py` and are asserted at import.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sign/signature convention mismatch (spec vs numga STA, reverse vs involute, intrinsic/extrinsic types) | Phase 1 convention tests; keep all convention choices in one module |
| Generic Cl(2,4) exp has no closed form in numga | Scaling-and-squaring with unit-norm re-projection; compare against matrix exponential of an explicit 8×8 complex / 4×4 SU(2,2)-type representation as an independent oracle |
| numga API alpha and changes | Pin a commit hash; vendor copy already in repo |
| Finite-difference noise hides gauge covariance errors | Use JAX autodiff and analytic fields for identity tests, lattices only for demos |
| Spec is incomplete for dynamics | Keep Tier A/B split; log every added assumption in `RESULTS.md` |
| Ω as map-on-vectors vs components confusion | One storage class; always exposes `Ω(a)` for a vector a and the component form ω^{AB}_μ |
| Performance of 64-dim products on grids | numga sparse/unrolled kernels; restrict to the 15-dim bivector subspace types; JAX jit |

## 6. Decisions needed from you

1. **Signature convention:** **Decided (user): working algebra is `Algebra("t+x-y-z-p+n-")`**, i.e. the spec's Cl(2,4) literally (p=e₊, n=e₋; n_o=(p+n)/√2, n_∞=(p−n)/√2). No sign translation is needed for the spec's formulas. The only mismatch is with numga's own `STA` (`x+y+z+t-`, opposite sign), which matters only when comparing against numga's relativity examples; handle that with an explicit overall-sign map in `sta_embedding.py`.
2. **Tier B depth:** stop after Phase 6 (kinematics plus Lorentz sector) or continue into dynamics (Phase 7)? Default: do Tier A fully, then Phase 7 items 1–3 (torsion, normal connection), and treat actions (item 4) as stretch.
3. **Backend:** NumPy only, or JAX as the primary for derivatives? Default: JAX for identity tests, NumPy for lattice demos.

## 7. Suggested order and rough effort

| Phase | Effort | Depends on |
|---|---|---|
| 1 algebra + env | 0.5 day | none |
| 2 versors + gauge | 0.5 day | 1 |
| 3 derivatives | 0.5 day | 1 |
| 4 D, F, tests | 1 day | 2, 3 |
| 5 transport | 1 day | 4 |
| 6 STA sector + Schwarzschild | 1–2 days | 4 |
| 7 Cartan/dynamics | 2–4 days (open-ended) | 4, 6 |
| 8 demos + write-up | 1 day | all |

First milestone: end of Phase 4, where every kinematic identity in the spec §5–§7 is numerically confirmed.
