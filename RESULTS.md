# Results log

All checks below are executable: `source .venv/bin/activate && pytest` (59 tests, ~2 min). Demos:
`cd src && python -m gtg_csta.examples.scenarios` (figures in `output/`).
Working algebra: `Algebra("t+x-y-z-p+n-")` = the spec's Cl(2,4), literally. Backend: numga on JAX (float64).

## Verified (Tier A, spec sections 2-8)

| Spec | Claim | Status | Evidence |
|---|---|---|---|
| 2.1 | e_t²=+1, e_{xyz}²=-1, e_+²=+1, e_-²=-1; 15 bivectors | verified | `test_algebra.py` |
| 3 | n_o²=n_∞²=0, n_o·n_∞=1; 15 = 6+4+4+1 generators span the bivectors | verified | `test_algebra.py` (GEN invertible) |
| 3 | so(2,4) sector structure: [M,M]⊂M, [M,P]⊂P, [M,K]⊂K, [M,D]=0, [P,P]=[K,K]=0, [D,P]∝P, [D,K]∝-K-type grading, [P,K]⊂M⊕D; Jacobi | verified | `test_structure_constants_so24` |
| 5 | V Ṽ = 1 for V = exp(B), random 15-component B | verified, 1e-9 | numga generic exp agrees with an independent scipy `expm` of the 32×32 left-regular representation |
| 5 | V and -V give the same adjoint action; adjoint action preserves grade-2 and the inner product | verified | `test_gauge_curvature.py` |
| 5.1 | Ω' = VΩṼ − 2(a·∇V)Ṽ stays bivector-valued | verified, 1e-11 | `test_spec_5_1...` |
| 6 | D'Ψ' = V DΨ | verified: JAX 1e-11; finite-difference 1e-7 | `test_spec_6...` |
| 7 | F = ∂Ω − ∂Ω + ½[Ω,Ω] is bivector valued, antisymmetric, covariant (F' = VFṼ) | verified | `test_gauge_curvature.py` |
| 7.1 | [D_μ,D_ν]Ψ = ½F_μνΨ (including the ½ in front of the commutator) | verified, 1e-10 | `test_spec_7_1...` |
| 7 | Bianchi: cyclic D_λF_μν = 0 with D X = ∂X + ½[Ω,X] | verified, 1e-9 | `test_bianchi_identity` |
| 7 | pure gauge Ω = −2(∂V)Ṽ is flat; generic Ω is not | verified | `test_pure_gauge_is_flat...` |
| 8 | transport T = P exp(−½∫Ω) is a versor (TT̃=1) | verified, 1e-10 | `test_transport.py` |
| 8 | pure-gauge transport = V(x2)V(x1)⁻¹ ; non-flat Ω has non-trivial holonomy | verified | `test_transport.py` |
| 8 | spec caveat: a general connection is not pure gauge | verified: holonomy ≠ 1 for the same class of fields that fail flatness | `test_pure_gauge_holonomy_trivial_but_generic_not` |

Derived (not stated in spec) and now pinned numerically:
* Parallel transport convention: Ψ(s) = T(s)Ψ(0), dT/ds = −½Ω(γ')T; later steps multiply on the left.
* Holonomy around the rectangle x0 → +μ → +ν → −μ → −ν of side ε: **T = exp(−½ ε² F_μν) + O(ε³)**.
  Same sign in three planes tested (μν = 01, 12, 03).
* Magnus-4 transport is 4th order; midpoint is 2nd order; 4th-order finite-difference stencil gives ~16× error drop per halving.
* Conformal action of versors on embedded points X = n_o + x − ½x² n_∞ (null cone preserved):
  * V = exp(c P_a): x → x + 2c e_a (translation, n_∞ fixed). So P_a really are translations.
  * V = exp(λ D): x → e^{−2λ} x (dilation).
  * V = exp(c K_a): fixes n_o, maps null cone to itself (special conformal).
  * V = exp(c M_ab): preserves x² (Lorentz).
  This confirms the spec's generator labelling (P ↔ e_a∧n_∞, K ↔ e_a∧n_o).

## Lorentz sector (spec 9.1)

* Restricting Ω to M_ab keeps F in the M sector (P=K=D = 0 to 1e-12), and Lorentz-rotor gauge transformations keep Ω in the sector.
* A generic Spin(2,4) transformation takes a Lorentz connection out of the sector (P, K, D components O(0.1)); so the Lorentz restriction is gauge-fixing, not a gauge-invariant statement. This is the spec's caveat, made concrete.
* Sign map to numga's own STA (`x+y+z+t-`, opposite sign): φ(M_ab) = −M_ab is a Lie-algebra homomorphism (structure constants differ by exactly −1).
* Schwarzschild (Painlevé–Gullstrand tetrad, Levi-Civita spin connection embedded as Ω in the Lorentz sector):
  F^{ab}_{μν} from the CSTA formula equals the Riemann tensor computed independently from Christoffel symbols (1e-9);
  ½[Ω,v] reproduces the Levi-Civita derivative of frame vectors; Ricci = 0 (1e-16); Kretschmann = 48M²/r⁶ (relative error 3e-15, r from 2.2 to 10).

## Tier B: translation sector, normal conformal connection (spec 9.2, 10)

* **Translation sector = coframe.** Put e^a_μ in the P coordinates, Levi-Civita ω in the M sector: F_P = 0 (torsion-free) for Schwarzschild, FRW and a generic static-lapse metric.
* **Normal conditions** (F_P = 0, F_D = 0, F_M trace-free) are always solvable pointwise for the K coefficients f (16 unknowns, rank 16), and the solution is
  **f^a_μ = P^a_μ, the Schouten tensor** ½(Ric − R g/6), exactly (ratio 1.0000000000000002 in both non-vacuum test spacetimes) in this normalisation. For Ricci-flat spacetimes f = 0.
* With the normal f: F_M = trace-free (Weyl-type) part, F_K = Cotton-type: zero for FRW (conformally flat), nonzero for the lapse metric, zero in vacuum.
* **Quadratic action density.** g^{μρ}g^{νσ}⟨F_μν F̃_ρσ⟩₀ of the full 15-component F equals **½ C_{abcd}C^{abcd}** (Weyl-squared) to 1e-9 for the three spacetimes; for the Lorentz-only connection it does not (it also sees Ricci). This confirms spec section 10's remark that a quadratic curvature action gives Weyl-squared gravity, at the level of the Lagrangian density.

## Not done / not established (be careful when citing)

* **No field equations, no compensator, no Einstein recovery.** Only the density of the quadratic action was checked. Varying it (Bach equations), adding a compensator field, and a symmetry-breaking mechanism were not attempted. Spec 11.4 remains open.
* **Displacement gauge field of GTG.** The translation sector was identified with an orthonormal coframe of a torsion-free metric, which is standard Cartan geometry. A check that GTG's fiducial-flat-background displacement field h(a) = coframe is a faithful reading of spec 9.2 and 11.2 was not made; Schwarzschild was done through the PG tetrad and Levi-Civita connection, not via GTG's h-and-rotor equations. The Phase-6 plan item "reproduce Doran–Lasenby Schwarzschild" is therefore met only through the equivalent tetrad route.
* **Dilation gauge field b** was set to zero everywhere in Tier B; its dynamics/gauge-fixing was not explored (spec 11.3).
* **Fields are callables on points, not 4D lattices** (deviation from Phase 3 option 1). This was enough for every identity and costs nothing in accuracy; lattice demos can be added later.
* **Fundamental versor fields (spec 11.1)** not explored beyond pure-gauge flatness.
* Phase 6's Lorentz-sector equivalence is kinematic: it verifies that F in the sector is the Riemann bivector map; no GTG dynamical equations were involved.

## Findings about the spec

* All formulas stated in sections 4-8 are mutually consistent as written (checked numerically, including all factors of ½).
* The only convention that needed pinning down was the holonomy orientation sign above, and the normalisation of P (translation by 2c per unit coefficient) and D (scale e^{−2λ}), both due to the ½ in the spin-representation sandwich.
