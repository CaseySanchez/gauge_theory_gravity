# Results log

All checks below are executable: `source .venv/bin/activate && pytest` (121 tests, ~3.5 min). Demos:
`cd src && python -m gtg_csta.examples.scenarios` and `python -m gtg_csta.examples.render` (figures in `output/`).
Working algebra: `Algebra("t+x-y-z-p+n-")` = the spec's Cl(2,4), literally. Backend: numga on JAX (float64).

## Implementation: extensors, with an independent array reference

The package is written with numga extensors (`src/gtg_csta/`). The first implementation used raw coefficient
arrays and `einsum`; it is kept as `src/gtg_csta/_array/` and used as an independent oracle
(`tests/array_reference/`, and cross-checks in the main tests). Statements below were first verified with the
array version and re-verified on the extensor version; the two agree to 1e-12 or better.

What is an extensor now:
* Connection Ω : `Bivector <- Spacetime` (`Spacetime` = the 4-direction subspace of the 6 vectors); `Omega(a)` is binding.
* Curvature F : `Bivector <- (Spacetime, Spacetime)` = `swap(dΩ) - dΩ + Ω.commutator(Ω)`, with `dΩ = derivative(Ω)(x)`
  from numga's JAX `derivative` (a map with one more input, the step) and numga's `commutator` (= [,]/2, exactly the spec's ½[Ω,Ω]).
  Bianchi uses a small `permute_inputs` helper (numga has no axis permutation for multilinear maps).
* Gauge transformation `V >> Omega` (sandwich of a map's output) minus `2 dV ~V`; covariant derivative `dPsi + ½ Omega * Psi`.
* The sector split M/P/K/D is four extensor projectors `Bivector <- Bivector` built from the null basis by
  contractions and wedges (no 15x15 matrix); the connection is `Ω_L + e ^ n_∞ + f ^ n_o + b (n_o ^ n_∞)`.
* Coframe `Spacetime <- Spacetime`, metric `g = e | e`, frame `E = e.inverse()`, Christoffel from `derivative(g)`,
  spin connection `Omega = ½ (W * (S ^ S)).contract(1,4).contract(2,3)` (two metric contractions), Ricci as a
  contraction `(S | F(E, S)).contract(1,2)`, Schouten from it, the quadratic density by two `.contract`s.
* Still raw arrays: the `GEN` matrix (generator <-> blade coordinates, used only to *specify* test connections and in
  the structure-constant test), the 16-unknown least-squares for the normal conditions (`jax.jacfwd` + `lstsq`,
  assembled from extensor residuals), the Gauss-point scan over transport steps (a `lax.scan` on kernels), and the
  ray tracer's Hamiltonian flow. Nothing in numga's linear-algebra extensions (`solve`, `lstsq`, `eigh`) was needed except `inverse`.

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

## Renderings (`src/gtg_csta/examples/render.py`, figures `output/render_*.png`)

* **Field-strength slice** (`render_slices.png`): the gauge-invariant `<F F~>` on a plane through the hole, agrees with 24M²/r⁶ to 1e-11 down to r≈0.4; it diverges toward the singularity, with the horizon (r=2M) unremarkable in PG coordinates. The translation-sector shift √(2M/r) is shown as a gauge-dependent comparison.
* **Holonomy map** (`render_holonomy.png`): max |T_loop − 1| for 0.4×0.4 loops of the full conformal connection (coframe + Lorentz), ≈ ε²·curvature, from 1e-4 at r=8 to O(1) at the centre.
* **Lensed image** (`render_lensing.png`): backward null-geodesic ray tracing (Hamiltonian flow of the metric `e | e`) of a thin disk (r = 6–14M) and a checkered sky from a static camera at r=30M, 20° above the disk plane. Validation: the measured shadow angular radius is 0.1670 rad against the analytic 0.1681 (static observer, sin α = (3√3 M/D)√(1−2M/D)), 0.65% low at 160×120 pixels. It shows the shadow, the direct disk below and the lensed secondary image arching over the top.
* Rays that reach r < 2.2M when traced backward are classed black: in the PG chart, a backward ray hugging the horizon is continued onto an unphysical branch and returns to large r, which produced spurious speckle inside the shadow in the first version of this render (fixed).
* The first versions of the holonomy map and lensing image were wrong (as reported when they were made). Causes: (i) `jax.jit(jax.vmap(holonomy_loop))` returns wrong values for batches of more than about 16 points while `jax.vmap` without `jit`, `jax.lax.map` and single-point calls agree with each other (reproduced with the array and the extensor transport alike; the map now uses `lax.map`); I did not establish whether the fault is in numga, JAX or XLA. (ii) The camera lay in the disk plane, and the horizon speckle above.
* Not done: Kerr (frame dragging); no redshift/beaming; the disk is a plain colour ramp in radius.

## ParaView export of spacetime slices (`src/gtg_csta/export.py`)

Implemented, **not run**: the simulation itself (evaluating the fields on a 3D grid for each time step) has not been executed.
* Format: VTKHDF ImageData, one `.vtkhdf` file per time step plus a `<stem>.vtkhdf.series` JSON that ParaView reads as a time series. Needs `h5py` (writer) and, for the tests, `vtk`.
* Verified: the writer's layout, by reading files back with VTK's own `vtkHDFReader` (point ordering for scalars, 3-vectors and `uint8`, dimensions, origin and spacing), `tests/test_export.py`. ParaView itself was not opened.
* Written but not executed: the field evaluator and the slow test `-m slow` on a 3×3×3 grid (`pytest -m slow tests/test_export.py`).
* Geometries: `flyby` (Schwarzschild boosted to v=0.6, a vacuum solution; its Weyl²/2 is the boosted r⁻⁶ profile, Cotton and Ricci vanish) and `growing_mass` (PG form with M(t) from 1 to 2, a toy with matter, so the Cotton sector switches on while the mass changes).
* Fields per step: ½C² (normal connection), ½Riem² (Lorentz-only), R, magnitudes of the sectors M/P/K/D of the normal connection's curvature, the translation-sector shift vector, log10 versions of the invariants, and a validity mask.
* Cost, from a timing of 4000 points on CPU with `jit(vmap)`: about 0.65 ms/point, i.e. about 70 s per frame at n = 48 (110k points), about 20 min for 16 frames. Run: `PYTHONPATH=src python -m gtg_csta.export flyby --n 48 --frames 16`.
