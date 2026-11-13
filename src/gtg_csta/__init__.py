"""Gauge theory of gravity in the real conformal spacetime algebra Cl(2,4), on numga."""
import os

# On a CUDA build of JAX, XLA's Triton fusion of float64 dot products miscompiles when two extensor chains share
# subexpressions inside one `jit` (observed on an RTX 3060: the second of two jitted invariants came out ~1e-19
# instead of 4e-3, correct on CPU and with this flag). It must be set before JAX starts a backend.
_FLAG = "--xla_gpu_enable_triton_gemm=false"
if _FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _FLAG).strip()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
