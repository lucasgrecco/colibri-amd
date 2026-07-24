# HIP Known Bugs — Colibrì AMD GPU Backend

This file tracks bugs discovered during HIP backend development and testing.
The HIP backend shares source with CUDA via `backend_gpu_compat.h` — the
same `#ifdef COLI_CUDA` code path is compiled for both vendors.

---

## Template

```markdown
## BUG-XXX: short description
- **Found:** YYYY-MM-DD
- **ROCm:** X.Y
- **GPU:** model
- **Symptom:** what the user sees
- **Cause:** root cause analysis
- **Workaround:** temporary fix
- **Fix:** commit hash
- **Regression test:** test file and function
```

---

## Active Bugs

_No bugs discovered yet._

---

## Resolved Bugs

### BUG-001: PIE link error on gcc (hipcc object)
- **Found:** 2026-07-15
- **ROCm:** 6.16.13
- **GPU:** RX 7900 XTX (gfx1100)
- **Symptom:** `relocation R_X86_64_32 against '.rodata.str1.1' can not be used when making a PIE object`
- **Cause:** gcc defaults to `-fPIE`; hipcc compiled `backend_cuda.o` without `-fPIE`, causing a link incompatibility
- **Workaround:** `HIPCCFLAGS="... -fPIE" make colibri HIP=1`
- **Fix:** `c05b65b` — added `-fPIE` to `HIPCCFLAGS` in `c/Makefile`
- **Regression test:** `make colibri HIP=1 HIP_ARCH=gfx1100` (link must succeed)

### BUG-002: GPU float matmul diverges from CPU (non-token-exact) — partially resolved
- **Found:** 2026-07-15
- **ROCm:** 6.16.13
- **GPU:** RX 7900 XTX (gfx1100)
- **Symptom:** GLM-5.2 int4 model on GPU produces different tokens than CPU (not token-exact; e.g. 9/32 vs 11/32 on oracle)
- **Cause:** GPU float matmuls round differently than the CPU int8-dot (IDOT) kernels; same behavior documented in upstream issue #100. WMMA was disabled on HIP (`COLI_GPU_HAS_WMMA=0`)
- **Workaround:** Use FP32 (16-bit) for token-exact match, or accept quantization noise on int4
- **Fix:** Partially resolved in Phase 2. `COLI_CUDA_TC_W4A16=1` now activates the W4A16 tensor-core path via rocWMMA (FP16 16×16×16). The `grouped_s4_wmma` kernel (INT4, 8×8×32) remains a no-op because rocWMMA has no `precision::s4` — it falls back to `quant_matmul`. Measured: expert-matmul dropped 34% (98.6s → 65.4s)
- **Regression test:** `COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 COLI_CUDA_TC_W4A16=1 SNAP=./glm_tiny TF=1 ./colibri 64 16 16` (expect 32/32 on FP32)

### BUG-003: ./glm binary missing (renamed to colibri)
- **Found:** 2026-07-15
- **ROCm:** 6.16.13
- **GPU:** RX 7900 XTX (gfx1100)
- **Symptom:** `./glm: command not found` when running directly
- **Cause:** Upstream renamed `glm.c` → `colibri.c`; the build target is now `colibri$(EXE)`. `./glm` is only a phony alias inside `make`
- **Workaround:** Use `./colibri` directly, or `make colibri HIP=1` (not `make glm HIP=1`)
- **Fix:** N/A (upstream change)
- **Regression test:** N/A

---

## Debugging Tools

### hip_debug.h

HIP API error-checking macros. To activate, include the header in
`backend_cuda.cu` and build with `-DHIP_DEBUG`:

```bash
# Build with debug checks (after adding #include "hip_debug.h" to backend_cuda.cu)
HIPCCFLAGS="-O1 -g -fsanitize=address -fno-omit-frame-pointer -DHIP_DEBUG" \
  make colibri HIP=1 HIP_ARCH=gfx1100
```

**Note:** `hip_debug.h` is currently NOT included by `backend_cuda.cu`.
To use it, add `#include "hip_debug.h"` near the top of `backend_cuda.cu`
(replacing the existing `#include` of `backend_gpu_compat.h`), then wrap
critical HIP calls with `HIP_CHECK(...)`.

### Profiling

```bash
# Enable H2D/kernel/D2H timing breakdown
COLI_CUDA=1 COLI_GPU=0 COLI_CUDA_PROFILE=1 ./colibri 64 16 16
```

### GPU memory info

```bash
# Check VRAM usage during inference
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 CUDA_EXPERT_GB=14 ./colibri run "test"
# Watch for: [CUDA] resident set: N tensors, X.XX GB VRAM
```
