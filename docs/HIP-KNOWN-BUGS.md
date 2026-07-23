# HIP Known Bugs — Colibrì AMD GPU Backend

This file tracks bugs discovered during HIP backend development and testing.
Each entry includes discovery date, ROCm version, GPU model, symptoms, root cause,
workaround, fix commit, and regression test.

---

## Template

```markdown
## BUG-XXX: short description
- **Descoberto:** YYYY-MM-DD
- **ROCm:** X.Y
- **GPU:** model
- **Sintoma:** what the user sees
- **Causa:** root cause analysis
- **Workaround:** temporary fix
- **Fix:** commit hash
- **Teste de regressão:** test file and function
```

---

## Active Bugs

_No bugs discovered yet. This file will be populated during testing._

---

## Resolved Bugs

_No bugs resolved yet._

---

## Debugging Tools

### layer_diff.py
Compare layer-by-layer output between CPU and HIP:
```bash
COLI_HIP=1 COLI_GPU=0 LAYER_DUMP=1 SNAP=./glm_tiny ./glm 64 16 16 2> hip_dump.txt
LAYER_DUMP=1 SNAP=./glm_tiny ./glm 64 16 16 2> cpu_dump.txt
python3 c/tools/layer_diff.py cpu_dump.txt hip_dump.txt
```

### hip_debug.h
Verbose HIP API logging:
```bash
# Build with debug checks
HIPFLAGS="-DHIP_DEBUG" make glm HIP=1

# Run — every HIP API call is checked and logged
COLI_HIP=1 COLI_GPU=0 ./glm 64 16 16
```

### Sanitizers
```bash
# Address Sanitizer
HIPFLAGS="-O1 -g -fsanitize=address -fno-omit-frame-pointer" make glm HIP=1
./tests/test_backend_hip

# Undefined Behavior Sanitizer
HIPFLAGS="-O1 -g -fsanitize=undefined" make glm HIP=1
./tests/test_backend_hip
```

### Profiling
```bash
# Enable H2D/kernel/D2H timing
COLI_HIP=1 COLI_GPU=0 COLI_HIP_PROFILE=1 ./glm 64 16 16
```
