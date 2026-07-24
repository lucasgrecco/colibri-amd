# HIP Setup Guide — Colibrì AMD GPU Backend

## Prerequisites

- **AMD GPU**: RX 7900 XTX (RDNA3, gfx1100) or compatible
- **ROCm 6.x**: Official AMD GPU compute stack
- **OS**: Linux (Ubuntu 22.04 or 24.04 recommended)

## 1. Install ROCm

```bash
# Ubuntu 24.04
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.x_amd64.deb
sudo apt install -y ./amdgpu-install_6.x_amd64.deb
sudo amdgpu-install -y --usecase=rocm,hip

# Add ROCm to PATH
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
rocminfo | grep -A5 "Agent 2"
hipcc --version
```

## 2. Build Colibrì with HIP

```bash
cd c

# Build with HIP for RX 7900 XTX (gfx1100)
make colibri HIP=1 HIP_ARCH=gfx1100

# Other common targets:
#   make colibri HIP=1 HIP_ARCH=gfx1030   # RX 6900 XT (RDNA2)
#   make colibri HIP=1 HIP_ARCH=gfx942      # MI300X (CDNA3)
#   make colibri HIP=1 HIP_ARCH=gfx90a      # MI250X (CDNA2)
#   make colibri HIP=1 HIP_ARCH=native      # Auto-detect local GPU

# Run unit tests on the GPU
make hip-test HIP=1 HIP_ARCH=gfx1100
```

## 3. Run Inference

The HIP build uses the same env vars as the CUDA backend — the HIP
build defines `-DCOLI_CUDA`, so all `#ifdef COLI_CUDA` code paths
are active and call the HIP runtime via `backend_gpu_compat.h`.

```bash
# Chat with HIP backend
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 \
  COLI_MODEL=/path/to/glm52_i4 ./coli chat

# Serve API with HIP
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 \
  COLI_MODEL=/path/to/glm52_i4 ./coli serve --host 127.0.0.1 --port 8000

# One-shot generation
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 CUDA_EXPERT_GB=14 \
  COLI_MODEL=/path/to/glm52_i4 ./coli run "What is 2+2?"
```

## 4. Oracle Validation (token-exact match)

```bash
# Create the oracle model (requires torch + transformers)
python3 tools/make_glm_oracle.py

# Teacher-forcing: expect "32/32 positions match"
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=./glm_tiny TF=1 ./colibri 64 16 16

# Greedy: expect "20/20 tokens match"
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=./glm_tiny ./colibri 64 16 16
```

## 5. Environment Variables

These are the same variables used by the CUDA backend — the HIP
build enables them via `-DCOLI_CUDA` through `backend_gpu_compat.h`.

| Variable | Default | Description |
|----------|---------|-------------|
| `COLI_CUDA` | off | Enable the GPU backend (HIP build uses this, not `COLI_HIP`) |
| `COLI_GPU` / `COLI_GPUS` | auto | Device selection: `0`, or `0,1` for multi-GPU |
| `CUDA_DENSE` | 0 | Place dense (non-expert) tensors on the GPU |
| `CUDA_EXPERT_GB` | 0 | VRAM budget (GB) for caching experts on the GPU |
| `CUDA_RELEASE_HOST` | auto (1 if >1 device) | Release host-side copies after GPU upload |
| `COLI_CUDA_ATTN` | off | Run S≤4 attention on the GPU |
| `COLI_CUDA_ATTN_SHARD` | off | Shard the KV-b tensor across GPUs |
| `COLI_CUDA_PIPE` | 0 | Resident pipeline (prefill attention on GPU) |
| `COLI_CUDA_PROFILE` | 0 | Emit H2D/kernel/D2H timing |
| `COLI_CUDA_TC_INT4` | 0 | INT4 tensor-core path (WMMA — **no-op on HIP**, rocWMMA has no `precision::s4`. Falls back to `quant_matmul`) |
| `COLI_CUDA_TC_W4A16` | 0 | W4A16 tensor-core path. **Ativo em HIP** via rocWMMA (FP16 16×16×16). Set `=1` to enable |
| `COLI_CUDA_TC_W4A16_MIN` | 16 | Min rows to dispatch W4A16 tensor cores |
| `COLI_CUDA_W4_PACKED` | 1 | Use W4A32 packed expert kernels |
| `COLI_CUDA_DUAL_PROJ` | 1 | Fused gate+up projection |
| `COLI_CUDA_ASYNC` | 1 | Async expert group transfers |
| `COLI_CUDA_RESID` | 0 | Expert-group results stay on device |

## 6. Troubleshooting

### "hipcc not found"
```bash
make colibri HIP=1 HIPCC=/opt/rocm/bin/hipcc
```

### "COLI_GPU(S) requires COLI_CUDA=1"
The HIP build defines `-DCOLI_CUDA` automatically. If you see this,
you are running a CPU-only binary. Rebuild with `make colibri HIP=1`.

### "relocation R_X86_64_32 ... can not be used when making a PIE object"
This was a build bug fixed by adding `-fPIE` to `HIPCCFLAGS` in the
Makefile. If you hit this, ensure your branch includes that fix.

### Performance tuning
- `COLI_CUDA_PROFILE=1` — see H2D/kernel/D2H timing breakdown
- `CUDA_EXPERT_GB=N` — cache more experts in VRAM (e.g. 14 for a 24 GB card)
- `CUDA_DENSE=1` — keep dense tensors on GPU
- `COLI_CUDA_PIPE=1` — resident pipeline for prefill-heavy workloads
