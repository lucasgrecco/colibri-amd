# HIP Setup Guide — Colibrì AMD GPU Backend

## Prerequisites

- **AMD GPU**: RX 7900 XTX (RDNA3, gfx1100) or compatible
- **ROCm 6.x**: Official AMD GPU compute stack
- **OS**: Ubuntu 22.04 or 24.04 (ROCm officially supported)

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
rocminfo
hipconfig --full
```

## 2. Build Colibrì with HIP

```bash
cd c

# Build with HIP for RX 7900 XTX
make glm HIP=1 HIP_ARCH=gfx1100

# Other common targets:
#   make glm HIP=1 HIP_ARCH=gfx1030   # RX 6900 XT (RDNA2)
#   make glm HIP=1 HIP_ARCH=gfx942    # MI300X (CDNA3)
#   make glm HIP=1 HIP_ARCH=gfx90a    # MI250X (CDNA2)
```

## 3. Run Tests

```bash
# Unit tests (requires GPU)
make hip-test HIP=1 HIP_ARCH=gfx1100

# Oracle validation (token-exact match)
python3 tools/make_glm_oracle.py
COLI_HIP=1 COLI_GPU=0 SNAP=./glm_tiny TF=1 ./glm 64 16 16
# Expected: "32/32 positions match"

COLI_HIP=1 COLI_GPU=0 SNAP=./glm_tiny ./glm 64 16 16
# Expected: "20/20 tokens match"

# Python integration tests
COLI_HIP=1 COLI_GPU=0 python3 -m pytest tests/test_hip_server.py -v
```

## 4. Run Inference

```bash
# Chat with HIP backend
COLI_HIP=1 COLI_GPU=0 COLI_MODEL=/path/to/glm52_i4 ./coli chat

# Serve API with HIP
COLI_HIP=1 COLI_GPU=0 COLI_MODEL=/path/to/glm52_i4 ./coli serve --host 127.0.0.1 --port 8000

# Benchmark
COLI_HIP=1 COLI_GPU=0 COLI_MODEL=/path/to/glm52_i4 ./coli bench
```

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLI_HIP` | off | Enable HIP backend |
| `COLI_GPU` | (empty) | HIP device ordinal (e.g., `0`) |
| `COLI_GPUS` | (empty) | Multi-GPU list (e.g., `0,1`) — Phase 2+ |
| `HIP_DENSE` | 0 | Upload dense tensors to GPU |
| `HIP_EXPERT_GB` | 0 | VRAM budget for experts (e.g., `20`) |
| `HIP_RELEASE_HOST` | 0 | Free host RAM after GPU upload |
| `COLI_HIP_PIPE` | 0 | Resident pipeline (prefill attention on GPU) |
| `COLI_HIP_PROFILE` | 0 | Enable H2D/kernel/D2H timing |
| `COLI_HIP_W4_PACKED` | 1 | Use W4A32 packed expert kernels |
| `COLI_HIP_DUAL_PROJ` | 1 | Fused gate+up projection |
| `COLI_HIP_ASYNC` | 1 | Async expert group transfers |

## 6. Troubleshooting

### "hipcc not found"
Install ROCm or set `HIPCC` to the full path:
```bash
make glm HIP=1 HIPCC=/opt/rocm/bin/hipcc
```

### "HIP requested backend is unavailable"
Check GPU visibility:
```bash
rocminfo | grep "Name"
```
Ensure the GPU is not in use by another process.

### "invalid COLI_GPUS"
Use comma-separated device ordinals:
```bash
COLI_HIP=1 COLI_GPUS=0,1 ./coli chat
```

### Performance tuning
- Start with `COLI_HIP_PROFILE=1` to see H2D/kernel/D2H timing
- Increase `HIP_EXPERT_GB` to cache more experts in VRAM
- Enable `HIP_DENSE=1` to keep dense tensors on GPU
- For prefill-heavy workloads, try `COLI_HIP_PIPE=1`
