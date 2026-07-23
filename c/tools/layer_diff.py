#!/usr/bin/env python3
"""layer_diff.py — Compare layer-by-layer output between CPU and GPU runs.

NOTE: This tool requires LAYER_DUMP support in the engine, which is not yet
implemented. It is included as a placeholder for Phase 2 debugging. When the
engine gains a LAYER_DUMP=1 mode that emits "LAYER <n> <name>: <values>" lines
to stderr, this script will parse and compare them.

Usage (when LAYER_DUMP is implemented):
  COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 LAYER_DUMP=1 SNAP=./glm_tiny ./colibri 64 16 16 2> gpu_dump.txt
  LAYER_DUMP=1 SNAP=./glm_tiny ./colibri 64 16 16 2> cpu_dump.txt
  python3 c/tools/layer_diff.py cpu_dump.txt gpu_dump.txt

Expected output:
  Layer 0 attn_out: max_diff=0.00000012 (OK)
  Layer 12 attn_out: max_diff=0.0034 (DIVERGED)
  ...
  Summary: 73/75 layers match within tolerance (1e-4)
"""
import re
import sys


def parse_dump(path):
    """Parse LAYER_DUMP output into {layer_name: [values]}.

    Expects lines like:
      LAYER <n> <name>: <float> <float> ...
    """
    layers = {}
    current_key = None
    current_values = []

    with open(path) as f:
        for line in f:
            m = re.match(r"LAYER\s+(\d+)\s+(\S+):\s*(.*)", line)
            if m:
                if current_key and current_values:
                    layers[current_key] = current_values
                current_key = f"layer_{m.group(1)}_{m.group(2)}"
                current_values = [float(x) for x in m.group(3).split()]
            elif current_key:
                try:
                    current_values.extend(float(x) for x in line.split())
                except ValueError:
                    pass

    if current_key and current_values:
        layers[current_key] = current_values

    return layers


def compare_layers(cpu, gpu, tolerance=1e-4):
    """Compare CPU and GPU layer outputs. Returns 0 if all match, 1 otherwise."""
    all_keys = sorted(set(cpu.keys()) | set(gpu.keys()))
    diverged = []
    matched = 0

    for key in all_keys:
        if key not in cpu:
            print(f"  {key}: missing from CPU dump")
            diverged.append(key)
            continue
        if key not in gpu:
            print(f"  {key}: missing from GPU dump")
            diverged.append(key)
            continue

        cv = cpu[key]
        gv = gpu[key]

        if len(cv) != len(gv):
            print(f"  {key}: size mismatch (CPU={len(cv)}, GPU={len(gv)})")
            diverged.append(key)
            continue

        max_diff = max((abs(c - g) for c, g in zip(cv, gv)), default=0.0)

        if max_diff > tolerance:
            idx = max(range(len(cv)), key=lambda i: abs(cv[i] - gv[i]))
            print(f"  {key}: DIVERGED at idx {idx} (max_diff={max_diff:.8f})")
            diverged.append(key)
        else:
            print(f"  {key}: OK (max_diff={max_diff:.8f})")
            matched += 1

    print(f"\nSummary: {matched}/{len(all_keys)} layers match within tolerance ({tolerance})")
    return 1 if diverged else 0


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    cpu_path, gpu_path = sys.argv[1], sys.argv[2]
    print(f"Loading CPU dump: {cpu_path}")
    cpu = parse_dump(cpu_path)
    print(f"Loading GPU dump: {gpu_path}")
    gpu = parse_dump(gpu_path)
    print(f"CPU layers: {len(cpu)}, GPU layers: {len(gpu)}")

    if not cpu and not gpu:
        print("WARNING: No LAYER_DUMP output found in either file.")
        print("The engine does not yet support LAYER_DUMP=1. This tool is a placeholder.")
        sys.exit(0)

    print()
    sys.exit(compare_layers(cpu, gpu))


if __name__ == "__main__":
    main()