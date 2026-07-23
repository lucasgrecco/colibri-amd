#!/usr/bin/env python3
"""layer_diff.py — Compare layer-by-layer output between CPU and HIP runs.

Usage:
  COLI_HIP=1 COLI_GPU=0 LAYER_DUMP=1 SNAP=./glm_tiny ./glm 64 16 16 2> hip_dump.txt
  LAYER_DUMP=1 SNAP=./glm_tiny ./glm 64 16 16 2> cpu_dump.txt
  python3 tools/layer_diff.py cpu_dump.txt hip_dump.txt

Output:
  Layer 0: attention output diverges (max_diff=0.00000012)
  Layer 12: attention output diverges (max_diff=0.0034)  ← first divergence
  ...
  Summary: 75/75 layers match within tolerance (1e-4)
"""
import re
import sys


def parse_dump(path):
    """Parse LAYER_DUMP output into {layer_name: tensor_values}."""
    layers = {}
    current_layer = None
    current_values = []

    with open(path) as f:
        for line in f:
            # Match: "LAYER 5 attn_out: 0.123 0.456 ..."
            m = re.match(r"LAYER\s+(\d+)\s+(\S+):\s*(.*)", line)
            if m:
                if current_layer and current_values:
                    layers[current_layer] = current_values
                layer_num = int(m.group(1))
                name = m.group(2)
                current_layer = f"layer_{layer_num}_{name}"
                current_values = [float(x) for x in m.group(3).split()]
            elif current_layer:
                # Continuation line
                try:
                    current_values.extend(float(x) for x in line.split())
                except ValueError:
                    pass

    if current_layer and current_values:
        layers[current_layer] = current_values

    return layers


def compare_layers(cpu, hip, tolerance=1e-4):
    """Compare CPU and HIP layer outputs."""
    all_keys = sorted(set(cpu.keys()) | set(hip.keys()))
    diverged = []
    matched = 0

    for key in all_keys:
        if key not in cpu:
            print(f"  {key}: missing from CPU dump")
            diverged.append(key)
            continue
        if key not in hip:
            print(f"  {key}: missing from HIP dump")
            diverged.append(key)
            continue

        cpu_vals = cpu[key]
        hip_vals = hip[key]

        if len(cpu_vals) != len(hip_vals):
            print(f"  {key}: size mismatch (CPU={len(cpu_vals)}, HIP={len(hip_vals)})")
            diverged.append(key)
            continue

        max_diff = 0.0
        max_idx = 0
        for i, (c, h) in enumerate(zip(cpu_vals, hip_vals)):
            diff = abs(c - h)
            if diff > max_diff:
                max_diff = diff
                max_idx = i

        if max_diff > tolerance:
            print(f"  {key}: diverges at idx {max_idx} (max_diff={max_diff:.8f}, "
                  f"CPU={cpu_vals[max_idx]:.6f}, HIP={hip_vals[max_idx]:.6f})")
            diverged.append(key)
        else:
            matched += 1

    total = len(all_keys)
    print(f"\nSummary: {matched}/{total} layers match within tolerance ({tolerance})")
    if diverged:
        print(f"Diverged layers: {', '.join(diverged)}")
        return 1
    return 0


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    cpu_path, hip_path = sys.argv[1], sys.argv[2]
    print(f"Loading CPU dump: {cpu_path}")
    cpu = parse_dump(cpu_path)
    print(f"Loading HIP dump: {hip_path}")
    hip = parse_dump(hip_path)
    print(f"CPU layers: {len(cpu)}, HIP layers: {len(hip)}")
    print()

    sys.exit(compare_layers(cpu, hip))


if __name__ == "__main__":
    main()
