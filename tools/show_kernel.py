import sys
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "isolate_fused.cu"
path = Path("results/attention_mlp") / name
lines = path.read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines) if "void k_attention_fused" in l)
# find the closing brace of the kernel
depth = 0
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0 and i > start:
        end = i
        break
for i in range(max(0, start - 4), end + 1):
    print(f"{i + 1:4d}  {lines[i]}")
