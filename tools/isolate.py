"""Compare every materialised intermediate of a generated translation unit against the oracle.

    python tools/isolate.py attention_mlp fused --sm-count 16

Rebuilds the translation unit, compiles it, runs it with --dump-all and reports the per-tensor
error, so a failing kernel can be identified without guessing.
"""
import subprocess
import sys
from pathlib import Path

import shutil

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "driver"))
import fusionc_driver as d  # noqa: E402

workload = sys.argv[1] if len(sys.argv) > 1 else "attention_mlp"
variant = sys.argv[2] if len(sys.argv) > 2 else "fused"
flag = "--fuse" if variant == "fused" else "--no-fusion"
sm_count = "16"
if "--sm-count" in sys.argv:
    sm_count = sys.argv[sys.argv.index("--sm-count") + 1]

spec = d.WORKLOADS[workload]
work = d.RESULTS / workload
work.mkdir(parents=True, exist_ok=True)
cu = work / f"isolate_{variant}.cu"
d.JAVA = d.find_java()

proc = d.run([str(d.JAVA), "-jar", str(d.BUILD / "fusionc.jar"), "--input", str(d.ROOT / spec["dsl"]),
              flag, "--sm-count", sm_count, "-o", str(cu), "--quiet"])
if proc.returncode != 0:
    print(proc.stderr)
    raise SystemExit("compiler failed")

ok, text = d.nvcc_compile(cu, work / "isolate.exe", work / "isolate.nvcc.log")
print("compiled:", ok)
if not ok:
    print(text[:4000])
    raise SystemExit(1)

dump = work / "dump"
shutil.rmtree(dump, ignore_errors=True)   # never let a stale tensor file look like a result
dump.mkdir(exist_ok=True)
run = subprocess.run([str(work / "isolate.exe"), "--inputs", str(work / "inputs"),
                      "--dump-all", str(dump), "--iters", "3", "--warmup", "1"],
                     capture_output=True, text=True, errors="replace")
print("run:", run.returncode)
if run.returncode != 0:
    print(run.stderr[:2000])
    raise SystemExit(1)

consts, tensors = d.read_dsl(d.ROOT / spec["dsl"])
data = {}
for name, s in tensors.items():
    path = work / "inputs" / f"{name}.bin"
    if path.exists():
        raw = np.fromfile(path, dtype=np.float16 if s["dtype"] == "fp16" else np.float32)
        data[name] = raw.reshape(tuple(s["shape"]))
expected, intermediates = d.ORACLES[spec["oracle"]](data, consts)

checks = [(f"{name}", intermediates[key]) for name, key in d.INTERMEDIATE_TENSORS[workload]]
checks.append(("out.mlp_b2", expected))
for name, exp in checks:
    files = list(dump.glob(f"{name}.bin")) or list(dump.glob(f"*{name}*.bin"))
    if not files:
        print(f"{name:<18} no dump file")
        continue
    raw = np.fromfile(files[0], dtype=np.float16)
    if raw.size != exp.size:
        print(f"{name:<18} size mismatch: dumped {raw.size}, oracle {exp.size}")
        continue
    got = raw.astype(np.float64).reshape(exp.shape)
    diff = np.abs(got - exp)
    row_spread = float(np.abs(got - got[0]).max())
    print(f"{name:<18} max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} "
          f"max_row_spread={row_spread:.4e}  (dumped {files[0].name})")
    if diff.max() > 1e-2:
        bad = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"{'':18} worst at index {bad}: got {got[bad]:.6f} expected {exp[bad]:.6f}")
        print(f"{'':18} got[0,:6]     {np.round(got[0, :6], 5)}")
        print(f"{'':18} expected[0,:6] {np.round(exp[0, :6], 5)}")
