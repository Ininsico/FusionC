"""Insert a per-launch synchronization + progress print into a generated file, to find the
first kernel that faults. Usage: python tools/bisect_launch.py <file.cu>"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "driver"))
import fusionc_driver as d  # noqa: E402

src = Path(sys.argv[1])
work = src.parent
text = src.read_text(encoding="utf-8")

pattern = re.compile(r"( *if \(measure\) FX_CUDA_CHECK\(cudaEventRecord\(evStop\[(\d+)\], 0\)\);\n)")

def repl(match):
    index = match.group(2)
    return (match.group(1)
            + f'    FX_CUDA_CHECK(cudaDeviceSynchronize());\n'
            + f'    fprintf(stderr, "LAUNCH_OK {index}\\n");\n')

patched, count = pattern.subn(repl, text)
print("patched launches:", count)
out = work / "bisect.cu"
out.write_text(patched, encoding="utf-8")

ok, log = d.nvcc_compile(out, work / "bisect.exe", work / "bisect.nvcc.log")
print("compiled:", ok)
if not ok:
    print(log[:2000])
    raise SystemExit(1)

proc = subprocess.run([str(work / "bisect.exe"), "--inputs", str(work / "inputs"),
                       "--iters", "1", "--warmup", "0"], capture_output=True, text=True,
                      errors="replace")
lines = [l for l in proc.stdout.splitlines() if l.startswith("REGION") or l.startswith("TOTAL")]
print("stdout tail:", lines[-4:] if lines else "none")
print("stderr tail:")
for line in proc.stderr.splitlines()[-6:]:
    print("   ", line)
